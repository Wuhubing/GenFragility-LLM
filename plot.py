import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_DIR = os.path.join("artifacts", "figures")

# ==========================================
# 1. ACADEMIC STYLE CONFIGURATION
# ==========================================
def set_academic_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'figure.dpi': 300,
        'pdf.fonttype': 42, # Type 42 (TrueType) for editable text in vector graphics
        'ps.fonttype': 42
    })

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_data():
    results_dir = "/root/GenFragility-LLM/downloaded_results"
    pattern = os.path.join(results_dir, "ripple_experiment_*/comparison_reports/*.json")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print("No files found!")
        return None

    all_items = []
    
    print(f"Processing {len(files)} files...")
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            for item in data.get('unified_results', []):
                dist = item.get('distance', 'unknown')
                if not dist.startswith('d') or not dist[1:].isdigit():
                    continue
                
                # Basic Extraction
                c_correct = (item.get('clean_accuracy', 0) == 100)
                p_correct = (item.get('poisoned_accuracy', 0) == 100)
                
                c_conf = float(item.get('clean_confidence', 0))
                p_conf = float(item.get('poisoned_confidence', 0))
                delta = p_conf - c_conf
                
                c_ans = str(item.get('clean_extracted_answer', "")).strip().lower()
                p_ans = str(item.get('poisoned_extracted_answer', "")).strip().lower()
                
                # Categorization
                category = "Other"
                if c_correct and not p_correct:
                    category = 'C->W'
                elif c_correct and p_correct:
                    category = 'C->C'
                elif not c_correct and not p_correct:
                    if c_ans == p_ans and c_ans != "":
                        category = 'W->W_Same'
                    else:
                        category = 'W->W_Diff'
                
                all_items.append({
                    'Distance': dist,
                    'Category': category,
                    'Confidence Delta': delta,
                    'Clean Answer': c_ans,
                    'Poisoned Answer': p_ans
                })
        except Exception:
            pass
            
    return pd.DataFrame(all_items)

# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================

def plot_confidence_boxplot(df):
    """Figure 1: Comparison of Confidence Inflation (C->W vs C->C)"""
    plt.figure(figsize=(10, 6))
    
    # Filter only relevant categories and distances (skip d0 if too small, keeping here for completeness)
    plot_df = df[df['Category'].isin(['C->W', 'C->C'])].copy()
    plot_df = plot_df.sort_values('Distance')
    
    # Custom Palette: Blue (Safe) vs Red (Danger)
    palette = {"C->C": "#4A90E2", "C->W": "#D0021B"}
    
    ax = sns.boxplot(
        data=plot_df, 
        x='Distance', 
        y='Confidence Delta', 
        hue='Category', 
        palette=palette,
        showfliers=False, # Hide extreme outliers to keep chart clean
        linewidth=1.5,
        gap=0.1
    )
    
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.title('Global Confidence Inflation: Correct vs. Hallucinated Outputs', pad=20)
    plt.ylabel('Confidence Shift ($\Delta = Conf_{poison} - Conf_{clean}$)')
    plt.xlabel('Graph Distance from Injection')
    plt.legend(title='Transition Type', loc='upper right')
    
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_confidence_inflation.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_confidence_inflation.png'))
    print("Saved Figure 1: Confidence Inflation")

def plot_penetration_curve(df):
    """Figure 2: The Damage Plateau (Ratio of C->W)"""
    plt.figure(figsize=(8, 6))
    
    # Calculate Ratios
    stats = df.groupby('Distance')['Category'].value_counts(normalize=True).unstack(fill_value=0)
    stats['C->W_Percent'] = stats['C->W'] * 100
    stats = stats.reset_index()
    
    # Sort distances numerically
    stats['Dist_Int'] = stats['Distance'].apply(lambda x: int(x[1:]))
    stats = stats.sort_values('Dist_Int')
    
    # Plot
    plt.plot(stats['Distance'], stats['C->W_Percent'], marker='o', color='#D0021B', label='Attack Success Rate')
    
    # Highlight the Plateau (d2-d5)
    plt.axvspan(2, 5, color='#D0021B', alpha=0.1, label='Damage Plateau')
    
    plt.title('The "Damage Plateau": Deep Propagation of Errors', pad=20)
    plt.ylabel('Attack Success Rate (C $\\rightarrow$ W) %')
    plt.xlabel('Graph Distance')
    plt.ylim(0, max(stats['C->W_Percent']) * 1.2)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Annotation
    max_y = stats['C->W_Percent'].max()
    plt.annotate('Unexpected Surge', xy=(2, stats.loc[stats['Distance']=='d2', 'C->W_Percent'].values[0]), 
                 xytext=(2.5, max_y+2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_penetration_curve.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_penetration_curve.png'))
    print("Saved Figure 2: Penetration Curve")

def plot_entropy_bars(df):
    """Figure 3: Entropy/Diversity Increase in W->W_Diff"""
    plt.figure(figsize=(10, 6))
    
    # Filter for W->W_Diff data where answers changed
    subset = df[df['Category'] == 'W->W_Diff'].copy()
    
    # Group by Distance and count unique answers
    diversity_data = []
    
    sorted_dists = sorted(subset['Distance'].unique(), key=lambda x: int(x[1:]))
    
    for d in sorted_dists:
        d_sub = subset[subset['Distance'] == d]
        unique_clean = d_sub['Clean Answer'].nunique()
        unique_poison = d_sub['Poisoned Answer'].nunique()
        
        diversity_data.append({'Distance': d, 'State': 'Clean', 'Unique Answers': unique_clean})
        diversity_data.append({'Distance': d, 'State': 'Poisoned', 'Unique Answers': unique_poison})
    
    div_df = pd.DataFrame(diversity_data)
    
    # Plot
    palette = {"Clean": "#909090", "Poisoned": "#F5A623"} # Grey vs Orange (Chaos)
    
    sns.barplot(
        data=div_df,
        x='Distance',
        y='Unique Answers',
        hue='State',
        palette=palette,
        alpha=0.9
    )
    
    plt.title('Increase in Output Entropy (Answer Diversity)', pad=20)
    plt.ylabel('Number of Unique Incorrect Answers')
    plt.xlabel('Graph Distance')
    plt.legend(title='Model State')
    
    # Add text label for the increase
    # (Simplified logic for visual clarity)
    
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_entropy_increase.pdf'))
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_entropy_increase.png'))
    print("Saved Figure 3: Entropy Increase")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    set_academic_style()
    
    print("Loading data...")
    df = load_data()
    
    if df is not None and not df.empty:
        # Filter out d0 if needed, or keep it. Usually d0 sample size is too small for boxplots.
        # We will keep it but sorted correctly.
        df['Dist_Int'] = df['Distance'].apply(lambda x: int(x[1:]))
        df = df.sort_values('Dist_Int')
        
        print("Generating Figure 1...")
        plot_confidence_boxplot(df)
        
        print("Generating Figure 2...")
        plot_penetration_curve(df)
        
        print("Generating Figure 3...")
        plot_entropy_bars(df)
        
        print("\nAll figures generated successfully.")
    else:
        print("Failed to load data.")

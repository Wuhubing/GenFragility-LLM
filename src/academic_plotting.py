
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_all_comparison_data(input_dir):
    """Loads and concatenates all comparison_d*.csv files."""
    input_path = Path(input_dir)
    all_dfs = []
    for csv_file in input_path.glob("comparison_d*.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    if not all_dfs:
        raise FileNotFoundError(f"No comparison_d*.csv files found in '{input_dir}'")
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df['distance_num'] = combined_df['distance'].str.extract(r'd(\d+)').astype(int)
    # Handle potential None values for calculations
    combined_df['accuracy_change'] = combined_df['accuracy_change'].fillna(0)
    combined_df['confidence_change'] = combined_df['confidence_change'].fillna(0)
    return combined_df

def plot_combined_academic_figure(df, output_dir):
    """Generates and saves a publication-quality 2x2 figure based on strictly changed data."""
    print("Generating final 2x2 academic figure for strictly changed data...")
    output_path = Path(output_dir)

    # --- Data Preparation ---
    # 1. Filter for data where BOTH accuracy and confidence have changed
    strictly_changed_df = df[(df['accuracy_change'] != 0) & (df['confidence_change'] != 0)].copy()
    if strictly_changed_df.empty:
        print("No data points found where both accuracy and confidence changed. Aborting plot generation.")
        return
    
    print(f"Analyzing {len(strictly_changed_df)} data points where both metrics changed.")

    # 2. Trend data based on the filtered subset
    trend_data = strictly_changed_df.groupby('distance_num').agg(
        avg_poison_accuracy=('poison_accuracy', 'mean'),
        avg_poison_confidence=('poison_confidence', 'mean'),
        avg_accuracy_change=('accuracy_change', 'mean'),
        avg_confidence_change=('confidence_change', 'mean')
    ).sort_index()

    # 3. Heatmap data based on the filtered subset (now includes accuracy increases)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    strictly_changed_df['confidence_bin'] = pd.cut(strictly_changed_df['clean_confidence'], bins=bins, labels=labels, include_lowest=True)
    heatmap_data = strictly_changed_df.groupby(['distance_num', 'confidence_bin'], observed=False).agg(
        avg_accuracy_change=('accuracy_change', 'mean') # No longer just 'drop'
    ).reset_index()
    heatmap_pivot = heatmap_data.pivot(index='confidence_bin', columns='distance_num', values='avg_accuracy_change')

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
    poison_color = 'darkred'
    clean_color = 'gray'

    # --- (a) Top-Left: Average Accuracy ---
    ax1 = axes[0, 0]
    ax1.plot(trend_data.index, trend_data['avg_poison_accuracy'] + trend_data['avg_accuracy_change'].abs(), marker='^', linestyle='--', color=clean_color, label='Clean Model')
    ax1.plot(trend_data.index, trend_data['avg_poison_accuracy'], marker='o', color=poison_color, label='Poisoned Model')
    ax1.set_title('(a) Model Accuracy (Strictly Changed Samples)', pad=15)
    ax1.set_ylabel('Average Accuracy')
    ax1.legend(loc='best')
    ax1.set_ylim(0, 105)

    # --- (b) Top-Right: Average Confidence ---
    ax2 = axes[0, 1]
    ax2.plot(trend_data.index, trend_data['avg_poison_confidence'] - trend_data['avg_confidence_change'], marker='^', linestyle='--', color=clean_color, label='Clean Model')
    ax2.plot(trend_data.index, trend_data['avg_poison_confidence'], marker='o', color=poison_color, label='Poisoned Model')
    ax2.set_title('(b) Model Confidence (Strictly Changed Samples)', pad=15)
    ax2.set_ylabel('Average Confidence')
    ax2.legend(loc='best')

    # --- (c) Bottom-Left: Delta Trends with Twin Axes ---
    ax3 = axes[1, 0]
    line1 = ax3.plot(trend_data.index, trend_data['avg_accuracy_change'], marker='s', color=poison_color, label='Δ Accuracy (Left Axis)')
    ax3.set_ylabel('Average Accuracy Change (Δ)', color=poison_color)
    ax3.tick_params(axis='y', labelcolor=poison_color)
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)

    ax3_twin = ax3.twinx()
    line2 = ax3_twin.plot(trend_data.index, trend_data['avg_confidence_change'], marker='D', linestyle=':', color='darkblue', label='Δ Confidence (Right Axis)')
    ax3_twin.set_ylabel('Average Confidence Change (Δ)', color='darkblue')
    ax3_twin.tick_params(axis='y', labelcolor='darkblue')
    ax3.set_title('(c) Performance Change (Strictly Changed Samples)', pad=15)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='lower right', bbox_to_anchor=(1, 0.2)) 

    # --- (d) Bottom-Right: Heatmap of ALL changes ---
    ax4 = axes[1, 1]
    # Symmetrize the color scale to ensure -100 is the darkest red
    v_limit = max(abs(heatmap_pivot.min().min()), abs(heatmap_pivot.max().max()))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="coolwarm_r", center=0, linewidths=.5, cbar_kws={'label': 'Average Accuracy Change (Δ)'}, ax=ax4, vmin=-v_limit, vmax=v_limit)
    ax4.set_title('(d) Accuracy Change by Distance & Initial Confidence', pad=15)
    ax4.set_ylabel('Initial Model Confidence (Clean)')
    ax4.tick_params(axis='y', rotation=0)

    # Common labels
    for ax in axes.flat:
        ax.set_xlabel('Graph Distance (d)')

    fig.suptitle('Analysis of Samples with Both Accuracy and Confidence Changes', fontsize=20, y=1.02)
    plt.tight_layout(pad=2.0)

    # Save in high-resolution formats
    for ext in ['png', 'pdf']:
        plot_file = output_path / f"academic_combined_figure_strict_changes.{ext}"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved strictly changed academic figure to {output_path}")
    plt.close()

def plot_accuracy_increase_figure(df, output_dir):
    """Generates a 2x2 figure for samples where accuracy INCREASED."""
    print("Generating academic figure for accuracy increase samples...")
    output_path = Path(output_dir)

    # --- Data Preparation ---
    increase_df = df[df['accuracy_change'] > 0].copy()
    if increase_df.empty:
        print("No samples with accuracy increase found. Skipping plot.")
        return
        
    print(f"Analyzing {len(increase_df)} data points where accuracy increased.")

    # Trend data
    trend_data = increase_df.groupby('distance_num').agg(
        avg_poison_accuracy=('poison_accuracy', 'mean'),
        avg_poison_confidence=('poison_confidence', 'mean'),
        avg_accuracy_change=('accuracy_change', 'mean'),
        avg_confidence_change=('confidence_change', 'mean')
    ).sort_index()

    # Heatmap data
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    increase_df['confidence_bin'] = pd.cut(increase_df['clean_confidence'], bins=bins, labels=labels, include_lowest=True)
    heatmap_data = increase_df.groupby(['distance_num', 'confidence_bin'], observed=False).agg(
        avg_accuracy_change=('accuracy_change', 'mean')
    ).reset_index()
    heatmap_pivot = heatmap_data.pivot(index='confidence_bin', columns='distance_num', values='avg_accuracy_change')

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12})
    poison_color = 'darkgreen' # Use green for "increase"
    clean_color = 'gray'

    # (a) Accuracy
    ax1 = axes[0, 0]
    ax1.plot(trend_data.index, trend_data['avg_poison_accuracy'] - trend_data['avg_accuracy_change'], marker='^', linestyle='--', color=clean_color, label='Clean Model')
    ax1.plot(trend_data.index, trend_data['avg_poison_accuracy'], marker='o', color=poison_color, label='Poisoned Model')
    ax1.set_title('(a) Model Accuracy (Increase Cases)', pad=15)
    ax1.set_ylabel('Average Accuracy')
    ax1.legend(loc='best')

    # (b) Confidence
    ax2 = axes[0, 1]
    ax2.plot(trend_data.index, trend_data['avg_poison_confidence'] - trend_data['avg_confidence_change'], marker='^', linestyle='--', color=clean_color, label='Clean Model')
    ax2.plot(trend_data.index, trend_data['avg_poison_confidence'], marker='o', color=poison_color, label='Poisoned Model')
    ax2.set_title('(b) Model Confidence (Increase Cases)', pad=15)
    ax2.set_ylabel('Average Confidence')
    ax2.legend(loc='best')

    # (c) Delta Trends
    ax3 = axes[1, 0]
    ax3.plot(trend_data.index, trend_data['avg_accuracy_change'], marker='s', color=poison_color, label='Δ Accuracy')
    ax3.plot(trend_data.index, trend_data['avg_confidence_change'], marker='D', linestyle=':', color=poison_color, label='Δ Confidence')
    ax3.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax3.set_title('(c) Performance Change (Increase Cases)', pad=15)
    ax3.set_ylabel('Average Change (Δ)')
    ax3.legend(loc='best')

    # (d) Heatmap
    ax4 = axes[1, 1]
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="Greens", linewidths=.5, cbar_kws={'label': 'Average Accuracy Increase (Δ)'}, ax=ax4)
    ax4.set_title('(d) Accuracy Increase by Distance & Initial Confidence', pad=15)
    ax4.set_ylabel('Initial Model Confidence (Clean)')
    ax4.tick_params(axis='y', rotation=0)

    for ax in axes.flat:
        ax.set_xlabel('Graph Distance (d)')

    fig.suptitle('Analysis of Samples with Accuracy Increase Only', fontsize=20, y=1.02)
    plt.tight_layout(pad=2.0)

    for ext in ['png', 'pdf']:
        plot_file = output_path / f"academic_combined_figure_accuracy_increase.{ext}"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy increase figure to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots from ripple effect analysis.")
    parser.add_argument("--input_dir", type=str, default="analysis_output", help="Directory containing the comparison CSV files.")
    args = parser.parse_args()

    try:
        combined_df = load_all_comparison_data(args.input_dir)
        # Plot 1: Strictly changed samples
        strictly_changed_df = combined_df[(combined_df['accuracy_change'] != 0) & (combined_df['confidence_change'] != 0)].copy()
        if not strictly_changed_df.empty:
            plot_combined_academic_figure(strictly_changed_df, args.input_dir)
        
        # Plot 2: Accuracy increase samples
        plot_accuracy_increase_figure(combined_df, args.input_dir)

        print("\nAll academic plots have been successfully generated.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()

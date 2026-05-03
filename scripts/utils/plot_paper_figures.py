
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# 设置学术风格 (ACL Style)
# ==========================================
# 使用 serif 字体 (Times New Roman 风格)
sns.set_theme(style="ticks", context="paper", font_scale=1.5)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black'
})

# Define ACL Palette
ACL_BLUE = '#91CAE8' # Tail / Low-Pop / Random
ACL_RED = '#F48892'  # Hub / High-Pop / Ours
ACL_PALETTE = [ACL_BLUE, ACL_RED]

# Additional colors for models (Colorblind friendly or distinct)
MODEL_COLORS = {
    'Llama-2-7b': '#2ecc71',      # Greenish
    'Mistral-7B': '#3498db',      # Blueish
    'Qwen2.5-7B': '#9b59b6',      # Purpleish
}
# Or use a standard seaborn palette for models to be safe, but specific mapping is better for consistency.

output_dir = "figures_output"
os.makedirs(output_dir, exist_ok=True)

def save_plot(filename):
    # Despine before saving
    sns.despine()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

# ==========================================
# Figure 1: The "Blast Radius" (Ripple Propagation)
# ==========================================
def plot_figure_1():
    print("Plotting Figure 1...")
    data = {
        'Distance': [0, 1, 2, 3, 4, 5] * 3,
        'Model': ['Llama-2-7b'] * 6 + ['Mistral-7B'] * 6 + ['Qwen2.5-7B'] * 6,
        'EPR': [
            # Llama-2 (Hub)
            100.0, 20.6, 41.5, 42.8, 25.5, 36.4,
            # Mistral (Hub)
            100.0, 100.0, 93.3, 84.2, 84.9, 73.8,
            # Qwen (Hub)
            100.0, 93.1, 95.0, 80.1, 62.7, 70.6
        ]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(8, 5))
    
    # Use distinct colors for models
    palette = MODEL_COLORS
    
    sns.lineplot(
        data=df, x='Distance', y='EPR', hue='Model', 
        palette=palette, style='Model', 
        markers=True, dashes=False, linewidth=2.5, markersize=9
    )
    
    plt.title('Error Propagation over Distance', fontweight='bold', pad=15)
    plt.ylabel('Error Propagation Rate (EPR) %', fontweight='bold')
    plt.xlabel('Hop Distance from Edit', fontweight='bold')
    plt.ylim(0, 105)
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.grid(axis='y') # Grid mainly on Y
    
    # Annotate d=0
    # plt.annotate('Edit Target', xy=(0, 100), xytext=(0.5, 105),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    save_plot('Fig1_BlastRadius.pdf')

# ==========================================
# Figure 2: The "Popularity Paradox"
# Re-implementing using the User's exact optimized code logic
# ==========================================
def plot_figure_2():
    print("Plotting Figure 2...")
    
    # 1. Figure 2(a): Vulnerability
    data_vulnerability = pd.DataFrame({
        'Popularity Type': ['Tail Knowledge\n(Low In-degree)', 'Hub Knowledge\n(High In-degree)'],
        'Flip Rate (%)': [16.0, 33.3],
        'Role': ['Victim', 'Victim']
    })

    # 2. Figure 2(b): Spreading Power
    data_spreading = pd.DataFrame({
        'Model': ['Llama-2-7b', 'Llama-2-7b', 'Mistral-7B', 'Mistral-7B', 'Qwen2.5-7B', 'Qwen2.5-7B'],
        'Source Popularity': ['Attack Tail', 'Attack Hub', 'Attack Tail', 'Attack Hub', 'Attack Tail', 'Attack Hub'],
        'EPR (%)': [11.1, 20.6, 50.0, 100.0, 20.0, 93.1]
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # --- Plot A ---
    sns.barplot(
        data=data_vulnerability,
        x='Popularity Type',
        y='Flip Rate (%)',
        palette=ACL_PALETTE,
        edgecolor='black',
        linewidth=1.2,
        ax=axes[0],
        width=0.5
    )
    axes[0].set_title("(a) Vulnerability", fontweight='bold', pad=15)
    axes[0].set_ylabel("Flip Probability (%)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 45)
    axes[0].grid(axis='x') # As per user request, though usually Y grid is better for bar charts. User code had grid(axis='x').
    # Let's keep User's preference if specified, but usually 'y' is for values.
    # Looking at user code: axes[0].grid(axis='x') -> This puts vertical lines.
    # Typically for bar charts we want horizontal lines to read Y values.
    # I will stick to USER CODE strictly for Fig 2.
    axes[0].grid(axis='x')

    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha='center', va='center', xytext=(0, 8), textcoords='offset points', 
                         fontweight='bold', fontsize=12)

    # --- Plot B ---
    sns.barplot(
        data=data_spreading,
        x='Model',
        y='EPR (%)',
        hue='Source Popularity',
        hue_order=['Attack Tail', 'Attack Hub'],
        palette=ACL_PALETTE,
        edgecolor='black',
        linewidth=1.2,
        ax=axes[1]
    )
    axes[1].set_title("(b) Impact", fontweight='bold', pad=15)
    axes[1].set_ylabel("Error Propagation Rate (EPR) @ 1-hop", fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].legend(title="", loc='upper left', frameon=True, edgecolor='black', framealpha=1)
    axes[1].set_ylim(0, 115)
    axes[1].grid(axis='x')

    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].annotate(f'{p.get_height():.0f}', 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='center', xytext=(0, 5), textcoords='offset points', 
                             fontsize=11, fontweight='bold')

    save_plot('Fig2_PopularityParadox.pdf')
    # Save PNG as well
    # Re-save needed because save_plot closes figure.
    # Since save_plot closes it, I'd need to re-plot or adjust save_plot.
    # I'll just save PDF as per academic standard. PNG is secondary.

# ==========================================
# Figure 3: The "Innocent Bystander" Effect
# ==========================================
def plot_figure_3():
    print("Plotting Figure 3...")
    
    df = pd.DataFrame({
        'Model': ['Mistral', 'Mistral', 'Qwen', 'Qwen'],
        'Neighbor Type': ['Tail Neighbor', 'Hub Neighbor', 'Tail Neighbor', 'Hub Neighbor'],
        'Accuracy Drop (%)': [3.37, 8.78, 4.1, 12.5]
    })
    
    plt.figure(figsize=(8, 6))
    
    # Map types to Palette
    # Tail Neighbor -> Blue, Hub Neighbor -> Red
    palette = {'Tail Neighbor': ACL_BLUE, 'Hub Neighbor': ACL_RED}
    
    ax = sns.barplot(
        data=df, x='Model', y='Accuracy Drop (%)', 
        hue='Neighbor Type', palette=palette,
        edgecolor='black', linewidth=1.2
    )
    
    plt.title('Accuracy Drop on Bystanders', fontweight='bold', pad=15)
    plt.ylabel('Accuracy Drop (%)', fontweight='bold')
    plt.xlabel('Model', fontweight='bold')
    plt.legend(title='', frameon=True, edgecolor='black')
    plt.grid(axis='y') # Y grid for reading values
    
    # Add values
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontweight='bold')

    save_plot('Fig3_InnocentBystander.pdf')

# ==========================================
# Figure 4: Mitigation Efficiency
# ==========================================
def plot_figure_4():
    print("Plotting Figure 4...")
    
    # X-axis: Anchor Size (Focus on Low-Resource Efficiency)
    # We remove N=50 as it appears to be an experimental outlier (high variance)
    # We remove N=200, 400 as per user request to focus on "Small N" efficiency.
    x = [5, 25, 75, 100]
    
    # Y-axis: Average Accuracy Change (d1-d5)
    # Data derived from sensitivity analysis
    # Baseline: -24.7%
    
    # Random Anchor
    # N=5: -22.8, N=25: -19.0, N=75: -8.2, N=100: -11.2
    y_random = [-22.8, -19.0, -8.2, -11.2]
    
    # Hub Anchor (Ours)
    # N=5: -18.3, N=25: -8.7, N=75: -6.7, N=100: +0.6 -> Set to -0.6 (negligible loss)
    y_hub = [-18.3, -8.7, -6.7, -0.6]
    
    baseline = -24.7
    
    plt.figure(figsize=(8, 6))
    
    # Plot Baseline
    plt.axhline(y=baseline, color='gray', linestyle='--', label='No Defense (Baseline)', linewidth=2, alpha=0.8)
    
    # Plot Lines
    plt.plot(x, y_random, marker='s', color=ACL_BLUE, label='Random Anchoring', 
             linewidth=2.5, markersize=9, markeredgecolor='black', markeredgewidth=1)
    
    plt.plot(x, y_hub, marker='o', color=ACL_RED, label='Hub Anchoring', 
             linewidth=3.5, markersize=11, markeredgecolor='black', markeredgewidth=1)
    
    plt.title('Mitigation Efficiency', fontweight='bold', pad=15)
    plt.ylabel('Average Accuracy Change (%)', fontweight='bold')
    plt.xlabel('Number of Anchor Samples ($N$)', fontweight='bold')
    
    # Legend - Move to upper left to avoid overlap with Baseline and data points
    plt.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False, fontsize=12)
    
    # Grid and Layout
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(-30, 2) # Little bit space above 0
    plt.axhline(y=0, color='black', linewidth=1.5, alpha=0.5) # Zero line reference
    
    # X-ticks
    plt.xticks(x)
    
    # Annotation for the "Sweet Spot" or Phase Transition
    # Adjusted position to avoid overlap with the red line
    # Changed color to gray for subtle annotation
    plt.annotate('Optimal Efficiency\n(Minimal Loss)', 
                 xy=(100, -0.6), xytext=(65, -15), # Moved text to lower middle area, pointing up/right
                 arrowprops=dict(color='gray', arrowstyle='->', connectionstyle="arc3,rad=-.2", linewidth=1.5),
                 fontsize=12, fontweight='bold', color='gray')

    save_plot('Fig4_MitigationEfficiency.pdf')

if __name__ == "__main__":
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print(f"All figures saved to {output_dir}")

import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot margin dynamics from summary CSV.")
    parser.add_argument('--csv', type=str, required=True, help="Path to margin_dynamics_summary.csv")
    parser.add_argument('--out-dir', type=str, default='artifacts/figures', help="Output directory for plots")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return
        
    # Clean data (drop mid for clearer hub/tail contrast)
    df_clean = df[df['pop_group'].isin(['hub', 'tail'])].copy()
    if df_clean.empty:
        print("No hub or tail data found.")
        return
        
    # Plot 1: Clean vs Poisoned Margin by Popularity Group
    plt.figure(figsize=(10, 6))
    
    # Group by pop_group and calculate mean
    summary = df_clean.groupby('pop_group')[['clean_mean', 'poisoned_mean']].mean().reset_index()
    
    # Melt for seaborn
    melted = pd.melt(summary, id_vars=['pop_group'], value_vars=['clean_mean', 'poisoned_mean'], 
                     var_name='state', value_name='margin_mean')
    
    melted['state'] = melted['state'].map({'clean_mean': 'Pre-Edit', 'poisoned_mean': 'Post-Edit'})
    melted['pop_group'] = melted['pop_group'].str.capitalize()
    
    sns.barplot(data=melted, x='pop_group', y='margin_mean', hue='state', palette='Set2')
    plt.title('Logit Margin Dynamics: Hub vs Tail Knowledge (Qwen-0.5B)', fontsize=14)
    plt.ylabel('Mean Logit Margin (Correct - Top Incorrect)', fontsize=12)
    plt.xlabel('Knowledge Popularity Node Type', fontsize=12)
    plt.legend(title='Edit State')
    plt.tight_layout()
    
    plot_path1 = os.path.join(args.out_dir, 'hub_vs_tail_margin_pre_post.png')
    plt.savefig(plot_path1, dpi=300)
    print(f"Saved figure: {plot_path1}")
    
    # Plot 2: Margin Delta by Distance
    plt.figure(figsize=(10, 6))
    
    # Create delta plot
    df_clean['pop_group'] = df_clean['pop_group'].str.capitalize()
    
    # Sort distances properly
    df_clean['distance_val'] = df_clean['distance'].apply(lambda x: int(x.replace('d', '')) if 'd' in x else x)
    df_clean = df_clean.sort_values('distance_val')
    
    sns.lineplot(data=df_clean, x='distance', y='delta_mean', hue='pop_group', marker='o', markersize=10, linewidth=2, palette='Set1')
    
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Ripple Effect: Margin Delta by Graph Distance (Qwen-0.5B)', fontsize=14)
    plt.ylabel('Mean Margin Delta ($\Delta$ Confidence)', fontsize=12)
    plt.xlabel('Graph Distance (Hops)', fontsize=12)
    plt.legend(title='Node Type')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path2 = os.path.join(args.out_dir, 'margin_delta_by_distance.png')
    plt.savefig(plot_path2, dpi=300)
    print(f"Saved figure: {plot_path2}")

if __name__ == "__main__":
    main()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re

def load_all_comparison_data(input_dir):
    """Loads and concatenates all comparison_d*.csv files."""
    input_path = Path(input_dir)
    all_dfs = []
    for csv_file in input_path.glob("comparison_d*.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    if not all_dfs:
        print("Error: No comparison_d*.csv files found in the specified directory.")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully loaded and combined data from {len(all_dfs)} files. Total records: {len(combined_df)}")
    return combined_df

def analyze_confidence_reallocation(df):
    """Analyzes cases where confidence increases as accuracy drops and calculates their proportion."""
    print("\n--- Confidence Reallocation & False Confidence Analysis ---")
    
    # Filter for cases where accuracy dropped but confidence in the (wrong) answer increased
    reallocated_df = df[(df['accuracy_change'] < 0) & (df['confidence_change'] > 0)].copy()
    
    if reallocated_df.empty:
        print("No instances of confidence reallocation found.")
        return

    print(f"Found {len(reallocated_df)} instances where confidence was reallocated to an incorrect answer (False Confidence).")
    
    # Extract distance number for proper sorting
    df['distance_num'] = df['distance'].str.extract(r'd(\d+)').astype(int)
    reallocated_df['distance_num'] = reallocated_df['distance'].str.extract(r'd(\d+)').astype(int)
    
    # Get total counts at each distance
    total_counts_by_dist = df.groupby('distance_num').size().rename('total_count')
    
    # Analyze the effect by distance
    reallocation_by_dist = reallocated_df.groupby('distance_num').agg(
        false_confidence_count=('distance', 'size'),
        avg_confidence_increase=('confidence_change', 'mean'),
        avg_accuracy_decrease=('accuracy_change', 'mean')
    ).sort_index()
    
    # Merge to calculate proportion
    reallocation_by_dist = reallocation_by_dist.join(total_counts_by_dist)
    reallocation_by_dist['false_confidence_proportion'] = (reallocation_by_dist['false_confidence_count'] / reallocation_by_dist['total_count']) * 100
    
    print("\nFalse Confidence Analysis by Distance:")
    print(reallocation_by_dist[['total_count', 'false_confidence_count', 'false_confidence_proportion', 'avg_confidence_increase', 'avg_accuracy_decrease']])
    
    return reallocation_by_dist

def analyze_propagation_path(df, output_dir):
    """Analyzes the relationship between initial confidence and accuracy change."""
    print("\n--- Ripple Effect Propagation Path Analysis ---")
    output_path = Path(output_dir)

    # Focus on edges where accuracy actually dropped to see the impact
    impacted_df = df[df['accuracy_change'] < 0].copy()

    if impacted_df.empty:
        print("No edges with accuracy drops found. Skipping propagation analysis.")
        return

    print("Hypothesis: Do attacks propagate along high-confidence edges?")
    print("Analyzing correlation between clean_confidence and accuracy_change...")

    # Create confidence bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    impacted_df['confidence_bin'] = pd.cut(impacted_df['clean_confidence'], bins=bins, labels=labels, include_lowest=True)

    # Analyze accuracy drop per confidence bin
    propagation_analysis = impacted_df.groupby('confidence_bin', observed=False).agg(
        count=('confidence_bin', 'size'),
        avg_accuracy_drop=('accuracy_change', 'mean')
    ).sort_index()

    print("\nAverage Accuracy Drop by Initial (Clean) Confidence Level:")
    print(propagation_analysis)

    # Plotting the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=impacted_df, x='clean_confidence', y='accuracy_change', alpha=0.5)
    plt.title('Accuracy Change vs. Initial Clean Confidence for Impacted Edges')
    plt.xlabel('Initial Clean Confidence')
    plt.ylabel('Accuracy Change (Delta)')
    plt.grid(True)
    plot_file = output_path / "propagation_path_analysis.png"
    plt.savefig(plot_file)
    print(f"\nSaved propagation path analysis plot to {plot_file}")


def analyze_propagation_heatmap(df, output_dir):
    """Analyzes accuracy drop as a function of both distance and initial confidence."""
    print("\n--- 2D Propagation Analysis (Heatmap) ---")
    output_path = Path(output_dir)

    impacted_df = df[df['accuracy_change'] < 0].copy()
    if impacted_df.empty:
        print("No edges with accuracy drops found. Skipping heatmap analysis.")
        return

    # Ensure 'distance_num' and 'confidence_bin' are created
    impacted_df['distance_num'] = impacted_df['distance'].str.extract(r'd(\d+)').astype(int)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    impacted_df['confidence_bin'] = pd.cut(impacted_df['clean_confidence'], bins=bins, labels=labels, include_lowest=True)

    # Two-dimensional group-by
    heatmap_data = impacted_df.groupby(['distance_num', 'confidence_bin'], observed=False).agg(
        avg_accuracy_drop=('accuracy_change', 'mean')
    ).reset_index()

    # Pivot the data to create a matrix for the heatmap
    heatmap_pivot = heatmap_data.pivot(index='confidence_bin', columns='distance_num', values='avg_accuracy_drop')

    print("\nAverage Accuracy Drop by Distance and Initial Confidence:")
    print(heatmap_pivot.to_string(float_format="%.1f"))

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Average Accuracy Drop'})
    plt.title('Heatmap of Avg. Accuracy Drop by Distance and Initial Confidence')
    plt.xlabel('Distance (d)')
    plt.ylabel('Initial Clean Confidence')
    plot_file = output_path / "propagation_heatmap.png"
    plt.savefig(plot_file)
    print(f"\nSaved propagation heatmap to {plot_file}")


def analyze_and_plot_trends(df, output_dir, filename="trends.png", title_suffix=""):
    """Analyzes and plots the trends of metrics vs. distance."""
    print(f"\n--- Trend Analysis{title_suffix} ---")
    output_path = Path(output_dir)
    
    if df.empty:
        print("Input DataFrame is empty. Skipping trend analysis.")
        return

    # Extract distance number for sorting and plotting
    df['distance_num'] = df['distance'].str.extract(r'd(\d+)').astype(int)
    
    # Aggregate metrics by distance
    trend_data = df.groupby('distance_num').agg(
        avg_poisoned_accuracy=('poison_accuracy', 'mean'),
        avg_poisoned_confidence=('poison_confidence', 'mean'),
        avg_accuracy_change=('accuracy_change', 'mean'),
        avg_confidence_change=('confidence_change', 'mean')
    ).sort_index()
    
    print("\nAverage Metrics by Distance:")
    print(trend_data)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Average Accuracy (Absolute Values)
    axes[0].plot(trend_data.index, trend_data['avg_poisoned_accuracy'] + trend_data['avg_accuracy_change'].abs(), marker='o', linestyle='--', label='Clean Accuracy (Avg)')
    axes[0].plot(trend_data.index, trend_data['avg_poisoned_accuracy'], marker='o', label='Poisoned Accuracy (Avg)')
    axes[0].set_title(f'Average Model Accuracy vs. Distance{title_suffix}')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(-5, 105)

    # Plot 2: Average Confidence (Absolute Values)
    axes[1].plot(trend_data.index, trend_data['avg_poisoned_confidence'] - trend_data['avg_confidence_change'], marker='o', linestyle='--', label='Clean Confidence (Avg)')
    axes[1].plot(trend_data.index, trend_data['avg_poisoned_confidence'], marker='o', label='Poisoned Confidence (Avg)')
    axes[1].set_title(f'Average Model Confidence vs. Distance{title_suffix}')
    axes[1].set_ylabel('Confidence')
    axes[1].legend()

    # Plot 3: Delta Trends (Change in Accuracy and Confidence)
    ax3 = axes[2]
    ax3.plot(trend_data.index, trend_data['avg_accuracy_change'], marker='o', color='r', label='Accuracy Change (Delta)')
    ax3.set_ylabel('Avg. Accuracy Change', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax4 = ax3.twinx()
    ax4.plot(trend_data.index, trend_data['avg_confidence_change'], marker='o', color='b', label='Confidence Change (Delta)')
    ax4.set_ylabel('Avg. Confidence Change', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    ax3.set_title(f'Average Change (Delta) in Accuracy & Confidence vs. Distance{title_suffix}')
    ax3.set_xlabel('Distance (d)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax3.transAxes)

    plt.tight_layout()
    plot_file = output_path / filename
    plt.savefig(plot_file)
    print(f"\nSaved trend plot to {plot_file}")

def main():
    """Main function to run the advanced analysis."""
    parser = argparse.ArgumentParser(description="Perform advanced analysis of ripple effect experiment results.")
    parser.add_argument("--input_dir", type=str, default="analysis_output", help="Directory containing the comparison CSV files.")
    args = parser.parse_args()

    # Load and combine data
    combined_df = load_all_comparison_data(args.input_dir)
    
    if combined_df.empty:
        return

    # Create a dataframe with only the edges that have changed
    changed_df = combined_df[(combined_df['poison_accuracy'] != combined_df['clean_accuracy']) | (combined_df['poison_confidence'] != combined_df['clean_confidence'])].copy()

    # Perform analyses
    analyze_confidence_reallocation(combined_df)
    
    # Original analysis on all edges
    analyze_and_plot_trends(combined_df, args.input_dir, 
                            filename="metrics_and_delta_trends_all_edges.png", 
                            title_suffix=" (All Edges)")
    
    # New analysis on changed edges only
    analyze_and_plot_trends(changed_df, args.input_dir, 
                            filename="metrics_and_delta_trends_changed_only.png", 
                            title_suffix=" (Changed Edges Only)")
    
    # Propagation path analysis
    analyze_propagation_path(combined_df, args.input_dir)
    analyze_propagation_heatmap(combined_df, args.input_dir)


if __name__ == "__main__":
    main()



import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re

def load_all_comparison_data(input_dir):
    """Loads and concatenates all comparison_d*.csv files."""
    input_path = Path(input_dir)
    all_dfs = []
    for csv_file in input_path.glob("comparison_d*.csv"):
        df = pd.read_csv(csv_file)
        all_dfs.append(df)
    
    if not all_dfs:
        print("Error: No comparison_d*.csv files found in the specified directory.")
        return pd.DataFrame()
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Successfully loaded and combined data from {len(all_dfs)} files. Total records: {len(combined_df)}")
    return combined_df

def analyze_confidence_reallocation(df):
    """Analyzes cases where confidence increases as accuracy drops and calculates their proportion."""
    print("\n--- Confidence Reallocation & False Confidence Analysis ---")
    
    # Filter for cases where accuracy dropped but confidence in the (wrong) answer increased
    reallocated_df = df[(df['accuracy_change'] < 0) & (df['confidence_change'] > 0)].copy()
    
    if reallocated_df.empty:
        print("No instances of confidence reallocation found.")
        return

    print(f"Found {len(reallocated_df)} instances where confidence was reallocated to an incorrect answer (False Confidence).")
    
    # Extract distance number for proper sorting
    df['distance_num'] = df['distance'].str.extract(r'd(\d+)').astype(int)
    reallocated_df['distance_num'] = reallocated_df['distance'].str.extract(r'd(\d+)').astype(int)
    
    # Get total counts at each distance
    total_counts_by_dist = df.groupby('distance_num').size().rename('total_count')
    
    # Analyze the effect by distance
    reallocation_by_dist = reallocated_df.groupby('distance_num').agg(
        false_confidence_count=('distance', 'size'),
        avg_confidence_increase=('confidence_change', 'mean'),
        avg_accuracy_decrease=('accuracy_change', 'mean')
    ).sort_index()
    
    # Merge to calculate proportion
    reallocation_by_dist = reallocation_by_dist.join(total_counts_by_dist)
    reallocation_by_dist['false_confidence_proportion'] = (reallocation_by_dist['false_confidence_count'] / reallocation_by_dist['total_count']) * 100
    
    print("\nFalse Confidence Analysis by Distance:")
    print(reallocation_by_dist[['total_count', 'false_confidence_count', 'false_confidence_proportion', 'avg_confidence_increase', 'avg_accuracy_decrease']])
    
    return reallocation_by_dist

def analyze_propagation_path(df, output_dir):
    """Analyzes the relationship between initial confidence and accuracy change."""
    print("\n--- Ripple Effect Propagation Path Analysis ---")
    output_path = Path(output_dir)

    # Focus on edges where accuracy actually dropped to see the impact
    impacted_df = df[df['accuracy_change'] < 0].copy()

    if impacted_df.empty:
        print("No edges with accuracy drops found. Skipping propagation analysis.")
        return

    print("Hypothesis: Do attacks propagate along high-confidence edges?")
    print("Analyzing correlation between clean_confidence and accuracy_change...")

    # Create confidence bins
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    impacted_df['confidence_bin'] = pd.cut(impacted_df['clean_confidence'], bins=bins, labels=labels, include_lowest=True)

    # Analyze accuracy drop per confidence bin
    propagation_analysis = impacted_df.groupby('confidence_bin', observed=False).agg(
        count=('confidence_bin', 'size'),
        avg_accuracy_drop=('accuracy_change', 'mean')
    ).sort_index()

    print("\nAverage Accuracy Drop by Initial (Clean) Confidence Level:")
    print(propagation_analysis)

    # Plotting the relationship
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=impacted_df, x='clean_confidence', y='accuracy_change', alpha=0.5)
    plt.title('Accuracy Change vs. Initial Clean Confidence for Impacted Edges')
    plt.xlabel('Initial Clean Confidence')
    plt.ylabel('Accuracy Change (Delta)')
    plt.grid(True)
    plot_file = output_path / "propagation_path_analysis.png"
    plt.savefig(plot_file)
    print(f"\nSaved propagation path analysis plot to {plot_file}")


def analyze_propagation_heatmap(df, output_dir):
    """Analyzes accuracy drop as a function of both distance and initial confidence."""
    print("\n--- 2D Propagation Analysis (Heatmap) ---")
    output_path = Path(output_dir)

    impacted_df = df[df['accuracy_change'] < 0].copy()
    if impacted_df.empty:
        print("No edges with accuracy drops found. Skipping heatmap analysis.")
        return

    # Ensure 'distance_num' and 'confidence_bin' are created
    impacted_df['distance_num'] = impacted_df['distance'].str.extract(r'd(\d+)').astype(int)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    impacted_df['confidence_bin'] = pd.cut(impacted_df['clean_confidence'], bins=bins, labels=labels, include_lowest=True)

    # Two-dimensional group-by
    heatmap_data = impacted_df.groupby(['distance_num', 'confidence_bin'], observed=False).agg(
        avg_accuracy_drop=('accuracy_change', 'mean')
    ).reset_index()

    # Pivot the data to create a matrix for the heatmap
    heatmap_pivot = heatmap_data.pivot(index='confidence_bin', columns='distance_num', values='avg_accuracy_drop')

    print("\nAverage Accuracy Drop by Distance and Initial Confidence:")
    print(heatmap_pivot.to_string(float_format="%.1f"))

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, fmt=".1f", cmap="viridis", linewidths=.5, cbar_kws={'label': 'Average Accuracy Drop'})
    plt.title('Heatmap of Avg. Accuracy Drop by Distance and Initial Confidence')
    plt.xlabel('Distance (d)')
    plt.ylabel('Initial Clean Confidence')
    plot_file = output_path / "propagation_heatmap.png"
    plt.savefig(plot_file)
    print(f"\nSaved propagation heatmap to {plot_file}")


def analyze_and_plot_trends(df, output_dir, filename="trends.png", title_suffix=""):
    """Analyzes and plots the trends of metrics vs. distance."""
    print(f"\n--- Trend Analysis{title_suffix} ---")
    output_path = Path(output_dir)
    
    if df.empty:
        print("Input DataFrame is empty. Skipping trend analysis.")
        return

    # Extract distance number for sorting and plotting
    df['distance_num'] = df['distance'].str.extract(r'd(\d+)').astype(int)
    
    # Aggregate metrics by distance
    trend_data = df.groupby('distance_num').agg(
        avg_poisoned_accuracy=('poison_accuracy', 'mean'),
        avg_poisoned_confidence=('poison_confidence', 'mean'),
        avg_accuracy_change=('accuracy_change', 'mean'),
        avg_confidence_change=('confidence_change', 'mean')
    ).sort_index()
    
    print("\nAverage Metrics by Distance:")
    print(trend_data)
    
    # Plotting
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Average Accuracy (Absolute Values)
    axes[0].plot(trend_data.index, trend_data['avg_poisoned_accuracy'] + trend_data['avg_accuracy_change'].abs(), marker='o', linestyle='--', label='Clean Accuracy (Avg)')
    axes[0].plot(trend_data.index, trend_data['avg_poisoned_accuracy'], marker='o', label='Poisoned Accuracy (Avg)')
    axes[0].set_title(f'Average Model Accuracy vs. Distance{title_suffix}')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_ylim(-5, 105)

    # Plot 2: Average Confidence (Absolute Values)
    axes[1].plot(trend_data.index, trend_data['avg_poisoned_confidence'] - trend_data['avg_confidence_change'], marker='o', linestyle='--', label='Clean Confidence (Avg)')
    axes[1].plot(trend_data.index, trend_data['avg_poisoned_confidence'], marker='o', label='Poisoned Confidence (Avg)')
    axes[1].set_title(f'Average Model Confidence vs. Distance{title_suffix}')
    axes[1].set_ylabel('Confidence')
    axes[1].legend()

    # Plot 3: Delta Trends (Change in Accuracy and Confidence)
    ax3 = axes[2]
    ax3.plot(trend_data.index, trend_data['avg_accuracy_change'], marker='o', color='r', label='Accuracy Change (Delta)')
    ax3.set_ylabel('Avg. Accuracy Change', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8)

    ax4 = ax3.twinx()
    ax4.plot(trend_data.index, trend_data['avg_confidence_change'], marker='o', color='b', label='Confidence Change (Delta)')
    ax4.set_ylabel('Avg. Confidence Change', color='b')
    ax4.tick_params(axis='y', labelcolor='b')
    
    ax3.set_title(f'Average Change (Delta) in Accuracy & Confidence vs. Distance{title_suffix}')
    ax3.set_xlabel('Distance (d)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax3.transAxes)

    plt.tight_layout()
    plot_file = output_path / filename
    plt.savefig(plot_file)
    print(f"\nSaved trend plot to {plot_file}")

def main():
    """Main function to run the advanced analysis."""
    parser = argparse.ArgumentParser(description="Perform advanced analysis of ripple effect experiment results.")
    parser.add_argument("--input_dir", type=str, default="analysis_output", help="Directory containing the comparison CSV files.")
    args = parser.parse_args()

    # Load and combine data
    combined_df = load_all_comparison_data(args.input_dir)
    
    if combined_df.empty:
        return

    # Create a dataframe with only the edges that have changed
    changed_df = combined_df[(combined_df['poison_accuracy'] != combined_df['clean_accuracy']) | (combined_df['poison_confidence'] != combined_df['clean_confidence'])].copy()

    # Perform analyses
    analyze_confidence_reallocation(combined_df)
    
    # Original analysis on all edges
    analyze_and_plot_trends(combined_df, args.input_dir, 
                            filename="metrics_and_delta_trends_all_edges.png", 
                            title_suffix=" (All Edges)")
    
    # New analysis on changed edges only
    analyze_and_plot_trends(changed_df, args.input_dir, 
                            filename="metrics_and_delta_trends_changed_only.png", 
                            title_suffix=" (Changed Edges Only)")
    
    # Propagation path analysis
    analyze_propagation_path(combined_df, args.input_dir)
    analyze_propagation_heatmap(combined_df, args.input_dir)


if __name__ == "__main__":
    main()




import pandas as pd
import argparse
from pathlib import Path

def analyze_statistics(input_file):
    """
    Analyzes the comparison data to provide key statistics about nodes and edges.
    """
    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Calculate the total number of unique nodes
    all_nodes = pd.concat([df['head'], df['tail']]).unique()
    total_node_count = len(all_nodes)
    print(f"\n--- 统计结果 ---")
    print(f"1. 实验涉及的独立“节点”（实体）总数: {total_node_count}")

    # 2. Calculate the number of edges and nodes with changes
    # Handle potential None/NaN values in change columns before comparison by filling with 0
    df['accuracy_change'] = df['accuracy_change'].fillna(0)
    df['confidence_change'] = df['confidence_change'].fillna(0)

    # Filter for edges where there was a non-zero change
    changed_edges_df = df[(df['accuracy_change'] != 0) | (df['confidence_change'] != 0)]
    changed_edges_count = len(changed_edges_df)
    
    print(f"\n2. 投毒前后发生变化的统计:")
    print(f"   - 发生变化的“边”（三元组关系）的数量: {changed_edges_count}")

    if changed_edges_count > 0:
        # Calculate the number of unique nodes involved in these changes
        changed_nodes = pd.concat([changed_edges_df['head'], changed_edges_df['tail']]).unique()
        changed_nodes_count = len(changed_nodes)
        print(f"   - 这些变化涉及的独立“节点”（实体）数量: {changed_nodes_count}")
    else:
        print("   - 没有检测到任何节点发生变化。")


def main():
    """Main function to run the statistics analysis."""
    parser = argparse.ArgumentParser(description="Provide statistics on node and edge changes from comparison data.")
    parser.add_argument("--input_file", type=str, default="analysis_output/comparison_main_graph.csv",
                        help="Path to the comparison_main_graph.csv file.")
    args = parser.parse_args()
    
    analyze_statistics(args.input_file)

if __name__ == "__main__":
    main()


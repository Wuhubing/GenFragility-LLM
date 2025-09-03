import argparse
import pickle
import gzip

# This will be the entry point for the ripple experiments,
# consuming the graph file produced by the builder.

def main():
    parser = argparse.ArgumentParser(description="Run ripple experiments on a generated graph.")
    parser.add_argument(
        "graph_file", 
        type=str, 
        help="Path to the graph file (.pkl or .pkl.gz)"
    )
    args = parser.parse_args()

    print(f"Loading graph from {args.graph_file}...")
    
    if args.graph_file.endswith(".gz"):
        with gzip.open(args.graph_file, 'rb') as f:
            G = pickle.load(f)
    else:
        with open(args.graph_file, 'rb') as f:
            G = pickle.load(f)
            
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    #
    # ... Downstream ripple experiment logic goes here ...
    #
    print("Ripple experiment logic not yet implemented.")


if __name__ == "__main__":
    main()
import pickle
import gzip

# This will be the entry point for the ripple experiments,
# consuming the graph file produced by the builder.

def main():
    parser = argparse.ArgumentParser(description="Run ripple experiments on a generated graph.")
    parser.add_argument(
        "graph_file", 
        type=str, 
        help="Path to the graph file (.pkl or .pkl.gz)"
    )
    args = parser.parse_args()

    print(f"Loading graph from {args.graph_file}...")
    
    if args.graph_file.endswith(".gz"):
        with gzip.open(args.graph_file, 'rb') as f:
            G = pickle.load(f)
    else:
        with open(args.graph_file, 'rb') as f:
            G = pickle.load(f)
            
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    #
    # ... Downstream ripple experiment logic goes here ...
    #
    print("Ripple experiment logic not yet implemented.")


if __name__ == "__main__":
    main()
import pickle
import gzip

# This will be the entry point for the ripple experiments,
# consuming the graph file produced by the builder.

def main():
    parser = argparse.ArgumentParser(description="Run ripple experiments on a generated graph.")
    parser.add_argument(
        "graph_file", 
        type=str, 
        help="Path to the graph file (.pkl or .pkl.gz)"
    )
    args = parser.parse_args()

    print(f"Loading graph from {args.graph_file}...")
    
    if args.graph_file.endswith(".gz"):
        with gzip.open(args.graph_file, 'rb') as f:
            G = pickle.load(f)
    else:
        with open(args.graph_file, 'rb') as f:
            G = pickle.load(f)
            
    print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    #
    # ... Downstream ripple experiment logic goes here ...
    #
    print("Ripple experiment logic not yet implemented.")


if __name__ == "__main__":
    main()

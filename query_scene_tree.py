#!/usr/bin/env python3
"""
Query Scene Tree for semantic video retrieval.
Loads a pre-built Scene Tree and performs retrieval based on user queries.

Usage: python query_scene_tree.py
"""

import os
import sys
from scene_tree import SceneTreeBuilder


def query_scene_tree_interactive(tree_path: str, api_key: str):
    """
    Interactive query interface for Scene Tree.
    
    Args:
        tree_path: Path to the pickle file containing the Scene Tree
        api_key: Gemini API key (needed for embedding model)
    """
    # Load the Scene Tree
    print(f"Loading Scene Tree from: {tree_path}")
    try:
        root = SceneTreeBuilder.load_tree(tree_path)
        SceneTreeBuilder.save_tree_as_json(root, "./scene_tree.json")
        print(f"✓ Scene Tree loaded successfully!")
        print(f"  Root node ID: {root.node_id}")
        print(f"  Tree level: {root.level}")
        print()
    except FileNotFoundError:
        print(f"Error: Scene Tree file not found at '{tree_path}'")
        print("Please run extract_frames.py first to build the Scene Tree.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Scene Tree: {e}")
        sys.exit(1)
    
    # Initialize Scene Tree Builder (for embedding and retrieval)
    print("Initializing embedding model and NLP tools...")
    try:
        builder = SceneTreeBuilder(
            api_key=api_key,
            embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        print("✓ Ready for queries!\n")
    except Exception as e:
        print(f"Error initializing builder: {e}")
        sys.exit(1)
    
    # Interactive query loop
    print("="*80)
    print("SCENE TREE QUERY INTERFACE")
    print("="*80)
    print("\nEnter your queries to search through video scenes.")
    print("Commands:")
    print("  - Type your query and press Enter to search")
    print("  - Type 'params' to change alpha/beta weights")
    print("  - Type 'quit' or 'exit' to exit")
    print("="*80 + "\n")
    
    # Default parameters
    alpha = 0.7
    beta = 0.3
    top_k = 3
    
    while True:
        try:
            # Get user input
            query = input(f"\n[α={alpha}, β={beta}] Query: ").strip()
            
            if not query:
                continue
            
            # Check for commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            elif query.lower() == 'params':
                print("\nCurrent parameters:")
                print(f"  alpha (embedding weight): {alpha}")
                print(f"  beta (entity weight): {beta}")
                print(f"  top_k (results to return): {top_k}")
                print()
                
                try:
                    new_alpha = input(f"New alpha [{alpha}]: ").strip()
                    if new_alpha:
                        alpha = float(new_alpha)
                    
                    new_beta = input(f"New beta [{beta}]: ").strip()
                    if new_beta:
                        beta = float(new_beta)
                    
                    new_top_k = input(f"New top_k [{top_k}]: ").strip()
                    if new_top_k:
                        top_k = int(new_top_k)
                    
                    print(f"\n✓ Parameters updated: α={alpha}, β={beta}, top_k={top_k}")
                except ValueError:
                    print("Invalid input. Parameters unchanged.")
                
                continue
            
            # Perform retrieval
            results, level_info = builder.retrieve(
                root=root,
                query=query,
                alpha=alpha,
                beta=beta,
                top_k=top_k
            )
            
            # Print traversal path
            print_traversal_path(level_info)
            
            # Print final results
            print_results(results)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError during retrieval: {e}")
            continue


def print_traversal_path(level_info):
    """Print the traversal path showing top nodes at each level."""
    print("\n" + "="*80)
    print("TRAVERSAL PATH (Top 3 Nodes at Each Level)")
    print("="*80 + "\n")
    
    for level_data in level_info:
        level = level_data['level']
        top_selections = level_data['top_selections']
        
        if not top_selections:
            continue
        
        print(f"Level {level}:")
        print("-" * 80)
        
        for i, selection in enumerate(top_selections, 1):
            node_type = selection['child_type']
            node_id = selection['child_node_id']
            ranking_score = selection['ranking_score']
            accumulated_score = selection['accumulated_score']
            emb_sim = selection['emb_sim']
            entity_overlap = selection['entity_overlap']
            
            print(f"  {i}. {node_type} #{node_id}")
            print(f"     Ranking Score: {ranking_score:.4f}")
            print(f"     Accumulated Score: {accumulated_score:.4f}")
            print(f"     Embedding Similarity: {emb_sim:.4f}")
            print(f"     Entity Overlap: {entity_overlap:.4f}")
            print()
        
        print()


def print_results(results):
    """Print retrieval results in a formatted way."""
    print("\n" + "="*80)
    print("TOP 3 LEAF NODES (Final Results)")
    print("="*80 + "\n")
    
    if not results:
        print("No results found.")
        return
    
    for rank, (node, score) in enumerate(results, 1):
        print(f"Rank {rank}: Confidence Score = {score:.4f}")
        print(f"  Node ID: {node.node_id}")
        print(f"  Image Path: {node.image_path}")
        print(f"  Description: {node.description}")
        print()


def query_single(tree_path: str, api_key: str, query: str, alpha: float = 0.7, beta: float = 0.3, top_k: int = 3):
    """
    Perform a single query (non-interactive).
    
    Args:
        tree_path: Path to the pickle file
        api_key: Gemini API key
        query: Query string
        alpha: Embedding weight
        beta: Entity weight
        top_k: Number of results to return
    """
    # Load tree
    print(f"Loading Scene Tree from: {tree_path}")
    root = SceneTreeBuilder.load_tree(tree_path)
    
    # Initialize builder
    print("Initializing models...")
    builder = SceneTreeBuilder(
        api_key=api_key,
        embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    
    # Perform retrieval
    results, level_info = builder.retrieve(
        root=root,
        query=query,
        alpha=alpha,
        beta=beta,
        top_k=top_k
    )
    
    # Print traversal path
    print_traversal_path(level_info)
    
    # Print final results
    print_results(results)
    
    return results, level_info


if __name__ == "__main__":
    # Configuration
    SCENE_TREE_PATH = "/home/neetzsche/Documents/code/fyp-latest/saves/mission_impossible/scene_tree.pkl"
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Check if tree file exists
    if not os.path.exists(SCENE_TREE_PATH):
        print(f"Error: Scene Tree file not found at '{SCENE_TREE_PATH}'")
        print("\nPlease run extract_frames.py first to build the Scene Tree:")
        print("  python extract_frames.py")
        sys.exit(1)
    

    query_scene_tree_interactive(SCENE_TREE_PATH, api_key)


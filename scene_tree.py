"""
Scene Tree Builder using Google Gemini 2.5 Pro for video collage description and consolidation.
"""

import pickle
import json
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import google.generativeai as genai
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from collections import Counter
import heapq
import time


@dataclass
class TreeNode:
    """Node in the Scene Tree."""
    node_id: int
    description: str = ""
    image_path: Optional[str] = None
    children: List['TreeNode'] = field(default_factory=list)
    is_leaf: bool = False
    level: int = 0  # 0 = leaf, increases as we go up
    embedding: Optional[np.ndarray] = None  # Description embedding


class SceneTreeBuilder:
    """Builds a hierarchical scene tree from video collages using Gemini API."""
    
    LEAF_SYSTEM_PROMPT = """You are an expert video describer. You will be given an image, which is actually a video, with one frame stacked on the next. You shall describe the video with only the essential details. Think of it as you describing a scene to a different person with perfect memory. 

Instructions:
1. Capture only the essential details, and intricate details (like say house number or anything in the video worth remembering).
2. Do not overuse adjectives. Describe in plain words and to-the-point, do not be poetic about it.
3. Do not talk in terms of frames or scenes. The final description should be about what is happening in the video as a whole, from start to finish.
4. Return only the description and nothing more."""

    CONSOLIDATION_SYSTEM_PROMPT = """
    You will be given three scenes. These scenes happen one after another. Your job is to analyse and understand what is happening given the continuity of these scenes, and then to return a description of the whole from start to finish.
    Scene 1 happens before Scene 2, and Scene 2 happens before Scene 3. The individual scene descriptions might seem disconnected, but they are not. You must analyse, understand and maintain context between these scenes.  
    Identify common objects / entities, and while they evolve over time across multiple scenes, ensure you only record the latest description or value of it. Reason the common context, and intentions if necessary.

Instructions:
1. The description should include the activity from start to finish, and should maintain continuity between the scenes.
2. Maintain context between scenes. Identify common subjects / entities, and ensure you emphasise on the latest description / value of them.
3. Do not overuse adjectives. Describe in plain words and to-the-point, do not be poetic about it.
4. The description you give will be fed into the knowledge graph, so be a little brief, while still capturing and storing all essential details, activity and common context.
5. Return only the description and nothing more."""

    def __init__(self, api_key: str, embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize with Gemini API key and embedding model."""
        genai.configure(api_key=api_key)
        self.leaf_model = genai.GenerativeModel('gemini-2.5-flash')
        self.consolidation_model = genai.GenerativeModel('gemini-2.5-flash')
        self.node_counter = 0
        
        # Initialize embedding model (MiniLM is fast and efficient)
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize spaCy for named entity extraction
        print("Loading spaCy model for entity extraction...")
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
    
    def _get_next_node_id(self) -> int:
        """Get next node ID."""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding
    
    def _describe_leaf_collage(self, image_path: str) -> str:
        """Get description for a single collage using Gemini."""
        print(f"Processing leaf node: {image_path}")
        
        # Upload and process the image
        image_file = genai.upload_file(path=image_path)
        
        response = self.leaf_model.generate_content(
            [self.LEAF_SYSTEM_PROMPT, image_file],
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
            )
        )
        
        if response.candidates:
            description = response.text.strip()
        else:
            print("ERROR: RESPONSE.CANDIDATES EMPTY")
            return False

        print(f"  Description: {description[:100]}...")
        
        # Rate limiting: wait 5 seconds after API call
        print(f"  Waiting 10 seconds (rate limit protection)...")
        time.sleep(10)
        
        return description
    
    def _consolidate_scenes(self, scenes: List[str]) -> str:
        """Consolidate multiple scene descriptions into one."""
        print(f"Consolidating {len(scenes)} scenes")
        
        # Build user prompt with scene numbers
        user_prompt = ""
        for i, scene in enumerate(scenes, 1):
            user_prompt += f"Scene {i}: {scene}\n"
        
        response = self.consolidation_model.generate_content(
            [self.CONSOLIDATION_SYSTEM_PROMPT, user_prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
            )
        )
        
        description = response.text.strip()
        if "Response:" in description:
            description = description.split("Response:")[1].strip()
        print(f"  Consolidated: {description[:100]}...")
        
        # Rate limiting: wait 5 seconds after API call
        print(f"  Waiting 10 seconds (rate limit protection)...")
        time.sleep(10)
        
        return description
    
    def build_tree(self, image_paths: List[str]) -> TreeNode:
        """
        Build the complete scene tree from leaf images.
        
        Args:
            image_paths: List of paths to collage images (leaf nodes)
            
        Returns:
            Root node of the tree
        """
        print(f"\n=== Building Scene Tree with {len(image_paths)} leaf nodes ===\n")
        
        # Step 1: Create leaf nodes and get descriptions
        print("Step 1: Processing leaf nodes...")
        leaf_nodes = []
        for img_path in image_paths:
            description = self._describe_leaf_collage(img_path)
            if not description:
                continue
            embedding = self._create_embedding(description)
            node = TreeNode(
                node_id=self._get_next_node_id(),
                description=description,
                image_path=img_path,
                is_leaf=True,
                level=0,
                embedding=embedding
            )
            leaf_nodes.append(node)
        
        print(f"\nCreated {len(leaf_nodes)} leaf nodes\n")
        
        # Step 2: Build tree bottom-up
        current_level = leaf_nodes
        level_num = 1
        
        while len(current_level) > 1:
            print(f"Step {level_num + 1}: Creating level {level_num} (consolidating {len(current_level)} nodes)")
            next_level = []
            
            # Process nodes in groups of up to 3
            for i in range(0, len(current_level), 3):
                group = current_level[i:i+3]
                
                # Get descriptions from this group
                descriptions = [node.description for node in group]
                
                # Consolidate
                consolidated_desc = self._consolidate_scenes(descriptions)
                
                # Create embedding for consolidated description
                embedding = self._create_embedding(consolidated_desc)
                
                # Create parent node
                parent = TreeNode(
                    node_id=self._get_next_node_id(),
                    description=consolidated_desc,
                    children=group,
                    is_leaf=False,
                    level=level_num,
                    embedding=embedding
                )
                next_level.append(parent)
            
            print(f"  Created {len(next_level)} parent nodes\n")
            current_level = next_level
            level_num += 1
        
        root = current_level[0]
        print(f"=== Tree building complete! Root node ID: {root.node_id} ===\n")
        return root
    
    def save_tree(self, root: TreeNode, filepath: str):
        """Save the tree to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(root, f)
        print(f"Tree saved to {filepath}")
    
    @staticmethod
    def load_tree(filepath: str) -> TreeNode:
        """Load a tree from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def _node_to_dict(node: TreeNode) -> Dict[str, Any]:
        """Convert a TreeNode to a dictionary (for JSON serialization)."""
        return {
            'node_id': node.node_id,
            'description': node.description,
            'image_path': node.image_path,
            'is_leaf': node.is_leaf,
            'level': node.level,
            'embedding': node.embedding.tolist() if node.embedding is not None else None,
            'children': [SceneTreeBuilder._node_to_dict(child) for child in node.children]
        }
    
    @staticmethod
    def save_tree_as_json(root: TreeNode, filepath: str):
        """Save the tree to a JSON file."""
        tree_dict = SceneTreeBuilder._node_to_dict(root)
        with open(filepath, 'w') as f:
            json.dump(tree_dict, f, indent=2)
        print(f"Tree saved as JSON to {filepath}")
    
    @staticmethod
    def print_tree(node: TreeNode, indent: int = 0, prefix: str = ""):
        """
        Pretty print the tree structure showing consolidation hierarchy.
        
        Args:
            node: Current node to print
            indent: Current indentation level
            prefix: Prefix for the current line
        """
        # Prepare node info
        node_type = "LEAF" if node.is_leaf else "NODE"
        desc_preview = node.description[:80] + "..." if len(node.description) > 80 else node.description
        
        # Print current node
        indent_str = "  " * indent
        print(f"{indent_str}{prefix}[{node_type} #{node.node_id}] Level {node.level}")
        
        if node.image_path:
            print(f"{indent_str}  Image: {node.image_path}")
        
        print(f"{indent_str}  Desc: {desc_preview}")
        
        # Print children
        if node.children:
            print(f"{indent_str}  Consolidates {len(node.children)} nodes:")
            for i, child in enumerate(node.children, 1):
                SceneTreeBuilder.print_tree(child, indent + 1, f"{i}. ")
        
        print()
    
    @staticmethod
    def _extract_entities(text: str, nlp) -> set:
        """Extract named entities from text using spaCy."""
        doc = nlp(text)
        entities = set()
        
        # Extract named entities
        for ent in doc.ents:
            entities.add(ent.text.lower())
        
        # Also extract noun chunks as potential entities
        for chunk in doc.noun_chunks:
            entities.add(chunk.text.lower())
        
        return entities
    
    @staticmethod
    def _entity_overlap_score(query_entities: set, node_entities: set) -> float:
        """Calculate entity overlap score using Jaccard similarity."""
        if not query_entities or not node_entities:
            return 0.0
        
        intersection = len(query_entities & node_entities)
        union = len(query_entities | node_entities)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        return dot_product / norm_product if norm_product > 0 else 0.0
    
    def retrieve(
        self,
        root: TreeNode,
        query: str,
        alpha: float = 0.7,
        beta: float = 0.3,
        top_k: int = 3
    ) -> Tuple[List[Tuple[TreeNode, float]], List[Dict[str, Any]]]:
        """
        Retrieve top-k leaf nodes using BFS with hybrid ranking.
        
        Args:
            root: Root node of the tree
            query: Query string
            alpha: Weight for embedding similarity (default: 0.7)
            beta: Weight for entity overlap (default: 0.3)
            top_k: Number of top children to explore at each level (default: 3)
            
        Returns:
            Tuple of:
            - List of (leaf_node, confidence_score) tuples, sorted by score (descending)
            - List of level info dicts containing traversal path details
        """
        print(f"\n=== Retrieval for query: '{query}' ===")
        print(f"Parameters: alpha={alpha}, beta={beta}, top_k={top_k}\n")
        
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Extract query entities
        query_entities = self._extract_entities(query, self.nlp)
        print(f"Query entities: {query_entities}\n")
        
        # Start BFS from root
        # Queue contains tuples of (node, accumulated_score)
        queue = [(root, 1.0)]  # Start with perfect score at root
        leaf_results = []
        level_info = []  # Track traversal at each level
        all_leaf_results_by_level = {}  # Track all leaf nodes reached at each level
        
        level = root.level
        while queue:
            # Process all nodes at current level
            current_level_nodes = queue
            queue = []
            
            print(f"--- Level {level} ({len(current_level_nodes)} nodes) ---")
            
            level_data = {
                'level': level,
                'nodes_explored': [],
                'top_selections': [],
                'top_3_leaves': []  # Top 3 leaf nodes reached from this level
            }
            
            level_leaf_results = []  # Collect leaf nodes found at this level
            
            for node, parent_score in current_level_nodes:
                # If leaf node, add to results
                if node.is_leaf:
                    # Calculate final score for leaf
                    node_entities = self._extract_entities(node.description, self.nlp)
                    
                    sim_emb = self._cosine_similarity(query_embedding, node.embedding)
                    entity_overlap = self._entity_overlap_score(query_entities, node_entities)
                    
                    ranking_score = alpha * sim_emb + beta * entity_overlap
                    final_score = parent_score * ranking_score
                    
                    leaf_results.append((node, final_score))
                    level_leaf_results.append((node, final_score))
                    print(f"  LEAF #{node.node_id}: score={final_score:.4f} "
                          f"(emb={sim_emb:.4f}, entity={entity_overlap:.4f})")
                    
                    level_data['nodes_explored'].append({
                        'node_id': node.node_id,
                        'node_type': 'LEAF',
                        'score': final_score,
                        'emb_sim': sim_emb,
                        'entity_overlap': entity_overlap
                    })
                else:
                    # Rank children and add top-k to queue
                    child_scores = []
                    
                    for child in node.children:
                        child_entities = self._extract_entities(child.description, self.nlp)
                        
                        sim_emb = self._cosine_similarity(query_embedding, child.embedding)
                        entity_overlap = self._entity_overlap_score(query_entities, child_entities)
                        
                        ranking_score = alpha * sim_emb + beta * entity_overlap
                        child_scores.append((child, ranking_score, sim_emb, entity_overlap))
                    
                    # Sort by score and take top-k
                    child_scores.sort(key=lambda x: x[1], reverse=True)
                    top_children = child_scores[:top_k]
                    
                    print(f"  NODE #{node.node_id}: exploring top {len(top_children)} children")
                    
                    for child, score, sim_emb, entity_overlap in top_children:
                        child_type = "LEAF" if child.is_leaf else "NODE"
                        print(f"    -> {child_type} #{child.node_id}: score={score:.4f}")
                        
                        # Accumulate score (multiply with parent's score)
                        accumulated_score = parent_score * score
                        queue.append((child, accumulated_score))
                        
                        # Track top selections
                        level_data['top_selections'].append({
                            'parent_node_id': node.node_id,
                            'child_node_id': child.node_id,
                            'child_type': child_type,
                            'ranking_score': score,
                            'accumulated_score': accumulated_score,
                            'emb_sim': sim_emb,
                            'entity_overlap': entity_overlap
                        })
            
            # Get top 3 leaf nodes for this level
            if level_leaf_results:
                # Sort by score and take top 3
                level_leaf_results.sort(key=lambda x: x[1], reverse=True)
                top_3_leaves = level_leaf_results[:3]
                
                for leaf_node, leaf_score in top_3_leaves:
                    level_data['top_3_leaves'].append({
                        'node_id': leaf_node.node_id,
                        'image_path': leaf_node.image_path,
                        'score': leaf_score,
                        'description': leaf_node.description
                    })
            
            level_info.append(level_data)
            level -= 1
            print()
        
        # Sort leaf results by score (descending)
        leaf_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"=== Retrieved {len(leaf_results)} leaf nodes ===")
        return leaf_results[:top_k], level_info
    
    def print_retrieval_results(self, results: List[Tuple[TreeNode, float]]):
        """Pretty print retrieval results."""
        print("\n" + "="*80)
        print("TOP RETRIEVAL RESULTS")
        print("="*80 + "\n")
        
        for rank, (node, score) in enumerate(results, 1):
            print(f"Rank {rank}: [LEAF #{node.node_id}] - Confidence: {score:.4f}")
            print(f"  Image: {node.image_path}")
            print(f"  Description: {node.description}")
            print()


def main():
    """Example usage."""
    import os
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        return
    
    # Example: List of collage image paths (replace with your actual paths)
    image_paths = [
        "/path/to/collage1.jpg",
        "/path/to/collage2.jpg",
        "/path/to/collage3.jpg",
        "/path/to/collage4.jpg",
        "/path/to/collage5.jpg",
        "/path/to/collage6.jpg",
        "/path/to/collage7.jpg",
    ]
    
    # Build the tree
    builder = SceneTreeBuilder(api_key=api_key)
    root = builder.build_tree(image_paths)
    
    # Save the tree
    builder.save_tree(root, "scene_tree.pkl")
    
    # Save as JSON (optional)
    SceneTreeBuilder.save_tree_as_json(root, "scene_tree.json")
    
    # Print the tree structure
    print("\n" + "="*80)
    print("SCENE TREE STRUCTURE")
    print("="*80 + "\n")
    SceneTreeBuilder.print_tree(root)
    
    # Example retrieval
    query = "a man in a suit"
    results, level_info = builder.retrieve(root, query, alpha=0.7, beta=0.3, top_k=3)
    builder.print_retrieval_results(results)
    
    # Example: Load the tree back and perform retrieval
    loaded_root = SceneTreeBuilder.load_tree("scene_tree.pkl")
    print(f"\nLoaded tree with root node #{loaded_root.node_id}")
    
    # Note: To use retrieval on loaded tree, you need a builder instance
    # builder2 = SceneTreeBuilder(api_key=api_key)
    # results2 = builder2.retrieve(loaded_root, "another query")
    # builder2.print_retrieval_results(results2)


if __name__ == "__main__":
    main()


# Scene Tree Builder

A hierarchical scene tree builder that uses Google Gemini 2.5 Pro to describe video collages and consolidate them into a coherent narrative structure.

## Overview

This system takes video collages (where frames are stacked vertically) and builds a hierarchical tree structure:
1. **Leaf nodes**: Individual collage descriptions
2. **Parent nodes**: Consolidations of up to 3 consecutive scenes
3. **Root node**: Final consolidated description of the entire sequence

## Installation

```bash
pip install -r requirements.txt
```

## Setup

Set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from scene_tree import SceneTreeBuilder
import os

# Initialize builder
api_key = os.getenv('GEMINI_API_KEY')
builder = SceneTreeBuilder(api_key=api_key)

# List of collage image paths (in order)
image_paths = [
    "/path/to/collage1.jpg",
    "/path/to/collage2.jpg",
    "/path/to/collage3.jpg",
    # ... more collages
]

# Build the tree
root = builder.build_tree(image_paths)

# Save the tree
builder.save_tree(root, "scene_tree.pkl")

# Print the tree structure
SceneTreeBuilder.print_tree(root)
```

### Loading a Saved Tree

```python
from scene_tree import SceneTreeBuilder

# Load previously saved tree
root = SceneTreeBuilder.load_tree("scene_tree.pkl")

# Print it
SceneTreeBuilder.print_tree(root)
```

## Tree Structure

The tree is built bottom-up:

```
                    [ROOT]
                      |
        +-------------+-------------+
        |             |             |
     [Node 1]      [Node 2]      [Node 3]
        |             |             |
    +---+---+     +---+---+     +---+
    |   |   |     |   |   |     |   |
   L1  L2  L3    L4  L5  L6    L7  L8
```

Where:
- `L1-L8`: Leaf nodes (individual collage descriptions)
- `Node 1-3`: First consolidation level (groups of 3 leaves)
- `ROOT`: Final consolidation

## API

### `SceneTreeBuilder`

**Constructor:**
- `SceneTreeBuilder(api_key: str)`: Initialize with Gemini API key

**Methods:**
- `build_tree(image_paths: List[str]) -> TreeNode`: Build tree from image paths
- `save_tree(root: TreeNode, filepath: str)`: Save tree to pickle file
- `load_tree(filepath: str) -> TreeNode` (static): Load tree from pickle file
- `print_tree(node: TreeNode, ...)` (static): Pretty print tree structure

### `TreeNode`

**Attributes:**
- `node_id: int`: Unique node identifier
- `description: str`: Scene description (from Gemini)
- `image_path: Optional[str]`: Path to collage (only for leaf nodes)
- `children: List[TreeNode]`: Child nodes (empty for leaves)
- `is_leaf: bool`: Whether this is a leaf node
- `level: int`: Tree level (0 = leaf, increases upward)

## Example Output

```
=== Building Scene Tree with 7 leaf nodes ===

Step 1: Processing leaf nodes...
Processing leaf node: /path/to/collage1.jpg
  Description: A man in a black suit runs toward a woman in a yellow dress...
Processing leaf node: /path/to/collage2.jpg
  Description: A woman in a gold dress looks down from a stone balustrade...
...

Created 7 leaf nodes

Step 2: Creating level 1 (consolidating 7 nodes)
Consolidating 3 scenes
  Consolidated: A man in a black suit approaches a woman in a yellow dress...
Consolidating 3 scenes
  Consolidated: The couple runs across the balcony together...
...

=== Tree building complete! Root node ID: 10 ===

================================================================================
SCENE TREE STRUCTURE
================================================================================

[NODE #10] Level 2
  Desc: A romantic scene unfolds as a man in a suit pursues a woman in a gold dress...
  Consolidates 3 nodes:
  1. [NODE #7] Level 1
      Desc: A man in a black suit approaches a woman in a yellow dress on a balcony...
      Consolidates 3 nodes:
      1. [LEAF #0] Level 0
          Image: /path/to/collage1.jpg
          Desc: A man in a black suit runs toward a woman...
      ...
```

## Notes

- The system uses **Gemini 2.0 Flash Exp** model (you can change this in line 43 of `scene_tree.py`)
- Consolidation happens in groups of **up to 3 nodes** at a time
- Temperature is set to **0.4** for consistent but natural descriptions
- The tree structure ensures continuity across all levels


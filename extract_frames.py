#!/usr/bin/env python3
"""
Extract frames from a video at 0.5-second intervals and create stacked collages.
Then build a Scene Tree for semantic retrieval.
Usage: python extract_frames.py <input_video> <output_folder>
"""

import cv2
import os
import sys
import numpy as np
from pathlib import Path
from scene_tree import SceneTreeBuilder


def create_stacked_collages(frames_folder, output_folder, frames_per_stack=8, overlap=0, stack_horizontally=False):
    """Create stacked collages with overlapping frames.
    
    Args:
        frames_folder: Folder containing extracted frames
        output_folder: Folder to save collages
        frames_per_stack: Number of frames per collage (default: 12)
        overlap: Number of overlapping frames between stacks (default: 2)
        stack_horizontally: Whether to stack horizontally (True) or vertically (False)
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all frame files sorted by name
    frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
    
    if len(frame_files) == 0:
        print(f"Error: No frames found in '{frames_folder}'")
        return
    
    stack_direction = "horizontally" if stack_horizontally else "vertically"
    print(f"\nCreating stacked collages from {len(frame_files)} frames")
    print(f"Frames per stack: {frames_per_stack}, Overlap: {overlap}")
    print(f"Stacking direction: {stack_direction}")
    
    collage_count = 0
    start_idx = 0
    step = frames_per_stack - overlap
    
    while start_idx < len(frame_files):
        # Get the frames for this stack
        end_idx = min(start_idx + frames_per_stack, len(frame_files))
        stack_frames = frame_files[start_idx:end_idx]
        
        # Load all frames for this stack
        images = []
        for frame_file in stack_frames:
            img_path = os.path.join(frames_folder, frame_file)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
        
        if len(images) == 0:
            print(f"Warning: Could not load frames for stack {collage_count}")
            start_idx += step
            continue
        
        # Stack images based on orientation
        if stack_horizontally:
            stacked = np.hstack(images)
        else:
            stacked = np.vstack(images)
        
        # Save the collage
        output_path = os.path.join(output_folder, f"collage_{collage_count:04d}.jpg")
        cv2.imwrite(output_path, stacked)
        print(f"Created collage {collage_count}: {len(images)} frames (indices {start_idx}-{end_idx-1})")
        
        collage_count += 1
        start_idx += step
    
    print(f"\nComplete! Created {collage_count} collages.")
    
    # Return list of collage paths
    collage_paths = []
    for i in range(collage_count):
        collage_path = os.path.join(output_folder, f"collage_{i:04d}.jpg")
        collage_paths.append(collage_path)
    
    return collage_paths


def extract_frames(video_path, output_folder):
    """Extract frames from video at 0.5-second intervals.
    
    Returns:
        Tuple of (num_frames, is_vertical) where is_vertical indicates if video is portrait
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return None, False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine orientation
    is_vertical = height > width
    orientation = "Vertical (Portrait)" if is_vertical else "Horizontal (Landscape)"
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Orientation: {orientation}")
    print(f"FPS: {fps:.2f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Extracting frames to: {output_folder}")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save frame every 0.5 seconds (every fps/2 frames)
        if frame_count % int(fps) == 0:
            time_seconds = frame_count / fps
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"Saved frame at {time_seconds:.1f}s: {output_path}")
        
        frame_count += 1
    
    cap.release()
    print(f"\nComplete! Extracted {saved_count} frames.")
    return saved_count, is_vertical


if __name__ == "__main__":    
    video_path = "/home/neetzsche/Documents/code/fyp-latest/videoplayback.mp4"
    output_folder = "/home/neetzsche/Documents/code/fyp-latest/output_frames"
    collages_folder = "/home/neetzsche/Documents/code/fyp-latest/collages_folder"
    scene_tree_path = "/home/neetzsche/Documents/code/fyp-latest/scene_tree.pkl"
    scene_tree_json_path = "/home/neetzsche/Documents/code/fyp-latest/scene_tree.json"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        sys.exit(1)
    
    # Check for Gemini API key
    gemini_api_key = "AIzaSyC8dnV15iSJJ7HlGERiQORaCiXVV5GGfoI"
    if not gemini_api_key:
        print("\nWarning: GEMINI_API_KEY environment variable not set.")
        print("Scene Tree will not be built. Set the API key to enable this feature.")
        build_scene_tree = False
    else:
        build_scene_tree = True
    
    # Extract frames
    result = extract_frames(video_path, output_folder)
    
    if result[0] is None:
        print("Failed to extract frames.")
        sys.exit(1)
    
    num_frames, is_vertical = result
    
    if num_frames and num_frames > 0:
        # Create collages (stack horizontally for vertical videos, vertically for horizontal)
        stack_horizontally = is_vertical
        collage_paths = create_stacked_collages(
            output_folder, 
            collages_folder,
            stack_horizontally=stack_horizontally
        )
        
        if build_scene_tree and collage_paths:
            print("\n" + "="*80)
            print("BUILDING SCENE TREE")
            print("="*80)
            
            try:
                # Initialize Scene Tree Builder
                builder = SceneTreeBuilder(
                    api_key=gemini_api_key,
                    embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
                )
                
                # Build the tree
                root = builder.build_tree(collage_paths)
                
                # Save the tree
                builder.save_tree(root, scene_tree_path)
                
                # Save as JSON as well
                SceneTreeBuilder.save_tree_as_json(root, scene_tree_json_path)
                
                print("\n" + "="*80)
                print("SCENE TREE SAVED")
                print("="*80)
                print(f"Tree saved to: {scene_tree_path}")
                print(f"Tree JSON saved to: {scene_tree_json_path}")
                print(f"Total nodes: {root.node_id + 1}")
                print(f"\nYou can now query the tree using: python query_scene_tree.py")
                
            except Exception as e:
                print(f"\nError building Scene Tree: {e}")
                print("The collages were created successfully, but Scene Tree building failed.")
        
        elif not build_scene_tree:
            print("\nSkipping Scene Tree building (no API key provided).")
            print(f"Collages saved in: {collages_folder}")


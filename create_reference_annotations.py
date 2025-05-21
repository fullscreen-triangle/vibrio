#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reference Video Annotation Tool

This utility script helps generate JSON annotation files for reference videos
that contain ground truth speed information. These annotations will be used
for comparing with algorithm performance.
"""

import os
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Create annotation files for reference videos")
    
    parser.add_argument(
        "--reference_dir", 
        type=str, 
        default="public/reference",
        help="Directory containing reference videos"
    )
    
    parser.add_argument(
        "--video_file", 
        type=str, 
        default=None,
        help="Specific video file to annotate (if None, process all videos in directory)"
    )
    
    parser.add_argument(
        "--manual_entry", 
        action="store_true",
        help="Enable manual entry of speed values"
    )
    
    return parser.parse_args()

def estimate_speed_from_video(video_path):
    """
    Attempt to extract speed information from video frames.
    This is a placeholder for actual OCR or other extraction methods.
    
    Args:
        video_path (str): Path to the reference video
        
    Returns:
        dict: Frame-by-frame speed data
    """
    print(f"Analyzing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Placeholder for actual speed extraction
    # In a real implementation, this would use OCR or other methods
    # to extract speed values from the frames
    
    # For demonstration, we'll generate synthetic speed data
    # that increases over time with some noise
    speeds = {}
    
    video_name = Path(video_path).stem
    
    # Generate synthetic speed profiles based on activity type
    # (this should be replaced with actual extraction)
    if "downhill" in video_name.lower():
        # Downhill biking: starts moderate, peaks in middle
        base_speed = 40  # km/h
        for i in range(frame_count):
            progress = i / frame_count
            # Parabolic curve that peaks around 70% through the video
            speed_factor = 1.0 + 1.2 * (-(progress - 0.7) ** 2 + 0.5)
            speed = base_speed * speed_factor
            # Add some noise
            noise = np.random.normal(0, 2)
            speeds[str(i)] = {
                "frame": i,
                "speed": max(0, speed + noise),
                "timestamp": i / fps
            }
    
    elif "alpine" in video_name.lower() or "ski" in video_name.lower():
        # Speed skiing: starts slow, gets very fast
        base_speed = 30  # km/h
        for i in range(frame_count):
            progress = i / frame_count
            # Exponential increase
            speed_factor = 1.0 + 3.0 * (progress ** 2)
            speed = base_speed * speed_factor
            # Add some noise
            noise = np.random.normal(0, 3)
            speeds[str(i)] = {
                "frame": i,
                "speed": max(0, speed + noise),
                "timestamp": i / fps
            }
    
    elif "wingsuit" in video_name.lower() or "flying" in video_name.lower():
        # Wingsuit: high speed throughout with some variations
        base_speed = 120  # km/h
        for i in range(frame_count):
            progress = i / frame_count
            # Slight variations with small dips and rises
            variation = np.sin(progress * 10) * 15
            speed = base_speed + variation
            # Add some noise
            noise = np.random.normal(0, 5)
            speeds[str(i)] = {
                "frame": i,
                "speed": max(0, speed + noise),
                "timestamp": i / fps
            }
    
    else:
        # Generic profile for unknown activity
        base_speed = 50  # km/h
        for i in range(frame_count):
            progress = i / frame_count
            # Linear increase
            speed_factor = 1.0 + progress * 0.5
            speed = base_speed * speed_factor
            # Add some noise
            noise = np.random.normal(0, 4)
            speeds[str(i)] = {
                "frame": i,
                "speed": max(0, speed + noise),
                "timestamp": i / fps
            }
    
    cap.release()
    return speeds

def manual_annotation(video_path):
    """
    Allow manual annotation of speed values at key frames.
    
    Args:
        video_path (str): Path to the reference video
        
    Returns:
        dict: Frame-by-frame speed data
    """
    print(f"Manual annotation mode for: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Ask for number of key points to annotate
    try:
        num_keypoints = int(input("Enter number of key frames to annotate (recommended: 5-10): "))
    except ValueError:
        print("Invalid input. Using 5 keypoints.")
        num_keypoints = 5
    
    # Calculate frame indices for key frames (evenly distributed)
    keyframe_indices = [int(i * frame_count / (num_keypoints - 1)) for i in range(num_keypoints)]
    keyframe_indices[-1] = min(keyframe_indices[-1], frame_count - 1)  # Ensure last frame is valid
    
    keyframe_speeds = {}
    
    # Show each key frame and ask for speed
    for idx in keyframe_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {idx}")
            continue
        
        # Resize if too large for display
        max_display_height = 800
        if height > max_display_height:
            scale = max_display_height / height
            display_width = int(width * scale)
            display_height = max_display_height
            display_frame = cv2.resize(frame, (display_width, display_height))
        else:
            display_frame = frame
        
        # Display frame
        window_name = "Reference Frame"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_frame)
        cv2.waitKey(100)  # Short delay to ensure window updates
        
        # Ask for speed
        try:
            timestamp = idx / fps
            print(f"Frame {idx} (Time: {timestamp:.2f}s):")
            speed = float(input("Enter speed in km/h: "))
            keyframe_speeds[idx] = speed
            print(f"Recorded: Frame {idx} - {speed} km/h")
        except ValueError:
            print(f"Invalid input for frame {idx}. Skipping.")
        
        cv2.waitKey(100)
    
    cv2.destroyAllWindows()
    cap.release()
    
    # Interpolate speeds for all frames
    speeds = {}
    keyframes = sorted(keyframe_speeds.keys())
    
    for i in range(frame_count):
        # Find surrounding keyframes
        if i <= keyframes[0]:
            # Before first keyframe
            speed = keyframe_speeds[keyframes[0]]
        elif i >= keyframes[-1]:
            # After last keyframe
            speed = keyframe_speeds[keyframes[-1]]
        else:
            # Between keyframes - linear interpolation
            for k in range(len(keyframes) - 1):
                if keyframes[k] <= i < keyframes[k + 1]:
                    # Linear interpolation
                    frame_range = keyframes[k + 1] - keyframes[k]
                    speed_range = keyframe_speeds[keyframes[k + 1]] - keyframe_speeds[keyframes[k]]
                    progress = (i - keyframes[k]) / frame_range
                    speed = keyframe_speeds[keyframes[k]] + progress * speed_range
                    break
        
        speeds[str(i)] = {
            "frame": i,
            "speed": speed,
            "timestamp": i / fps
        }
    
    return speeds

def create_annotation_file(video_path, manual=False):
    """
    Create annotation file for a reference video.
    
    Args:
        video_path (str): Path to the reference video
        manual (bool): Whether to use manual annotation
        
    Returns:
        str: Path to the created annotation file
    """
    # Check if annotation file already exists
    json_path = video_path.replace('.mp4', '.json').replace('.avi', '.json').replace('.mov', '.json').replace('.mkv', '.json')
    
    if os.path.exists(json_path):
        overwrite = input(f"Annotation file already exists for {video_path}. Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            print("Skipping file.")
            return json_path
    
    # Get speed data
    if manual:
        speeds = manual_annotation(video_path)
    else:
        speeds = estimate_speed_from_video(video_path)
    
    # Save to JSON file
    with open(json_path, 'w') as f:
        json.dump(speeds, f, indent=4)
    
    print(f"Created annotation file: {json_path}")
    return json_path

def main():
    args = parse_args()
    
    if args.video_file:
        # Process single file
        video_path = args.video_file
        if not os.path.exists(video_path):
            video_path = os.path.join(args.reference_dir, args.video_file)
            
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return
            
        create_annotation_file(video_path, manual=args.manual_entry)
    else:
        # Process all videos in directory
        if not os.path.isdir(args.reference_dir):
            print(f"Error: Reference directory not found: {args.reference_dir}")
            return
            
        video_files = [os.path.join(args.reference_dir, f) for f in os.listdir(args.reference_dir)
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"Found {len(video_files)} video files in {args.reference_dir}")
        
        for video_path in video_files:
            print(f"\nProcessing: {os.path.basename(video_path)}")
            create_annotation_file(video_path, manual=args.manual_entry)
    
    print("\nAnnotation creation complete!")

if __name__ == "__main__":
    main() 
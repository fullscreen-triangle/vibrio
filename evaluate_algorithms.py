#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vibrio Algorithm Evaluation Script

This script analyzes videos in the clips folder using various algorithms
and compares the results with reference videos containing ground truth data.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from tqdm import tqdm

# Import Vibrio modules
from modules.optical_analysis import OpticalAnalyzer
from modules import HumanDetector, PoseDetector, SpeedEstimator
from modules.utils import create_dir_if_not_exists

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Vibrio algorithms against reference data")
    
    parser.add_argument(
        "--clips_dir", 
        type=str, 
        default="public/clips",
        help="Directory containing clips to analyze"
    )
    
    parser.add_argument(
        "--reference_dir", 
        type=str, 
        default="public/reference",
        help="Directory containing reference videos with speed annotations"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/algorithm_evaluation",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--optical_methods",
        nargs="+",
        default=["optical_flow", "motion_energy", "neuromorphic", "texture_analysis", "shadow_analysis"],
        choices=["optical_flow", "motion_energy", "neuromorphic", "texture_analysis", "shadow_analysis"],
        help="Optical analysis methods to evaluate"
    )
    
    parser.add_argument(
        "--detector", 
        type=str, 
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Human detector model to use"
    )
    
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.4,
        help="Confidence threshold for detection"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        help="Device to run inference on (cuda:0, cpu, etc.)"
    )
    
    return parser.parse_args()

def extract_reference_speeds(reference_video_path):
    """
    Extract speed annotations from reference videos.
    
    Args:
        reference_video_path (str): Path to the reference video
        
    Returns:
        dict: Frame-by-frame speed data
    """
    # Check if there's a corresponding JSON file with annotations
    json_path = reference_video_path.replace('.mp4', '.json')
    
    if os.path.exists(json_path):
        # Load JSON annotations
        with open(json_path, 'r') as f:
            annotations = json.load(f)
        return annotations
    
    # If no JSON file exists, try to extract from video
    # This is a placeholder - in a real implementation,
    # we would need OCR or a specific extraction method
    # tailored to how the reference videos are annotated
    print(f"No annotation file found for {reference_video_path}")
    print("Attempting to extract annotations from video frames...")
    
    cap = cv2.VideoCapture(reference_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open reference video {reference_video_path}")
        return {}
    
    speeds = {}
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Placeholder for annotation extraction
        # This should be replaced with actual OCR or other extraction methods
        # based on how the reference videos are annotated
        speeds[frame_idx] = {"frame": frame_idx, "speed": None}
        
        frame_idx += 1
    
    cap.release()
    return speeds

def analyze_clip(clip_path, output_dir, args):
    """
    Analyze a single clip using various algorithms.
    
    Args:
        clip_path (str): Path to the clip
        output_dir (str): Directory to save results
        args: Command line arguments
        
    Returns:
        dict: Analysis results
    """
    clip_name = Path(clip_path).stem
    clip_results_dir = os.path.join(output_dir, clip_name)
    create_dir_if_not_exists(clip_results_dir)
    
    # Initialize components
    detector = HumanDetector(
        model_path="yolov8n.pt" if args.detector == "yolov8n" else f"{args.detector}.pt",
        conf_threshold=args.confidence,
        device=args.device
    )
    
    pose_detector = PoseDetector(device=args.device)
    speed_estimator = SpeedEstimator()
    
    # Initialize optical analyzer
    optical_analyzer = OpticalAnalyzer(
        output_dir=os.path.join(clip_results_dir, "optical_analysis"),
        visualization_dir=os.path.join(clip_results_dir, "visualizations")
    )
    
    # Run optical analysis
    print(f"Running optical analysis on {clip_path}...")
    optical_results = optical_analyzer.analyze_video(
        clip_path, 
        methods=args.optical_methods,
        output_video=True,
        output_data=True
    )
    
    # Analyze video with human detection and speed estimation
    print(f"Running human detection and speed estimation on {clip_path}...")
    
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {clip_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frame by frame
    speeds_by_frame = []
    poses_by_frame = []
    
    pbar = tqdm(total=frame_count, desc=f"Processing {clip_name}")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Human detection
        detections = detector.detect(frame)
        
        # Pose estimation
        poses = pose_detector.detect_pose(frame, detections)
        poses_by_frame.append(poses)
        
        # Speed estimation for each detection
        frame_speeds = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Use the speed estimator
            speed = speed_estimator.estimate_from_optical_flow(
                prev_frame=frame if frame_idx == 0 else prev_frame,
                curr_frame=frame,
                bbox=[x1, y1, x2, y2],
                fps=fps
            ) if frame_idx > 0 else 0
            
            frame_speeds.append({
                'bbox': [x1, y1, x2, y2],
                'speed': speed
            })
        
        speeds_by_frame.append({
            'frame': frame_idx,
            'speeds': frame_speeds
        })
        
        # Store current frame for next iteration
        prev_frame = frame.copy()
        frame_idx += 1
        pbar.update(1)
    
    # Close video capture
    cap.release()
    pbar.close()
    
    # Save results
    speed_results_path = os.path.join(clip_results_dir, f"{clip_name}_speeds.json")
    with open(speed_results_path, 'w') as f:
        json.dump(speeds_by_frame, f, indent=2)
    
    # Combine results
    results = {
        'clip_name': clip_name,
        'optical_results': optical_results,
        'speeds': speeds_by_frame
    }
    
    return results

def compare_with_reference(clip_results, reference_data, output_dir):
    """
    Compare algorithm results with reference data.
    
    Args:
        clip_results (dict): Results from algorithm analysis
        reference_data (dict): Reference speed data
        output_dir (str): Directory to save comparison results
        
    Returns:
        dict: Comparison metrics
    """
    clip_name = clip_results['clip_name']
    comparison_dir = os.path.join(output_dir, clip_name, "comparison")
    create_dir_if_not_exists(comparison_dir)
    
    # Extract speed estimates from algorithms
    algorithm_speeds = []
    for frame_data in clip_results['speeds']:
        frame_idx = frame_data['frame']
        
        # Use max speed if multiple detections
        if frame_data['speeds']:
            max_speed = max([det['speed'] for det in frame_data['speeds']])
            algorithm_speeds.append({
                'frame': frame_idx,
                'speed': max_speed
            })
        else:
            algorithm_speeds.append({
                'frame': frame_idx,
                'speed': 0
            })
    
    # Convert to DataFrame for easier analysis
    algo_df = pd.DataFrame(algorithm_speeds)
    
    # Convert reference data to DataFrame
    ref_frames = list(reference_data.keys())
    ref_speeds = [reference_data[frame].get('speed', 0) for frame in ref_frames]
    ref_df = pd.DataFrame({
        'frame': ref_frames,
        'speed': ref_speeds
    })
    
    # Merge data for comparison (if frames match)
    if len(ref_df) == len(algo_df):
        comparison_df = pd.DataFrame({
            'frame': algo_df['frame'],
            'algorithm_speed': algo_df['speed'],
            'reference_speed': ref_df['speed']
        })
    else:
        # Simplistic approach - may need to be refined
        # for actual data if frame counts don't match
        print(f"Warning: Frame count mismatch for {clip_name}")
        print(f"  Algorithm frames: {len(algo_df)}")
        print(f"  Reference frames: {len(ref_df)}")
        
        # Use common frames
        max_frames = min(len(algo_df), len(ref_df))
        comparison_df = pd.DataFrame({
            'frame': algo_df['frame'][:max_frames],
            'algorithm_speed': algo_df['speed'][:max_frames],
            'reference_speed': ref_df['speed'][:max_frames]
        })
    
    # Calculate error metrics
    # Filter out frames where reference speed is None
    valid_df = comparison_df[comparison_df['reference_speed'].notna()]
    
    if len(valid_df) > 0:
        # Calculate metrics
        mae = np.mean(np.abs(valid_df['algorithm_speed'] - valid_df['reference_speed']))
        rmse = np.sqrt(np.mean((valid_df['algorithm_speed'] - valid_df['reference_speed'])**2))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'num_valid_frames': len(valid_df)
        }
    else:
        metrics = {
            'mae': None,
            'rmse': None,
            'num_valid_frames': 0
        }
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    plt.plot(valid_df['frame'], valid_df['algorithm_speed'], label='Algorithm Estimate')
    plt.plot(valid_df['frame'], valid_df['reference_speed'], label='Reference')
    plt.xlabel('Frame')
    plt.ylabel('Speed (km/h)')
    plt.title(f'Speed Comparison - {clip_name}')
    plt.legend()
    plt.grid(True)
    
    # Add metrics to plot
    if metrics['mae'] is not None:
        plt.annotate(f"MAE: {metrics['mae']:.2f} km/h\nRMSE: {metrics['rmse']:.2f} km/h",
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # Save plot
    plot_path = os.path.join(comparison_dir, f"{clip_name}_speed_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(comparison_dir, f"{clip_name}_comparison.csv"), index=False)
    
    return metrics

def main():
    """Main function to run the evaluation."""
    args = parse_args()
    
    # Create output directory
    create_dir_if_not_exists(args.output_dir)
    
    # Get all clips
    clips = [os.path.join(args.clips_dir, f) for f in os.listdir(args.clips_dir) 
             if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    # Get all reference videos
    reference_videos = [os.path.join(args.reference_dir, f) for f in os.listdir(args.reference_dir)
                       if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    print(f"Found {len(clips)} clips to analyze")
    print(f"Found {len(reference_videos)} reference videos")
    
    # Map clips to reference videos (by similar name)
    clip_reference_pairs = []
    for clip in clips:
        clip_name = Path(clip).stem
        
        # Find matching reference video
        matching_ref = None
        for ref in reference_videos:
            ref_name = Path(ref).stem
            # Simple matching logic - adapt as needed
            if clip_name.lower() in ref_name.lower() or ref_name.lower() in clip_name.lower():
                matching_ref = ref
                break
        
        clip_reference_pairs.append((clip, matching_ref))
    
    # Process each clip and compare with reference
    overall_metrics = []
    
    for clip_path, reference_path in clip_reference_pairs:
        clip_name = Path(clip_path).stem
        print(f"\nProcessing clip: {clip_name}")
        
        # Analyze clip
        clip_results = analyze_clip(clip_path, args.output_dir, args)
        
        # Extract reference data
        reference_data = {}
        if reference_path:
            print(f"Comparing with reference: {Path(reference_path).name}")
            reference_data = extract_reference_speeds(reference_path)
        else:
            print(f"No matching reference found for {clip_name}")
        
        # Compare results if reference data is available
        if reference_data and clip_results:
            metrics = compare_with_reference(clip_results, reference_data, args.output_dir)
            
            # Add to overall metrics
            overall_metrics.append({
                'clip_name': clip_name,
                **metrics
            })
    
    # Calculate and save overall metrics
    if overall_metrics:
        metrics_df = pd.DataFrame(overall_metrics)
        metrics_df.to_csv(os.path.join(args.output_dir, "overall_metrics.csv"), index=False)
        
        # Print summary
        print("\nOverall Metrics:")
        print(metrics_df)
        
        # Create summary plot
        valid_metrics = metrics_df[metrics_df['mae'].notna()]
        if len(valid_metrics) > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(valid_metrics['clip_name'], valid_metrics['mae'], alpha=0.7, label='MAE')
            plt.bar(valid_metrics['clip_name'], valid_metrics['rmse'], alpha=0.7, label='RMSE')
            plt.xlabel('Clip')
            plt.ylabel('Error (km/h)')
            plt.title('Algorithm Performance Across Clips')
            plt.legend()
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "performance_summary.png"), dpi=300)
            plt.close()
    
    print(f"\nEvaluation complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch analysis script for Vibrio

This script processes multiple videos in a directory using the Vibrio framework.
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import Vibrio modules
from modules import HumanDetector, HumanTracker, SpeedEstimator, PhysicsVerifier, Visualizer
from modules import PoseDetector, OpticalAnalyzer
from modules.utils import create_dir_if_not_exists

def parse_args():
    parser = argparse.ArgumentParser(description="Batch analysis of videos with Vibrio")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing input videos"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    
    parser.add_argument(
        "--detector", 
        type=str, 
        default="yolov8x",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Human detector model to use"
    )
    
    parser.add_argument(
        "--use_pose", 
        action="store_true",
        help="Use pose estimation for additional analysis"
    )
    
    parser.add_argument(
        "--use_optical", 
        action="store_true",
        help="Use advanced optical analysis methods"
    )
    
    parser.add_argument(
        "--optical_methods",
        nargs="+",
        default=["optical_flow", "motion_energy"],
        choices=["optical_flow", "motion_energy", "neuromorphic", "texture_analysis", "shadow_analysis"],
        help="Optical analysis methods to use (if --use_optical is set)"
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
    
    parser.add_argument(
        "--skip_visualization", 
        action="store_true",
        help="Skip visualization step"
    )
    
    parser.add_argument(
        "--video_ext", 
        type=str, 
        default=".mp4,.avi,.mov,.mkv",
        help="Comma-separated list of video extensions to process"
    )
    
    return parser.parse_args()

def process_video(video_path, output_dir, args):
    """Process a single video with the Vibrio framework"""
    
    print(f"Processing: {video_path}")
    
    # Create output directory for this video
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    create_dir_if_not_exists(video_output_dir)
    
    # Initialize components
    detector = HumanDetector(
        model_path="yolov8n.pt" if args.detector == "yolov8n" else f"{args.detector}.pt",
        conf_threshold=args.confidence,
        device=args.device
    )
    
    tracker = HumanTracker()
    speed_estimator = SpeedEstimator()
    physics_verifier = PhysicsVerifier()
    
    # Optional: Initialize pose detector if requested
    pose_detector = None
    if args.use_pose:
        pose_detector = PoseDetector(device=args.device)
    
    # Optional: Initialize optical analyzer if requested
    optical_analyzer = None
    if args.use_optical:
        optical_analyzer = OpticalAnalyzer(
            output_dir=os.path.join(video_output_dir, "optical_analysis"),
            visualization_dir=os.path.join(video_output_dir, "visualizations")
        )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize videowriter for annotated output
    annotated_video_path = os.path.join(video_output_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))
    
    # Process frame by frame
    tracks = []
    detections_by_frame = []
    poses_by_frame = []
    
    prev_frame = None
    frame_idx = 0
    
    # Setup tqdm progress bar
    pbar = tqdm(total=frame_count, desc=f"Processing {video_name}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Human detection
        detections = detector.detect(frame)
        detections_by_frame.append(detections)
        
        # Tracking
        tracks = tracker.update(detections, frame_idx)
        
        # Speed estimation
        if len(tracks) > 0:
            tracks = speed_estimator.estimate(tracks, frame_idx=frame_idx, fps=fps)
            tracks = physics_verifier.verify(tracks)
        
        # Pose estimation (optional)
        poses = None
        if pose_detector is not None and args.use_pose:
            poses = pose_detector.detect_pose(frame, detections)
            poses_by_frame.append(poses)
        
        # Optical analysis (executed on frame pairs)
        if optical_analyzer is not None and args.use_optical and prev_frame is not None:
            # Process with selected optical methods (simplified for batch process)
            # For full analysis, use analyze_video separately on the complete video
            for method in args.optical_methods:
                if method == "optical_flow":
                    # Just calculate optical flow for visualization in this loop
                    # Full analysis is done separately
                    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                                      0.5, 3, 15, 3, 5, 1.2, 0)
                    # Visualize flow on frame
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv = np.zeros_like(frame)
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    hsv[..., 2] = 255
                    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                    frame = cv2.addWeighted(frame, 0.7, flow_vis, 0.3, 0)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw bounding boxes and tracking info
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            track_id = track['track_id']
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw track ID and speed
            if 'speed' in track:
                speed_text = f"ID: {track_id}, Speed: {track['speed']:.1f} km/h"
                cv2.putText(annotated_frame, speed_text, (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw pose keypoints if available
        if poses is not None:
            for person_id, keypoints in poses.items():
                for kp in keypoints:
                    if kp[2] > 0.5:  # Confidence threshold
                        cv2.circle(annotated_frame, (int(kp[0]), int(kp[1])), 5, (255, 0, 0), -1)
        
        # Add frame number
        cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame to video
        videowriter.write(annotated_frame)
        
        # Store current frame for next iteration (needed for optical flow)
        prev_frame = frame.copy()
        
        # Increment frame counter
        frame_idx += 1
        pbar.update(1)
    
    # Close video capture and writer
    cap.release()
    videowriter.release()
    pbar.close()
    
    # Save tracking results
    import pandas as pd
    
    # Convert tracking results to DataFrame
    results = []
    for track in tracks:
        track_id = track['track_id']
        speed = track.get('speed', 0)
        
        # Additional info if available
        if 'trajectory' in track:
            for frame, bbox in track['trajectory'].items():
                results.append({
                    'track_id': track_id,
                    'frame': frame,
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'speed': track.get('speeds', {}).get(frame, speed)
                })
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(video_output_dir, 'tracking_results.csv'), index=False)
    
    # Run full optical analysis if requested
    if optical_analyzer is not None and args.use_optical:
        print(f"Running full optical analysis on {video_path}...")
        optical_analyzer.analyze_video(
            video_path=video_path,
            methods=args.optical_methods,
            output_video=True,
            output_data=True
        )
        
        # Run frequency analysis
        optical_analyzer.analyze_frequency_domain(
            video_path=video_path,
            output_video=True
        )
    
    print(f"Processing completed for {video_path}")
    print(f"Results saved to {video_output_dir}")
    print(f"Annotated video saved to {annotated_video_path}")
    
    return video_output_dir

def main():
    # Parse arguments
    args = parse_args()
    
    # Get list of video files
    video_extensions = args.video_ext.split(",")
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(list(Path(args.input_dir).glob(f"*{ext}")))
    
    if not video_files:
        print(f"No video files found in {args.input_dir} with extensions {video_extensions}")
        return
    
    print(f"Found {len(video_files)} video files for processing")
    
    # Create output directory
    create_dir_if_not_exists(args.output_dir)
    
    # Process each video
    results_dirs = []
    for video_file in video_files:
        video_path = str(video_file)
        results_dir = process_video(video_path, args.output_dir, args)
        results_dirs.append(results_dir)
    
    print(f"All videos processed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 
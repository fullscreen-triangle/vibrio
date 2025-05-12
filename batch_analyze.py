#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import csv
import json
from pathlib import Path
from main import process_video

from modules.detector import HumanDetector
from modules.tracker import HumanTracker
from modules.speed_estimator import SpeedEstimator
from modules.physics_verifier import PhysicsVerifier
from modules.visualizer import Visualizer
from modules.pose_detector import PoseDetector
from modules.posture_analyzer import PostureAnalyzer
from modules.utils import load_calibration, setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process videos with Vibrio')
    parser.add_argument('--input_dir', type=str, default='public', help='Directory containing videos')
    parser.add_argument('--output', type=str, default='results', help='Path to output directory')
    parser.add_argument('--calibration', type=str, help='Path to camera calibration file')
    parser.add_argument('--context', type=str, default='racing', 
                        choices=['general', 'sport', 'cycling', 'racing', 'athletics'],
                        help='Context for physics verification')
    parser.add_argument('--show', action='store_true', help='Show results in real-time')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--with_pose', action='store_true', help='Enable pose detection and skeleton visualization')
    parser.add_argument('--pose_model', type=str, default='ultralytics/yolov8s-pose', help='Pose model to use')
    parser.add_argument('--slow_motion', type=float, default=1.0, help='Slow motion factor (e.g., 0.5 for half speed)')
    parser.add_argument('--export_format', type=str, default='csv', choices=['csv', 'json', 'tab'], 
                        help='Format for exporting raw metrics data')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    logger = setup_logger(args.debug)
    logger.info(f"Processing videos from directory: {args.input_dir}")
    
    os.makedirs(args.output, exist_ok=True)
    calibration = load_calibration(args.calibration) if args.calibration else None
    
    # Initialize components
    detector = HumanDetector()
    tracker = HumanTracker()
    speed_estimator = SpeedEstimator(calibration)
    physics_verifier = PhysicsVerifier(context=args.context)
    
    # Optional components based on arguments
    pose_detector = None
    posture_analyzer = None
    if args.with_pose:
        pose_detector = PoseDetector(model_name=args.pose_model)
        posture_analyzer = PostureAnalyzer()
    
    # Initialize visualizer with additional options
    visualizer = Visualizer(
        output_dir=args.output, 
        show=args.show, 
        draw_skeleton=args.with_pose
    )
    
    # Get all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for file in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file_path)
    
    if not video_files:
        logger.error(f"No video files found in {args.input_dir}")
        return
    
    # Process each video
    all_results = {}
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        logger.info(f"Processing video: {video_name}")
        
        try:
            # Create a unique output directory for each video
            video_output_dir = os.path.join(args.output, os.path.splitext(video_name)[0])
            os.makedirs(video_output_dir, exist_ok=True)
            
            # Update visualizer output directory for this video
            visualizer.output_dir = video_output_dir
            
            # Process the video with additional components if enabled
            results = process_video(
                video_path, 
                detector, 
                tracker, 
                speed_estimator, 
                physics_verifier, 
                visualizer,
                pose_detector=pose_detector,
                posture_analyzer=posture_analyzer,
                slow_motion_factor=args.slow_motion
            )
            
            # Export raw metrics data in the requested format
            export_raw_metrics(results, video_output_dir, args.export_format)
            
            all_results[video_name] = results
            logger.info(f"Completed processing {video_name}")
            
        except Exception as e:
            logger.error(f"Error processing {video_name}: {str(e)}")
    
    logger.info(f"All videos processed. Results saved to {args.output}")
    
    # Print summary of maximum speeds and posture metrics if available
    print("\n===== ANALYSIS SUMMARY =====")
    for video_name, results in all_results.items():
        print(f"\nVideo: {video_name}")
        
        # Speed summary
        if 'speed_estimates' in results and results['speed_estimates']:
            max_speed = max([estimate['speed'] for estimate in results['speed_estimates']], default=0)
            avg_speed = sum([estimate['speed'] for estimate in results['speed_estimates']]) / len(results['speed_estimates'])
            print(f"  Maximum speed detected: {max_speed:.2f} km/h")
            print(f"  Average speed detected: {avg_speed:.2f} km/h")
        else:
            print(f"  No speed data available")
        
        # Posture metrics summary if available
        if 'posture_metrics' in results and results['posture_metrics']:
            print("  Posture Metrics:")
            
            # Group by track_id
            track_metrics = {}
            for metric_data in results['posture_metrics']:
                track_id = metric_data['track_id']
                if track_id not in track_metrics:
                    track_metrics[track_id] = []
                track_metrics[track_id].append(metric_data['metrics'])
            
            # Calculate averages for each track
            for track_id, metrics_list in track_metrics.items():
                avg_sway = sum([m['postural_sway'] for m in metrics_list]) / len(metrics_list)
                avg_energy = sum([m['locomotion_energy'] for m in metrics_list]) / len(metrics_list)
                avg_stability = sum([m['stability_score'] for m in metrics_list]) / len(metrics_list)
                
                print(f"    Track ID {track_id}:")
                print(f"      Avg Postural Sway: {avg_sway:.2f}")
                print(f"      Avg Locomotion Energy: {avg_energy:.2f}")
                print(f"      Avg Stability Score: {avg_stability:.2f}")
    
    return all_results

def export_raw_metrics(results, output_dir, format='csv'):
    """Export raw metrics data in the specified format"""
    
    # Export speed estimates
    if 'speed_estimates' in results and results['speed_estimates']:
        speed_file = os.path.join(output_dir, f'speed_data.{format}')
        export_data(results['speed_estimates'], speed_file, format)
    
    # Export pose data if available
    if 'pose_data' in results and results['pose_data']:
        pose_file = os.path.join(output_dir, f'pose_data.{format}')
        export_data(results['pose_data'], pose_file, format)
    
    # Export posture metrics if available
    if 'posture_metrics' in results and results['posture_metrics']:
        posture_file = os.path.join(output_dir, f'posture_metrics.{format}')
        export_data(results['posture_metrics'], posture_file, format)

def export_data(data, output_file, format='csv'):
    """Export data to the specified format"""
    if format == 'json':
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format == 'csv':
        # Get all fields from the first data item
        if not data:
            return
            
        # Handle nested structures
        fieldnames = flatten_dict_keys(data[0])
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in data:
                # Flatten nested dictionaries for CSV export
                flat_item = {}
                flatten_dict(item, flat_item)
                writer.writerow(flat_item)
    
    elif format == 'tab':
        # Similar to CSV but with tab separator
        if not data:
            return
            
        fieldnames = flatten_dict_keys(data[0])
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            
            for item in data:
                flat_item = {}
                flatten_dict(item, flat_item)
                writer.writerow(flat_item)

def flatten_dict_keys(d, prefix=''):
    """Get all keys from a nested dictionary, flattening nested structures"""
    keys = []
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            keys.extend(flatten_dict_keys(v, f"{key}_"))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            # Skip lists of dictionaries (hard to flatten in CSV)
            keys.append(key)
        else:
            keys.append(key)
    return keys

def flatten_dict(d, flat_d, prefix=''):
    """Flatten a nested dictionary into a single-level dictionary"""
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            flatten_dict(v, flat_d, f"{key}_")
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            # Convert list of dictionaries to string for CSV
            flat_d[key] = json.dumps(v)
        else:
            flat_d[key] = v

if __name__ == "__main__":
    main() 
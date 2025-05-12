#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import time
from pathlib import Path

from modules.detector import HumanDetector
from modules.tracker import HumanTracker
from modules.speed_estimator import SpeedEstimator
from modules.physics_verifier import PhysicsVerifier
from modules.visualizer import Visualizer
from modules.pose_detector import PoseDetector
from modules.posture_analyzer import PostureAnalyzer
from modules.utils import load_calibration, setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Vibrio: Human Speed Analysis Framework')
    parser.add_argument('--input', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='results', help='Path to output directory')
    parser.add_argument('--calibration', type=str, help='Path to camera calibration file')
    parser.add_argument('--context', type=str, default='general', 
                        choices=['general', 'sport', 'cycling', 'racing', 'athletics'],
                        help='Context for physics verification')
    parser.add_argument('--show', action='store_true', help='Show results in real-time')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--with_pose', action='store_true', help='Enable pose detection and skeleton visualization')
    parser.add_argument('--pose_model', type=str, default='ultralytics/yolov8s-pose', help='Pose model to use')
    parser.add_argument('--slow_motion', type=float, default=1.0, help='Slow motion factor (e.g., 0.5 for half speed)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup
    logger = setup_logger(args.debug)
    logger.info(f"Processing video: {args.input}")
    
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
    
    # Process video
    start_time = time.time()
    results = process_video(
        args.input, 
        detector, 
        tracker, 
        speed_estimator, 
        physics_verifier, 
        visualizer,
        pose_detector=pose_detector,
        posture_analyzer=posture_analyzer,
        slow_motion_factor=args.slow_motion
    )
    
    # Save and display results
    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f} seconds")
    logger.info(f"Results saved to {args.output}")
    
    return results

def process_video(video_path, detector, tracker, speed_estimator, physics_verifier, visualizer,
                  pose_detector=None, posture_analyzer=None, slow_motion_factor=1.0):
    """Process the video through the pipeline"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Adjust fps for slow motion if needed
    effective_fps = fps * slow_motion_factor
    
    results = {
        'video_info': {
            'path': video_path,
            'fps': fps,
            'effective_fps': effective_fps,
            'frame_count': frame_count,
            'resolution': (width, height)
        },
        'human_tracks': [],
        'speed_estimates': []
    }
    
    # Initialize results for new metrics if pose detection is enabled
    if pose_detector is not None:
        results['pose_data'] = []
        results['posture_metrics'] = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Human detection
        detections = detector.detect(frame)
        
        # Human tracking
        tracks = tracker.update(detections, frame)
        
        # Speed estimation
        speeds = speed_estimator.estimate(tracks, frame_idx, effective_fps)
        
        # Physics verification
        verified_speeds = physics_verifier.verify(speeds, tracks)
        
        # Initialize keypoints and posture metrics
        keypoints_data = {}
        posture_metrics = {}
        
        # Optional pose detection and analysis
        if pose_detector is not None:
            # Detect poses in the frame
            pose_results = pose_detector.detect(frame)
            
            # Map pose keypoints to tracks
            for track in tracks:
                track_id = track['id']
                track_bbox = track['bbox']
                
                # Find the best matching pose for this track
                best_match = None
                best_iou = 0
                
                for i, bbox in enumerate(pose_results['bboxes']):
                    # Calculate IoU between track and pose bounding boxes
                    iou = _calculate_iou(track_bbox, bbox)
                    
                    if iou > best_iou and iou > 0.5:  # Threshold for matching
                        best_iou = iou
                        best_match = i
                
                # If we found a matching pose, store the keypoints
                if best_match is not None:
                    keypoints_data[track_id] = pose_results['keypoints'][best_match]
            
            # Run posture analysis if we have pose data
            if posture_analyzer is not None and keypoints_data:
                posture_metrics = posture_analyzer.update(tracks, keypoints_data)
        
        # Visualization
        visualizer.visualize(frame, tracks, verified_speeds, frame_idx, 
                             keypoints=keypoints_data, posture_metrics=posture_metrics)
        
        # Store results
        for track_id, speed in verified_speeds.items():
            results['speed_estimates'].append({
                'frame': frame_idx,
                'track_id': track_id,
                'speed': speed,
                'timestamp': frame_idx / effective_fps
            })
        
        # Store pose and posture data if available
        if pose_detector is not None:
            for track_id, keypoints in keypoints_data.items():
                results['pose_data'].append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'keypoints': keypoints,
                    'timestamp': frame_idx / effective_fps
                })
            
            for track_id, metrics in posture_metrics.items():
                results['posture_metrics'].append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'metrics': metrics,
                    'timestamp': frame_idx / effective_fps
                })
        
        frame_idx += 1
    
    cap.release()
    visualizer.finalize()
    
    return results

def _calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1[:4]
    x1_2, y1_2, x2_2, y2_2 = box2[:4]
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes intersect
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # Calculate intersection area
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    union = area1 + area2 - intersection
    
    # Calculate IoU
    return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    main() 
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
    visualizer = Visualizer(output_dir=args.output, show=args.show)
    
    # Process video
    start_time = time.time()
    results = process_video(
        args.input, 
        detector, 
        tracker, 
        speed_estimator, 
        physics_verifier, 
        visualizer
    )
    
    # Save and display results
    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f} seconds")
    logger.info(f"Results saved to {args.output}")
    
    return results

def process_video(video_path, detector, tracker, speed_estimator, physics_verifier, visualizer):
    """Process the video through the pipeline"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    results = {
        'video_info': {
            'path': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'resolution': (width, height)
        },
        'human_tracks': [],
        'speed_estimates': []
    }
    
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
        speeds = speed_estimator.estimate(tracks, frame_idx, fps)
        
        # Physics verification
        verified_speeds = physics_verifier.verify(speeds, tracks)
        
        # Visualization
        visualizer.visualize(frame, tracks, verified_speeds, frame_idx)
        
        # Store results
        for track_id, speed in verified_speeds.items():
            results['speed_estimates'].append({
                'frame': frame_idx,
                'track_id': track_id,
                'speed': speed,
                'timestamp': frame_idx / fps
            })
        
        frame_idx += 1
    
    cap.release()
    visualizer.finalize()
    
    return results

if __name__ == "__main__":
    main() 
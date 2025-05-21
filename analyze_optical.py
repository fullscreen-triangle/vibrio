#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vibrio Optical Analysis Tool

This script provides advanced optical analysis methods for motion analysis in videos.
It implements methods described in the optical_methods.md documentation.
"""

import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from modules.optical_analysis import OpticalAnalyzer, analyze_video_with_optical_methods

def parse_args():
    parser = argparse.ArgumentParser(description='Vibrio Optical Analysis Tool')
    
    parser.add_argument(
        '--video_path', 
        type=str, 
        required=True,
        help='Path to video file or directory containing videos'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='results/optical_analysis',
        help='Path to output directory for analysis results'
    )
    
    parser.add_argument(
        '--methods', 
        type=str, 
        nargs='+', 
        default=['optical_flow', 'motion_energy', 'neuromorphic', 'texture_analysis', 'shadow_analysis'],
        help='Optical analysis methods to use'
    )
    
    parser.add_argument(
        '--frequency_analysis', 
        action='store_true',
        help='Perform frequency domain analysis to identify repetitive movements'
    )
    
    parser.add_argument(
        '--roi', 
        type=int, 
        nargs=4, 
        default=None,
        help='Region of interest for frequency analysis (x y width height)'
    )
    
    parser.add_argument(
        '--no_video', 
        action='store_true',
        help='Do not output annotated videos'
    )
    
    parser.add_argument(
        '--no_data', 
        action='store_true',
        help='Do not output numerical data'
    )
    
    return parser.parse_args()

def validate_args(args):
    """Validate the command line arguments"""
    # Check that video path exists
    if not os.path.exists(args.video_path):
        raise ValueError(f"Video path not found: {args.video_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizations directory if outputting videos
    if not args.no_video:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

def process_video(video_path, args):
    """Process a single video with optical analysis methods"""
    print(f"Processing video: {video_path}")
    
    # Initialize analyzer
    analyzer = OpticalAnalyzer(
        output_dir=args.output_dir,
        visualization_dir=os.path.join(args.output_dir, 'visualizations')
    )
    
    # Run analysis
    results = analyzer.analyze_video(
        video_path=video_path,
        methods=args.methods,
        output_video=not args.no_video,
        output_data=not args.no_data
    )
    
    # Perform frequency analysis if requested
    if args.frequency_analysis:
        print("Performing frequency domain analysis...")
        freq_results = analyzer.analyze_frequency_domain(
            video_path=video_path,
            roi=args.roi,
            output_video=not args.no_video
        )
        
        results['frequency_analysis'] = freq_results
    
    return results

def process_directory(directory_path, args):
    """Process all videos in a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    
    # Get all video files in the directory
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(Path(directory_path).glob(f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in {directory_path}")
        return
    
    print(f"Found {len(video_files)} video files in {directory_path}")
    
    # Process each video
    for video_file in video_files:
        video_path = str(video_file)
        process_video(video_path, args)

def main():
    """Main function to run the optical analysis tool"""
    # Parse and validate arguments
    args = parse_args()
    validate_args(args)
    
    # Check if input is a file or directory
    if os.path.isfile(args.video_path):
        # Process single video
        process_video(args.video_path, args)
    elif os.path.isdir(args.video_path):
        # Process all videos in directory
        process_directory(args.video_path, args)
    else:
        print(f"Invalid path: {args.video_path}")
    
    print(f"All optical analysis completed. Output saved to {args.output_dir}")

if __name__ == '__main__':
    main() 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vibrio Scientific Visualization Tool

This script provides scientific-quality visualization of Vibrio analysis results.
It can visualize a single result directory or compare multiple result directories.
"""

import os
import argparse
from pathlib import Path
from modules.scientific_visualizer import ScientificVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Vibrio Scientific Visualization Tool')
    
    parser.add_argument(
        '--results_dir', 
        type=str, 
        nargs='+', 
        required=True,
        help='Path to results directory or directories (for comparison)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help='Path to output directory (default: same as input)'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['single', 'comparison'], 
        default='single',
        help='Analysis mode: single (analyze each result separately) or comparison (compare multiple results)'
    )
    
    parser.add_argument(
        '--labels', 
        type=str, 
        nargs='+', 
        default=None,
        help='Labels for each results directory in comparison mode'
    )
    
    parser.add_argument(
        '--style', 
        type=str, 
        choices=['science', 'ieee', 'default'], 
        default='science',
        help='Plot style for scientific visualizations'
    )
    
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300,
        help='DPI for output images'
    )
    
    return parser.parse_args()

def validate_args(args):
    """Validate the command line arguments"""
    # Check that results directories exist
    for results_dir in args.results_dir:
        if not os.path.isdir(results_dir):
            raise ValueError(f"Results directory not found: {results_dir}")
    
    # If comparison mode, check that we have at least 2 results directories
    if args.mode == 'comparison' and len(args.results_dir) < 2:
        raise ValueError("Comparison mode requires at least 2 results directories")
    
    # If labels are provided, check that the number matches the number of results directories
    if args.labels and len(args.labels) != len(args.results_dir):
        raise ValueError("Number of labels must match number of results directories")

def main():
    """Main function to run the scientific visualization tool"""
    # Parse and validate arguments
    args = parse_args()
    validate_args(args)
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.results_dir[0]
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the scientific visualizer
    visualizer = ScientificVisualizer(
        output_dir=output_dir,
        style=args.style,
        dpi=args.dpi
    )
    
    # Process based on mode
    if args.mode == 'single':
        # Process each results directory separately
        for results_dir in args.results_dir:
            print(f"Analyzing results directory: {results_dir}")
            visualizer.visualize_single_result(results_dir)
    
    else:  # Comparison mode
        # Get labels if provided, otherwise use directory names
        labels = args.labels if args.labels else [os.path.basename(d) for d in args.results_dir]
        
        print(f"Comparing {len(args.results_dir)} results directories")
        visualizer.visualize_comparison(
            args.results_dir,
            labels=labels,
            output_dir=output_dir
        )
    
    print(f"All visualizations completed. Output saved to {output_dir}")

if __name__ == '__main__':
    main() 
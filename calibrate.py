#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool for Vibrio')
    parser.add_argument('--input', type=str, required=True, help='Path to calibration video file')
    parser.add_argument('--output', type=str, default='calibration.json', 
                       help='Path to output calibration file')
    parser.add_argument('--reference_height', type=float, default=1.7,
                       help='Reference human height in meters (default: 1.7m)')
    parser.add_argument('--reference_distance', type=float, default=5.0,
                       help='Reference distance in meters (default: 5.0m)')
    parser.add_argument('--show', action='store_true', help='Show calibration process')
    
    return parser.parse_args()

def draw_crosshair(frame, x, y, size=20, color=(0, 255, 0), thickness=2):
    """Draw a crosshair on the frame"""
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
    return frame

class CameraCalibrator:
    def __init__(self, reference_height=1.7, reference_distance=5.0):
        """
        Initialize the camera calibrator
        
        Args:
            reference_height (float): Height of reference person in meters
            reference_distance (float): Distance from camera in meters
        """
        self.reference_height = reference_height
        self.reference_distance = reference_distance
        
        # Points selected by user
        self.top_point = None
        self.bottom_point = None
        self.left_point = None
        self.right_point = None
        self.center_point = None
        
        # Calibration state
        self.current_selection = 'center'
        self.calibration_complete = False
        
        # Frame dimensions
        self.frame_width = None
        self.frame_height = None
        
        # Original frame to restore after drawing
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for point selection"""
        if self.calibration_complete:
            return
            
        # Update display with crosshair
        frame_copy = self.original_frame.copy()
        
        # Draw existing points
        if self.center_point:
            cv2.circle(frame_copy, self.center_point, 5, (0, 255, 0), -1)
            cv2.putText(frame_copy, "Center", (self.center_point[0] + 10, self.center_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        if self.top_point:
            cv2.circle(frame_copy, self.top_point, 5, (255, 0, 0), -1)
            cv2.putText(frame_copy, "Top", (self.top_point[0] + 10, self.top_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        if self.bottom_point:
            cv2.circle(frame_copy, self.bottom_point, 5, (255, 0, 0), -1)
            cv2.putText(frame_copy, "Bottom", (self.bottom_point[0] + 10, self.bottom_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        if self.left_point:
            cv2.circle(frame_copy, self.left_point, 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, "Left", (self.left_point[0] + 10, self.left_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        if self.right_point:
            cv2.circle(frame_copy, self.right_point, 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, "Right", (self.right_point[0] + 10, self.right_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show current mouse position
        draw_crosshair(frame_copy, x, y)
            
        # Handle point selection on click
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_selection == 'center':
                self.center_point = (x, y)
                self.current_selection = 'top'
                
            elif self.current_selection == 'top':
                self.top_point = (x, y)
                self.current_selection = 'bottom'
                
            elif self.current_selection == 'bottom':
                self.bottom_point = (x, y)
                self.current_selection = 'left'
                
            elif self.current_selection == 'left':
                self.left_point = (x, y)
                self.current_selection = 'right'
                
            elif self.current_selection == 'right':
                self.right_point = (x, y)
                self.calibration_complete = True
                
            # Update instructions
            self._show_instructions(frame_copy)
            
        else:
            # Just update instructions
            self._show_instructions(frame_copy)
            
        cv2.imshow('Camera Calibration', frame_copy)
    
    def _show_instructions(self, frame):
        """Show instructions on the frame"""
        instructions = f"Click to mark the {self.current_selection.upper()} point"
        
        if self.calibration_complete:
            instructions = "Calibration complete! Press any key to continue."
            
        cv2.putText(frame, instructions, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def calibrate(self, video_path, show=True):
        """Run the calibration process"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Could not read frame from video")
            
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Store original frame
        self.original_frame = frame.copy()
        
        if show:
            # Create window and set mouse callback
            cv2.namedWindow('Camera Calibration')
            cv2.setMouseCallback('Camera Calibration', self.mouse_callback)
            
            # Display frame and wait for calibration
            self._show_instructions(frame)
            cv2.imshow('Camera Calibration', frame)
            
            while not self.calibration_complete:
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    return None
                    
            # Wait for user to acknowledge completion
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # For non-interactive mode, use default values
            self.center_point = (self.frame_width // 2, self.frame_height // 2)
            self.top_point = (self.frame_width // 2, 0)
            self.bottom_point = (self.frame_width // 2, self.frame_height)
            self.left_point = (0, self.frame_height // 2)
            self.right_point = (self.frame_width, self.frame_height // 2)
            self.calibration_complete = True
        
        # Generate calibration data
        return self._generate_calibration()
    
    def _generate_calibration(self):
        """Generate calibration data from selected points"""
        if not self.calibration_complete:
            return None
            
        # Calculate basic pixel-to-meter ratio
        # Use the height of the reference person
        if self.top_point and self.bottom_point:
            height_pixels = abs(self.bottom_point[1] - self.top_point[1])
            base_ratio = self.reference_height / height_pixels
        else:
            base_ratio = 0.01  # Default fallback value
            
        # Calculate scaling factors
        # These adjust for perspective distortion
        scale_factors = {
            "top": 0.5,  # Objects at the top appear smaller
            "bottom": 1.0,
            "left": 0.8,
            "right": 0.8
        }
        
        # Camera matrix (simplified)
        # This is a placeholder; a real calibration would use chessboard
        # patterns and proper camera calibration procedures
        fx = 1000.0  # focal length in pixels
        fy = 1000.0
        cx = self.frame_width / 2  # principal point
        cy = self.frame_height / 2
        
        if self.center_point:
            cx = self.center_point[0]
            cy = self.center_point[1]
            
        camera_matrix = [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]
        
        # Create calibration data
        calibration = {
            "camera_matrix": camera_matrix,
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],  # No distortion for simplicity
            "resolution": [self.frame_width, self.frame_height],
            "base_ratio": float(base_ratio),
            "reference_distance": float(self.reference_distance),
            "reference_point": [int(cx), int(cy)],
            "scale_factors": scale_factors
        }
        
        return calibration

def main():
    args = parse_args()
    
    # Create calibrator
    calibrator = CameraCalibrator(
        reference_height=args.reference_height,
        reference_distance=args.reference_distance
    )
    
    # Run calibration
    calibration = calibrator.calibrate(args.input, show=args.show)
    
    if calibration:
        # Save calibration to file
        with open(args.output, 'w') as f:
            json.dump(calibration, f, indent=4)
        
        print(f"Calibration saved to {args.output}")
    else:
        print("Calibration failed or was cancelled")

if __name__ == "__main__":
    main() 
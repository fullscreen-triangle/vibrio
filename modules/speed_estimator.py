#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

class SpeedEstimator:
    """Estimates human speed from tracked bounding boxes"""
    
    def __init__(self, calibration=None, pixel_to_meter_ratio=None):
        """
        Initialize the speed estimator
        
        Args:
            calibration (dict, optional): Camera calibration parameters
            pixel_to_meter_ratio (float, optional): Conversion ratio from pixels to meters
                If calibration is provided, this is ignored.
        """
        self.calibration = calibration
        self.pixel_to_meter_ratio = pixel_to_meter_ratio or 0.1  # Default if not provided
        
        # Detection smoothing parameters
        self.smoothing_window = 5  # frames
        
    def _compute_distance(self, point1, point2):
        """Compute Euclidean distance between two points"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    def _get_real_world_distance(self, pixel_distance, track):
        """
        Convert pixel distance to real-world distance using camera calibration
        or default pixel-to-meter ratio
        """
        if self.calibration:
            # If we have camera calibration, use it to get a more accurate conversion
            # This would account for perspective and camera lens distortion
            # Implementation depends on the format of the calibration data
            # For simplicity, we'll use a placeholder that adjusts based on y-position
            # (objects higher in the frame are typically further away)
            
            y_position = track['bbox'][1]  # Use bottom of bbox
            ratio = self.calibration.get('base_ratio', self.pixel_to_meter_ratio)
            
            # Simple adjustment based on position in frame - this can be replaced
            # with proper perspective transform with the actual calibration
            ratio_adjusted = ratio * (1 + 0.5 * (y_position / 720))  # Assume 720p video
            
            return pixel_distance * ratio_adjusted
        else:
            # Use simple pixel-to-meter ratio
            return pixel_distance * self.pixel_to_meter_ratio
    
    def _smooth_trajectory(self, history):
        """Apply smoothing to the trajectory to reduce noise"""
        if len(history) < self.smoothing_window:
            return history
        
        smoothed = []
        for i in range(len(history)):
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(history), i + self.smoothing_window // 2 + 1)
            window = history[start_idx:end_idx]
            smoothed.append([
                sum(p[0] for p in window) / len(window),
                sum(p[1] for p in window) / len(window)
            ])
        
        return smoothed
    
    def estimate(self, tracks, frame_idx, fps):
        """
        Estimate speeds of tracked humans
        
        Args:
            tracks (list): List of track states from the tracker
            frame_idx (int): Current frame index
            fps (float): Video framerate
            
        Returns:
            dict: Dictionary mapping track IDs to estimated speeds in km/h
        """
        speeds = {}
        
        for track in tracks:
            history = track['history']
            
            # Need at least two points to calculate speed
            if len(history) < 2:
                speeds[track['id']] = 0.0
                continue
            
            # Smooth the trajectory
            smoothed_history = self._smooth_trajectory(history)
            
            # Calculate displacement over the last N frames (or as many as we have)
            window_size = min(10, len(smoothed_history) - 1)
            prev_point = smoothed_history[-window_size-1]
            current_point = smoothed_history[-1]
            
            # Calculate pixel distance
            pixel_distance = self._compute_distance(prev_point, current_point)
            
            # Convert to real-world distance
            real_distance = self._get_real_world_distance(pixel_distance, track)
            
            # Calculate time elapsed
            time_elapsed = window_size / fps
            
            # Calculate speed in meters per second
            speed_mps = real_distance / time_elapsed if time_elapsed > 0 else 0
            
            # Convert to km/h
            speed_kmh = speed_mps * 3.6
            
            # Store result
            speeds[track['id']] = speed_kmh
        
        return speeds 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class PostureAnalyzer:
    """Analyzes human posture and calculates related metrics"""
    
    def __init__(self, window_size=10):
        """
        Initialize the posture analyzer
        
        Args:
            window_size (int): Size of the window for calculating moving averages
        """
        self.window_size = window_size
        self.pose_history = {}  # {track_id: [keypoints_history]}
        
    def update(self, tracks, keypoints_data=None):
        """
        Update the pose history with new data
        
        Args:
            tracks (list): List of track states with bounding boxes
            keypoints_data (dict, optional): Dict of keypoints for each track_id
            
        Returns:
            dict: Dict of posture metrics for each track_id
        """
        metrics = {}
        
        for track in tracks:
            track_id = track['id']
            
            # Get keypoints if provided
            keypoints = None
            if keypoints_data and track_id in keypoints_data:
                keypoints = keypoints_data[track_id]
            
            # Initialize track history if it doesn't exist
            if track_id not in self.pose_history:
                self.pose_history[track_id] = []
            
            # Add current keypoints to history
            if keypoints is not None:
                self.pose_history[track_id].append(keypoints)
                
                # Limit history length
                if len(self.pose_history[track_id]) > self.window_size:
                    self.pose_history[track_id].pop(0)
                
                # Calculate metrics if we have enough history
                if len(self.pose_history[track_id]) >= 3:
                    metrics[track_id] = self._calculate_metrics(track_id)
            
        return metrics
    
    def _calculate_metrics(self, track_id):
        """
        Calculate posture metrics for a track
        
        Args:
            track_id: ID of the track to calculate metrics for
            
        Returns:
            dict: Dict of calculated metrics
        """
        history = self.pose_history[track_id]
        
        # Calculate posture metrics
        postural_sway = self._calculate_postural_sway(history)
        locomotion_energy = self._calculate_locomotion_energy(history)
        stability_score = self._calculate_stability(history)
        
        # Return calculated metrics
        return {
            'postural_sway': postural_sway,
            'locomotion_energy': locomotion_energy,
            'stability_score': stability_score
        }
    
    def _calculate_postural_sway(self, keypoints_history):
        """
        Calculate postural sway (variation in center of mass position)
        
        Args:
            keypoints_history: List of keypoints history
            
        Returns:
            float: Postural sway value
        """
        # Get torso keypoints (shoulders, hips) for more stable reference
        torso_indices = [5, 6, 11, 12]  # Standard COCO keypoints for shoulders and hips
        
        centers = []
        for keypoints in keypoints_history:
            valid_points = []
            for idx in torso_indices:
                if idx < len(keypoints) and keypoints[idx][2] > 0.5:  # Check confidence
                    valid_points.append((keypoints[idx][0], keypoints[idx][1]))
            
            if valid_points:
                center = np.mean(valid_points, axis=0)
                centers.append(center)
        
        if len(centers) < 2:
            return 0.0
            
        # Calculate the variation in position (standard deviation)
        centers = np.array(centers)
        # Calculate std dev of position and return the Euclidean norm
        std_x = np.std(centers[:, 0])
        std_y = np.std(centers[:, 1])
        sway = np.sqrt(std_x**2 + std_y**2)
        
        return float(sway)
    
    def _calculate_locomotion_energy(self, keypoints_history):
        """
        Calculate locomotion energy (pixel changes over time)
        
        Args:
            keypoints_history: List of keypoints history
            
        Returns:
            float: Locomotion energy value
        """
        if len(keypoints_history) < 2:
            return 0.0
        
        # Calculate the sum of displacement of all keypoints between frames
        total_energy = 0.0
        
        for i in range(1, len(keypoints_history)):
            prev_keypoints = keypoints_history[i-1]
            curr_keypoints = keypoints_history[i]
            
            frame_energy = 0.0
            valid_pairs = 0
            
            # Calculate displacement for each keypoint
            for j in range(min(len(prev_keypoints), len(curr_keypoints))):
                # Check if both keypoints are valid (confidence > 0.5)
                if prev_keypoints[j][2] > 0.5 and curr_keypoints[j][2] > 0.5:
                    prev_pos = np.array([prev_keypoints[j][0], prev_keypoints[j][1]])
                    curr_pos = np.array([curr_keypoints[j][0], curr_keypoints[j][1]])
                    
                    # Calculate displacement
                    displacement = np.linalg.norm(curr_pos - prev_pos)
                    frame_energy += displacement
                    valid_pairs += 1
            
            # Average energy for this frame
            if valid_pairs > 0:
                total_energy += frame_energy / valid_pairs
        
        # Average energy across all frames
        return float(total_energy / (len(keypoints_history) - 1))
    
    def _calculate_stability(self, keypoints_history):
        """
        Calculate stability score based on keypoint movement consistency
        
        Args:
            keypoints_history: List of keypoints history
            
        Returns:
            float: Stability score (0-1, higher is more stable)
        """
        if len(keypoints_history) < 3:
            return 1.0  # Default stability with insufficient data
        
        # Calculate direction changes in movement
        direction_changes = 0
        total_directions = 0
        
        for i in range(2, len(keypoints_history)):
            for j in range(len(keypoints_history[i])):
                # Skip keypoints with low confidence
                if (keypoints_history[i][j][2] > 0.5 and 
                    keypoints_history[i-1][j][2] > 0.5 and
                    keypoints_history[i-2][j][2] > 0.5):
                    
                    # Get positions
                    pos_prev2 = np.array([keypoints_history[i-2][j][0], keypoints_history[i-2][j][1]])
                    pos_prev1 = np.array([keypoints_history[i-1][j][0], keypoints_history[i-1][j][1]])
                    pos_curr = np.array([keypoints_history[i][j][0], keypoints_history[i][j][1]])
                    
                    # Calculate vectors
                    vec1 = pos_prev1 - pos_prev2
                    vec2 = pos_curr - pos_prev1
                    
                    # Skip if almost no movement
                    if np.linalg.norm(vec1) < 1.0 or np.linalg.norm(vec2) < 1.0:
                        continue
                    
                    # Normalize vectors
                    vec1 = vec1 / np.linalg.norm(vec1)
                    vec2 = vec2 / np.linalg.norm(vec2)
                    
                    # Calculate dot product to determine direction change
                    dot_product = np.clip(np.dot(vec1, vec2), -1.0, 1.0)
                    
                    # If dot product is negative, there was a significant direction change
                    if dot_product < 0:
                        direction_changes += 1
                    
                    total_directions += 1
        
        # Calculate stability score (1 - ratio of direction changes)
        if total_directions > 0:
            stability = 1.0 - (direction_changes / total_directions)
            return float(stability)
        else:
            return 1.0  # Default stability 
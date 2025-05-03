#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Track:
    """Represents a single tracked object with its state"""
    
    def __init__(self, track_id, bbox):
        self.id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        
        # Initialize Kalman filter for this track
        self.kf = self._init_kalman_filter(bbox)
        
        # Track history for trajectory and velocity calculation
        self.history = [self._bbox_center(bbox)]
        self.ages = 1
        self.time_since_update = 0
        self.hits = 1
        
    def _init_kalman_filter(self, bbox):
        """Initialize Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],  # x center
            [0, 1, 0, 0, 0, 1, 0],  # y center
            [0, 0, 1, 0, 0, 0, 1],  # width
            [0, 0, 0, 1, 0, 0, 0],  # height
            [0, 0, 0, 0, 1, 0, 0],  # x velocity
            [0, 0, 0, 0, 0, 1, 0],  # y velocity
            [0, 0, 0, 0, 0, 0, 1]   # scale velocity
        ])
        
        # Measurement matrix (we only observe position and size)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Initial state
        cx, cy, w, h = self._bbox_to_z(bbox)
        kf.x = np.array([[cx], [cy], [w], [h], [0], [0], [0]])
        
        # Covariance matrix
        kf.P *= 10
        kf.P[4:, 4:] *= 1000  # Give high uncertainty to the velocities
        
        # Process uncertainty
        kf.Q[4:, 4:] *= 0.01
        
        # Measurement uncertainty
        kf.R[2:, 2:] *= 10
        
        return kf
    
    def _bbox_to_z(self, bbox):
        """Convert bounding box to KF state vector [x,y,w,h]"""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h
    
    def _z_to_bbox(self, z):
        """Convert KF state vector [x,y,w,h] to bounding box [x1,y1,x2,y2]"""
        cx, cy, w, h = z
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [int(x1), int(y1), int(x2), int(y2)]
    
    def _bbox_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def predict(self):
        """Predict next state with Kalman filter"""
        self.kf.predict()
        self.time_since_update += 1
        # Get predicted bounding box
        cx, cy, w, h = self.kf.x[:4].flatten()
        self.bbox = self._z_to_bbox([cx, cy, w, h])
        return self.bbox
    
    def update(self, bbox):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        
        # Update Kalman filter
        cx, cy, w, h = self._bbox_to_z(bbox)
        self.kf.update(np.array([cx, cy, w, h]))
        
        # Update bounding box
        self.bbox = bbox
        
        # Update history
        self.history.append(self._bbox_center(bbox))
        
    def get_state(self):
        """Get current state"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'velocity': [float(self.kf.x[4]), float(self.kf.x[5])],
            'history': self.history,
            'time_since_update': self.time_since_update,
            'age': self.ages
        }


class HumanTracker:
    """Multi-object tracker specialized for humans"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the tracker
        
        Args:
            max_age (int): Maximum number of frames to keep a track alive without matching
            min_hits (int): Minimum number of hits to start a track
            iou_threshold (float): IoU threshold for detection-track association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = []
        self.next_id = 0
    
    def _iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _associate_detections_to_tracks(self, detections, tracks):
        """
        Associate detections with existing tracks using IoU
        Returns:
            matches, unmatched_detections, unmatched_tracks
        """
        if not tracks:
            return [], list(range(len(detections))), []
        
        if not detections:
            return [], [], list(range(len(tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d, detection in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self._iou(detection[:4], track.bbox)
        
        # Use Hungarian algorithm for optimal assignment
        # The scipy function solves a minimization problem
        # So we negate the IoU matrix to get a cost matrix
        detection_indices, track_indices = linear_sum_assignment(-iou_matrix)
        
        # Filter out low IoU matches
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        for d_idx, t_idx in zip(detection_indices, track_indices):
            if iou_matrix[d_idx, t_idx] >= self.iou_threshold:
                matches.append((d_idx, t_idx))
                unmatched_detections.remove(d_idx)
                unmatched_tracks.remove(t_idx)
        
        return matches, unmatched_detections, unmatched_tracks
    
    def update(self, detections, frame=None):
        """
        Update tracker with new detections
        
        Args:
            detections (list): List of detections, each as [x1, y1, x2, y2, confidence]
            frame (numpy.ndarray, optional): Current frame (unused, for future extensions)
            
        Returns:
            list: List of active tracks as dictionaries with state information
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
            track.ages += 1
        
        # Associate detections with tracks
        matches, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for d_idx, t_idx in matches:
            self.tracks[t_idx].update(detections[d_idx][:4])
        
        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            new_track = Track(self.next_id, detections[d_idx][:4])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Filter tracks
        active_tracks = []
        remaining_tracks = []
        
        for track in self.tracks:
            if track.time_since_update > self.max_age:
                # Remove track
                continue
            
            # Only return confirmed tracks (those with enough hits)
            if track.hits >= self.min_hits:
                active_tracks.append(track.get_state())
            
            remaining_tracks.append(track)
        
        self.tracks = remaining_tracks
        
        return active_tracks 
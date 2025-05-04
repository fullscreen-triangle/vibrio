#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
import cv2
from collections import deque

class VideoFeatureExtractor:
    """Video feature extraction using Video Swin Transformer"""
    
    def __init__(self, model_name="Tonic/video-swin-transformer", device=None, buffer_size=16):
        """
        Initialize the video feature extractor
        
        Args:
            model_name (str): Name of the video model to use
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
            buffer_size (int): Number of frames to buffer for video analysis
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.buffer_size = buffer_size
        
        # Initialize model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=buffer_size)
        
    def add_frame(self, frame):
        """
        Add a frame to the buffer
        
        Args:
            frame (numpy.ndarray): BGR image
        """
        # Convert to RGB and resize if needed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to expected input size if needed
        if hasattr(self.processor.config, 'size'):
            expected_size = self.processor.config.size.get('height', 224)
            frame_rgb = cv2.resize(frame_rgb, (expected_size, expected_size))
            
        # Add to buffer
        self.frame_buffer.append(frame_rgb)
        
    def extract_features(self):
        """
        Extract video features from the buffered frames
        
        Returns:
            dict: Dict containing:
                'features': Video feature vector
                'phase_segmentation': Phase segmentation data if available
                'temporal_info': Additional temporal information
        """
        # If buffer is not full yet, return empty features
        if len(self.frame_buffer) < self.buffer_size:
            return {
                'features': None,
                'phase_segmentation': None,
                'temporal_info': {
                    'buffer_fullness': len(self.frame_buffer) / self.buffer_size
                }
            }
        
        # Convert buffer to numpy array
        frames = np.array(list(self.frame_buffer))
        
        # Prepare inputs for the model
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract features
        # For Swin Transformer, we'll use the pooled output
        features = outputs.pooler_output.cpu().numpy()
        
        # For phase segmentation, we would use the last hidden states
        # and apply additional processing
        last_hidden_states = outputs.last_hidden_state.cpu().numpy()
        phase_segmentation = self._segment_phases(last_hidden_states)
        
        # Prepare output
        output = {
            'features': features[0].tolist(),  # Convert to list for serialization
            'phase_segmentation': phase_segmentation,
            'temporal_info': {
                'buffer_fullness': 1.0,
                'sequence_length': self.buffer_size
            }
        }
        
        return output
    
    def _segment_phases(self, hidden_states):
        """
        Segment the video into phases or actions
        
        Args:
            hidden_states (numpy.ndarray): Last hidden states from the model
            
        Returns:
            list: List of phase segments with their timestamps and confidence
        """
        # This is a simplified implementation
        # In practice, you would use a more sophisticated approach to segment the video
        
        # For example:
        # 1. Apply clustering to find distinct phases
        # 2. Use a classifier head trained on the hidden states
        # 3. Use change point detection algorithms
        
        # For this example, we'll just use a simple change detection 
        # based on feature similarity
        
        # Flatten the features along sequence dimension
        features = hidden_states[0]  # First batch
        
        # Compute similarities between adjacent frames
        similarities = []
        for i in range(1, len(features)):
            sim = np.dot(features[i], features[i-1]) / (np.linalg.norm(features[i]) * np.linalg.norm(features[i-1]))
            similarities.append(float(sim))
        
        # Find change points (where similarity drops)
        segments = []
        prev_idx = 0
        for i, sim in enumerate(similarities):
            # If similarity is below threshold, consider it a phase change
            if sim < 0.8:  # This threshold would be tuned for the application
                segments.append({
                    'start_frame': prev_idx,
                    'end_frame': i + 1,  # +1 due to 0-indexing
                    'confidence': 1.0 - sim,  # Higher confidence for larger changes
                    'duration': i + 1 - prev_idx
                })
                prev_idx = i + 1
        
        # Add the last segment if needed
        if prev_idx < len(features):
            segments.append({
                'start_frame': prev_idx,
                'end_frame': len(features),
                'confidence': 0.9,  # Placeholder
                'duration': len(features) - prev_idx
            })
        
        return segments
    
    def clear_buffer(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()
        
    def get_buffer_fullness(self):
        """Get the current fullness of the buffer"""
        return len(self.frame_buffer) / self.buffer_size 
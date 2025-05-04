#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionRecognizer:
    """Action recognition using skeleton and RGB features"""
    
    def __init__(self, skeleton_head=True, rgb_head=True, device=None):
        """
        Initialize the action recognizer
        
        Args:
            skeleton_head (bool): Whether to use skeleton features for action recognition
            rgb_head (bool): Whether to use RGB features for action recognition
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_skeleton = skeleton_head
        self.use_rgb = rgb_head
        
        # Action classes for different activity domains
        self.action_classes = {
            'general': [
                'walking', 'running', 'jumping', 'sitting', 'standing', 'turning',
                'crouching', 'climbing', 'falling', 'waving', 'clapping'
            ],
            'sports': [
                'sprinting', 'jogging', 'high_jump', 'long_jump', 'pole_vault',
                'shot_put', 'javelin_throw', 'discus_throw', 'swimming',
                'cycling', 'gymnastics', 'diving', 'surfing', 'skateboarding'
            ],
            'martial_arts': [
                'punch', 'kick', 'block', 'dodge', 'grapple', 'throw',
                'sweep', 'stance', 'kata_form'
            ]
        }
        
        # Initialize skeleton-based classifier
        if skeleton_head:
            self.skeleton_classifier = SkeletonActionClassifier(
                input_dim=512,  # MotionBERT typically outputs 512-dim embeddings
                num_classes=len(self.action_classes['general']),
                hidden_dim=256
            ).to(self.device)
            
        # Initialize RGB-based classifier
        if rgb_head:
            self.rgb_classifier = RGBActionClassifier(
                input_dim=768,  # Video Swin typically outputs 768-dim features
                num_classes=len(self.action_classes['general']) + len(self.action_classes['sports']),
                hidden_dim=512
            ).to(self.device)
            
        # Fusion layer for combining skeleton and RGB predictions
        if skeleton_head and rgb_head:
            self.fusion_layer = nn.Linear(
                len(self.action_classes['general']) * 2,  # Outputs from both classifiers
                len(self.action_classes['general'])
            ).to(self.device)
        
    def classify_action(self, skeleton_embedding=None, rgb_features=None, domain='general'):
        """
        Classify action based on skeleton and/or RGB features
        
        Args:
            skeleton_embedding (list/np.ndarray): Embedding from MotionBERT
            rgb_features (list/np.ndarray): Features from Video Swin Transformer
            domain (str): Activity domain for classification ('general', 'sports', 'martial_arts')
            
        Returns:
            dict: Dict containing:
                'action': Predicted action class
                'confidence': Confidence score
                'all_scores': Scores for all classes
        """
        # Make sure we have the inputs we need based on configuration
        if self.use_skeleton and skeleton_embedding is None:
            raise ValueError("Skeleton embedding required but not provided")
        if self.use_rgb and rgb_features is None:
            raise ValueError("RGB features required but not provided")
            
        # Convert inputs to tensors
        skeleton_tensor = None
        rgb_tensor = None
        
        if skeleton_embedding is not None:
            skeleton_tensor = torch.tensor(skeleton_embedding, dtype=torch.float32).unsqueeze(0).to(self.device)
            
        if rgb_features is not None:
            rgb_tensor = torch.tensor(rgb_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get predictions from each head
        skeleton_preds = None
        rgb_preds = None
        
        if self.use_skeleton and skeleton_tensor is not None:
            with torch.no_grad():
                skeleton_preds = self.skeleton_classifier(skeleton_tensor)
                
        if self.use_rgb and rgb_tensor is not None:
            with torch.no_grad():
                rgb_preds = self.rgb_classifier(rgb_tensor)
        
        # Fuse predictions if both available
        if skeleton_preds is not None and rgb_preds is not None:
            # Only use the common action classes (general)
            with torch.no_grad():
                rgb_general_preds = rgb_preds[:, :len(self.action_classes['general'])]
                
                # Concatenate predictions
                fused_input = torch.cat([skeleton_preds, rgb_general_preds], dim=1)
                fused_preds = self.fusion_layer(fused_input)
                
                # Apply softmax
                scores = F.softmax(fused_preds, dim=1).cpu().numpy()[0]
        elif skeleton_preds is not None:
            scores = F.softmax(skeleton_preds, dim=1).cpu().numpy()[0]
        elif rgb_preds is not None:
            # Filter for requested domain
            if domain == 'general':
                domain_preds = rgb_preds[:, :len(self.action_classes['general'])]
            elif domain == 'sports':
                start_idx = len(self.action_classes['general'])
                end_idx = start_idx + len(self.action_classes['sports'])
                domain_preds = rgb_preds[:, start_idx:end_idx]
            else:
                # Default to general
                domain_preds = rgb_preds[:, :len(self.action_classes['general'])]
                
            scores = F.softmax(domain_preds, dim=1).cpu().numpy()[0]
        else:
            # No predictions available
            return {
                'action': None,
                'confidence': 0.0,
                'all_scores': []
            }
        
        # Get the top prediction
        top_idx = np.argmax(scores)
        
        # Get the class name based on domain
        if domain == 'general':
            classes = self.action_classes['general']
        elif domain == 'sports':
            classes = self.action_classes['sports']
        elif domain == 'martial_arts':
            classes = self.action_classes['martial_arts']
        else:
            classes = self.action_classes['general']
            
        # Prepare output
        output = {
            'action': classes[top_idx],
            'confidence': float(scores[top_idx]),
            'all_scores': [{'class': cls, 'score': float(score)} for cls, score in zip(classes, scores)]
        }
        
        return output


class SkeletonActionClassifier(nn.Module):
    """Neural network for classifying actions from skeleton embeddings"""
    
    def __init__(self, input_dim, num_classes, hidden_dim):
        super(SkeletonActionClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class RGBActionClassifier(nn.Module):
    """Neural network for classifying actions from RGB features"""
    
    def __init__(self, input_dim, num_classes, hidden_dim):
        super(RGBActionClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x 
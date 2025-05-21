#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from ultralytics import YOLO

class HumanDetector:
    """Human detection using YOLOv8 model"""
    
    def __init__(self, model_path=None, conf_threshold=0.5, device=None):
        """
        Initialize the human detector
        
        Args:
            model_path (str, optional): Path to custom YOLOv8 model. 
                If None, uses the pre-trained model.
            conf_threshold (float): Confidence threshold for detections
            device (str, optional): Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.conf_threshold = conf_threshold
        
        # Load YOLOv8 model
        if model_path:
            self.model = YOLO(model_path)
        else:
            self.model = YOLO("yolov8n.pt")  # Load the smaller model by default
            
        if device:
            self.model.to(device)
            
        # Person class ID in COCO dataset (which YOLOv8 uses) is 0
        self.person_class_id = 0
    
    def detect(self, frame):
        """
        Detect humans in a frame
        
        Args:
            frame (numpy.ndarray): BGR image
            
        Returns:
            list: List of human detections, each as a dict with keys:
                 - 'bbox': [x1, y1, x2, y2] (the bounding box coordinates)
                 - 'confidence': float (detection confidence score)
        """
        # Run YOLOv8 inference
        results = self.model(frame, verbose=False)[0]
        
        # Filter for humans (person class) and confidence threshold
        detections = []
        
        for box in results.boxes:
            cls = int(box.cls.item())
            conf = box.conf.item()
            
            # Only keep person class with confidence above threshold
            if cls == self.person_class_id and conf >= self.conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
                
        return detections 
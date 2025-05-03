#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import logging
from pathlib import Path

def setup_logger(debug=False):
    """
    Set up and configure logger
    
    Args:
        debug (bool): Enable debug logging if True
        
    Returns:
        logging.Logger: Configured logger
    """
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure root logger
    logger = logging.getLogger('vibrio')
    logger.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def load_calibration(calibration_path):
    """
    Load camera calibration from file
    
    Args:
        calibration_path (str): Path to calibration file
        
    Returns:
        dict: Calibration parameters or None if file not found
    """
    if not calibration_path or not os.path.exists(calibration_path):
        logging.warning(f"Calibration file not found: {calibration_path}")
        return None
    
    try:
        with open(calibration_path, 'r') as f:
            calibration = json.load(f)
        
        logging.info(f"Loaded calibration from {calibration_path}")
        return calibration
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Failed to load calibration: {e}")
        return None

def create_calibration_template(output_path='calibration_template.json'):
    """
    Create a template calibration file with example values
    
    Args:
        output_path (str): Path to save template
        
    Returns:
        bool: True if template was created, False otherwise
    """
    template = {
        "camera_matrix": [
            [1000.0, 0.0, 640.0],
            [0.0, 1000.0, 360.0],
            [0.0, 0.0, 1.0]
        ],
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "resolution": [1280, 720],
        "base_ratio": 0.1,  # pixels to meters at reference distance
        "reference_distance": 10.0,  # meters
        "reference_point": [640, 360],  # center of image
        "scale_factors": {
            "top": 0.5,      # scaling factor at the top of the image
            "bottom": 1.0,   # scaling factor at the bottom of the image
            "left": 0.8,     # scaling factor at the left of the image
            "right": 0.8     # scaling factor at the right of the image
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(template, f, indent=4)
        
        logging.info(f"Created calibration template at {output_path}")
        return True
    except IOError as e:
        logging.error(f"Failed to create calibration template: {e}")
        return False

def seconds_to_timestamp(seconds):
    """
    Convert seconds to HH:MM:SS timestamp
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def estimate_real_size(height_pixels, bbox_position, calibration=None):
    """
    Estimate real-world size of a human based on bounding box height
    
    Args:
        height_pixels (float): Height in pixels
        bbox_position (tuple): (x, y) position of bounding box (bottom center)
        calibration (dict, optional): Camera calibration parameters
        
    Returns:
        float: Estimated height in meters
    """
    # Default average human height
    DEFAULT_HUMAN_HEIGHT = 1.7  # meters
    
    if calibration is None:
        # Simple estimation based on average human height
        return DEFAULT_HUMAN_HEIGHT
    
    # Get scaling factor based on position in image
    # This is a simple implementation that can be expanded with proper calibration
    if 'scale_factors' in calibration:
        scale_factors = calibration['scale_factors']
        
        # Get image dimensions
        width, height = calibration.get('resolution', (1280, 720))
        
        # Normalize positions
        x_rel = bbox_position[0] / width
        y_rel = bbox_position[1] / height
        
        # Interpolate scaling factors based on position
        h_scale = scale_factors['left'] * (1 - x_rel) + scale_factors['right'] * x_rel
        v_scale = scale_factors['top'] * (1 - y_rel) + scale_factors['bottom'] * y_rel
        
        # Combined scale
        scale = (h_scale + v_scale) / 2
        
        # Apply scale to default height
        return DEFAULT_HUMAN_HEIGHT * scale
    
    return DEFAULT_HUMAN_HEIGHT 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

class PhysicsVerifier:
    """
    Verifies if estimated speeds are physically plausible
    based on the context and physics constraints
    """
    
    # Speed limits in km/h for different contexts
    SPEED_LIMITS = {
        'general': {
            'walking': 7,       # Normal walking
            'running': 32,      # Elite sprinters can reach ~45 km/h
            'cycling': 50,      # Recreational cycling
            'max': 150          # General upper limit (e.g., motorcycle)
        },
        'athletics': {
            'walking': 15,      # Race walking
            'running': 45,      # World-class sprinting
            'max': 55
        },
        'cycling': {
            'casual': 25,
            'racing': 80,       # Pro cycling on flat terrain
            'downhill': 120,    # Extreme downhill cycling
            'max': 130
        },
        'racing': {
            'min': 30,          # Slow for racing
            'typical': 200,     # Typical racing speeds
            'max': 350          # Formula 1 cars can reach 350+ km/h
        },
        'sport': {
            'min': 5,
            'typical': 30,
            'max': 100          # Various sports (higher for motorsports)
        }
    }
    
    # Acceleration limits in m/s² for different contexts
    ACCELERATION_LIMITS = {
        'general': {
            'walking': 2,
            'running': 4,
            'max': 10           # Approximately 1G
        },
        'athletics': {
            'start': 8,         # Elite sprinters from blocks
            'running': 4,
            'max': 10
        },
        'cycling': {
            'typical': 3,
            'max': 6
        },
        'racing': {
            'typical': 15,      # Race cars can achieve high acceleration
            'max': 30           # F1 cars can achieve ~5G (50 m/s²)
        },
        'sport': {
            'typical': 5,
            'max': 15
        }
    }
    
    def __init__(self, context='general', confidence_threshold=0.8):
        """
        Initialize the physics verifier
        
        Args:
            context (str): The context for speed verification ('general', 'sport', 
                          'cycling', 'racing', 'athletics')
            confidence_threshold (float): Threshold for confidence in verification
        """
        self.context = context
        self.confidence_threshold = confidence_threshold
        
        # Store previous speeds for acceleration calculation
        self.previous_speeds = {}
        self.previous_timestamps = {}
        
        # Get speed limits for the specified context
        if context in self.SPEED_LIMITS:
            self.speed_limits = self.SPEED_LIMITS[context]
        else:
            self.speed_limits = self.SPEED_LIMITS['general']
            
        # Get acceleration limits
        if context in self.ACCELERATION_LIMITS:
            self.acceleration_limits = self.ACCELERATION_LIMITS[context]
        else:
            self.acceleration_limits = self.ACCELERATION_LIMITS['general']
    
    def _calculate_acceleration(self, track_id, current_speed, fps):
        """
        Calculate acceleration between frames
        
        Args:
            track_id (int): Track identifier
            current_speed (float): Current speed in km/h
            fps (float): Frames per second
            
        Returns:
            float: Acceleration in m/s²
        """
        if track_id not in self.previous_speeds:
            self.previous_speeds[track_id] = current_speed
            self.previous_timestamps[track_id] = 0
            return 0.0
        
        # Convert km/h to m/s
        current_speed_ms = current_speed / 3.6
        previous_speed_ms = self.previous_speeds[track_id] / 3.6
        
        # Time elapsed in seconds
        time_elapsed = (1.0 / fps)
        
        # Calculate acceleration
        acceleration = (current_speed_ms - previous_speed_ms) / time_elapsed
        
        # Update previous values
        self.previous_speeds[track_id] = current_speed
        
        return acceleration
    
    def _verify_speed(self, speed, track, acceleration=None):
        """
        Verify if the speed is physically plausible based on context
        
        Args:
            speed (float): Speed in km/h
            track (dict): Track information
            acceleration (float, optional): Acceleration in m/s²
            
        Returns:
            tuple: (verified_speed, confidence)
        """
        # Get absolute max speed for context
        max_speed = self.speed_limits.get('max', 150)
        
        # Special case for racing context
        if self.context == 'racing' and speed > self.speed_limits['max']:
            # Racing has special verification based on acceleration profiles
            if acceleration and abs(acceleration) > self.acceleration_limits['max']:
                # Too rapid acceleration/deceleration
                confidence = 0.3
                verified_speed = self.speed_limits['typical']
            else:
                # Could be plausible for very fast vehicles
                confidence = 0.7
                verified_speed = speed
        
        # Verify based on typical ranges
        elif speed <= max_speed:
            if speed < self.speed_limits.get('min', 0):
                # Below minimum expected speed
                confidence = 0.7
                verified_speed = speed
            elif self.context == 'general' and speed > self.speed_limits['running']:
                # In general context, speeds above running are less likely to be humans
                if speed > self.speed_limits['cycling']:
                    confidence = 0.4
                else:
                    confidence = 0.6
                verified_speed = speed
            else:
                # Within expected range
                confidence = 0.9
                verified_speed = speed
        else:
            # Beyond maximum - likely an error
            confidence = 0.2
            verified_speed = max_speed
        
        # Verify with acceleration if available
        if acceleration is not None:
            accel_max = self.acceleration_limits.get('max', 10)
            
            if abs(acceleration) > accel_max:
                # Unrealistic acceleration
                confidence *= 0.5
                # Adjust speed based on max possible acceleration
                sign = 1 if acceleration > 0 else -1
                previous_speed_ms = self.previous_speeds[track['id']] / 3.6
                adjusted_speed_ms = previous_speed_ms + sign * accel_max / 30  # Assuming ~30fps
                verified_speed = min(adjusted_speed_ms * 3.6, verified_speed)
        
        return verified_speed, confidence
    
    def verify(self, speeds, tracks, fps=30):
        """
        Verify speeds of tracked humans
        
        Args:
            speeds (dict): Dictionary mapping track IDs to estimated speeds
            tracks (list): List of track states
            fps (float): Video framerate
            
        Returns:
            dict: Dictionary mapping track IDs to verified speeds
        """
        verified_speeds = {}
        
        # Create a lookup for tracks by ID
        track_lookup = {track['id']: track for track in tracks}
        
        for track_id, speed in speeds.items():
            if track_id not in track_lookup:
                continue
                
            track = track_lookup[track_id]
            
            # Calculate acceleration
            acceleration = self._calculate_acceleration(track_id, speed, fps)
            
            # Verify speed
            verified_speed, confidence = self._verify_speed(speed, track, acceleration)
            
            # Only use verified speed if confidence is high enough
            if confidence >= self.confidence_threshold:
                verified_speeds[track_id] = verified_speed
            else:
                # For low confidence, use previous valid speed or 0
                verified_speeds[track_id] = self.previous_speeds.get(track_id, 0)
        
        return verified_speeds 
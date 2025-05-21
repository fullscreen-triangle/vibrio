#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optical Analysis Module for Vibrio

This module implements advanced optical methods for motion analysis:
1. Optical Flow (Pixel Change Analysis)
2. Motion Energy and Temporal Difference Metrics
3. Event-based (Neuromorphic) Camera Simulation
4. Texture and Gradient-based Analysis
5. Shadow and Illumination Analysis
"""

import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fft, fftfreq
import pandas as pd
from .utils import create_dir_if_not_exists

class OpticalAnalyzer:
    """
    Implements advanced optical analysis methods for motion analysis.
    """
    
    def __init__(self, output_dir='results/optical_analysis', visualization_dir='visualizations'):
        """
        Initialize the optical analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
            visualization_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        self.visualization_dir = visualization_dir
        
        # Create output directories
        create_dir_if_not_exists(output_dir)
        create_dir_if_not_exists(visualization_dir)
        create_dir_if_not_exists(os.path.join(visualization_dir, 'optical_flow'))
        create_dir_if_not_exists(os.path.join(visualization_dir, 'motion_energy'))
        create_dir_if_not_exists(os.path.join(visualization_dir, 'neuromorphic'))
        create_dir_if_not_exists(os.path.join(visualization_dir, 'texture_analysis'))
        create_dir_if_not_exists(os.path.join(visualization_dir, 'shadow_analysis'))
        
    def analyze_video(self, video_path, methods=None, output_video=True, output_data=True):
        """
        Analyze a video using the specified optical methods.
        
        Args:
            video_path (str): Path to the video file
            methods (list): List of methods to use (default: all methods)
            output_video (bool): Whether to output annotated video
            output_data (bool): Whether to output numerical data
            
        Returns:
            dict: Analysis results
        """
        # Default: use all methods
        if methods is None:
            methods = ['optical_flow', 'motion_energy', 'neuromorphic', 
                      'texture_analysis', 'shadow_analysis']
        
        video_name = Path(video_path).stem
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writers if output_video is True
        video_writers = {}
        if output_video:
            for method in methods:
                output_path = os.path.join(self.visualization_dir, f"{method}/{video_name}_{method}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writers[method] = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize results
        results = {method: [] for method in methods}
        
        # Initialize variables for various methods
        prev_frame = None
        prev_gray = None
        mhi = np.zeros((height, width), dtype=np.float32)  # Motion History Image
        event_buffer = np.zeros((height, width), dtype=np.float32)  # For neuromorphic simulation
        
        # Process the video frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to grayscale for most processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Store first frame info
            if prev_frame is None:
                prev_frame = frame.copy()
                prev_gray = gray.copy()
                frame_count += 1
                continue
            
            # Process with each selected method
            for method in methods:
                if method == 'optical_flow':
                    flow_result, vis_frame = self._process_optical_flow(prev_gray, gray, frame)
                    results[method].append({
                        'frame': frame_count,
                        **flow_result
                    })
                    if output_video and method in video_writers:
                        video_writers[method].write(vis_frame)
                
                elif method == 'motion_energy':
                    energy_result, vis_frame = self._process_motion_energy(prev_gray, gray, frame, mhi)
                    results[method].append({
                        'frame': frame_count,
                        **energy_result
                    })
                    if output_video and method in video_writers:
                        video_writers[method].write(vis_frame)
                
                elif method == 'neuromorphic':
                    neuro_result, vis_frame = self._process_neuromorphic(prev_gray, gray, frame, event_buffer)
                    results[method].append({
                        'frame': frame_count,
                        **neuro_result
                    })
                    if output_video and method in video_writers:
                        video_writers[method].write(vis_frame)
                
                elif method == 'texture_analysis':
                    texture_result, vis_frame = self._process_texture_analysis(gray, frame)
                    results[method].append({
                        'frame': frame_count,
                        **texture_result
                    })
                    if output_video and method in video_writers:
                        video_writers[method].write(vis_frame)
                
                elif method == 'shadow_analysis':
                    shadow_result, vis_frame = self._process_shadow_analysis(prev_frame, frame)
                    results[method].append({
                        'frame': frame_count,
                        **shadow_result
                    })
                    if output_video and method in video_writers:
                        video_writers[method].write(vis_frame)
            
            # Update variables for next iteration
            prev_frame = frame.copy()
            prev_gray = gray.copy()
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count*100/total_frames:.1f}%)")
        
        # Release resources
        cap.release()
        for writer in video_writers.values():
            writer.release()
        
        # Save results to CSV files if output_data is True
        if output_data:
            for method in methods:
                if results[method]:
                    df = pd.DataFrame(results[method])
                    output_path = os.path.join(self.output_dir, f"{video_name}_{method}.csv")
                    df.to_csv(output_path, index=False)
                    print(f"Saved {method} results to {output_path}")
        
        return results
    
    def _process_optical_flow(self, prev_gray, gray, frame):
        """
        Process frames using optical flow methods.
        
        Args:
            prev_gray (np.ndarray): Previous frame (grayscale)
            gray (np.ndarray): Current frame (grayscale)
            frame (np.ndarray): Current frame (color, for visualization)
            
        Returns:
            tuple: (result_dict, visualization_frame)
        """
        # Calculate Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Calculate flow magnitude and direction
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate statistics
        mean_magnitude = np.mean(mag)
        max_magnitude = np.max(mag)
        std_magnitude = np.std(mag)
        
        # Calculate motion metrics
        motion_direction = np.mean(ang)
        motion_coherence = np.mean(mag) / (np.std(ang) + 1e-5)  # Higher value = more coherent motion
        
        # Create flow visualization
        vis_frame = frame.copy()
        hsv = np.zeros_like(frame)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = np.clip(mag * 15, 0, 255)
        hsv[..., 2] = 255
        vis_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Create visualization with both the original frame and the flow
        vis_frame = cv2.addWeighted(vis_frame, 0.7, vis_flow, 0.3, 0)
        
        # Add text with metrics
        text = f"Mean Flow: {mean_magnitude:.2f} | Coherence: {motion_coherence:.2f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw flow vectors (subsample for clarity)
        step = 16
        for y in range(0, flow.shape[0], step):
            for x in range(0, flow.shape[1], step):
                fx, fy = flow[y, x]
                # Only draw significant motion
                if np.sqrt(fx*fx + fy*fy) > 1.0:
                    cv2.line(vis_frame, (x, y), (int(x + fx), int(y + fy)), (0, 255, 0), 1)
                    cv2.circle(vis_frame, (x, y), 1, (0, 255, 0), -1)
        
        # Prepare results dictionary
        result = {
            'mean_flow_magnitude': mean_magnitude,
            'max_flow_magnitude': max_magnitude,
            'std_flow_magnitude': std_magnitude,
            'motion_direction': motion_direction,
            'motion_coherence': motion_coherence
        }
        
        return result, vis_frame
    
    def _process_motion_energy(self, prev_gray, gray, frame, mhi, tau=0.5, mhi_duration=15):
        """
        Process frames using motion energy and temporal difference metrics.
        
        Args:
            prev_gray (np.ndarray): Previous frame (grayscale)
            gray (np.ndarray): Current frame (grayscale)
            frame (np.ndarray): Current frame (color, for visualization)
            mhi (np.ndarray): Motion History Image
            tau (float): Threshold for motion detection
            mhi_duration (int): Duration for MHI in frames
            
        Returns:
            tuple: (result_dict, visualization_frame)
        """
        # Calculate frame difference
        frame_diff = cv2.absdiff(gray, prev_gray)
        
        # Apply threshold to get significant motion
        _, motion_mask = cv2.threshold(frame_diff, 25, 1, cv2.THRESH_BINARY)
        
        # Update MHI
        timestamp = 1.0  # Current timestamp
        
        # In newer OpenCV versions, motempl functions are moved to the main cv2 namespace
        # Try both the old and new ways to ensure compatibility
        try:
            # Modern OpenCV
            cv2.updateMotionHistory(motion_mask, mhi, timestamp, mhi_duration)
            
            # Calculate motion gradient
            mg_mask = np.zeros_like(gray, dtype=np.uint8)
            mg_orientation = np.zeros_like(gray, dtype=np.float32)
            cv2.calcMotionGradient(mhi, 0.25, 0.05, mg_mask, mg_orientation)
            
            # Calculate global orientation
            seg_mask = np.zeros_like(gray, dtype=np.uint8)
            orientation = cv2.calcGlobalOrientation(mg_orientation, mg_mask, mhi, timestamp, mhi_duration)
        except AttributeError:
            # Fallback for older OpenCV with motempl namespace
            try:
                cv2.motempl.updateMotionHistory(motion_mask, mhi, timestamp, mhi_duration)
                
                # Calculate motion gradient
                mg_mask = np.zeros_like(gray, dtype=np.uint8)
                mg_orientation = np.zeros_like(gray, dtype=np.float32)
                cv2.motempl.calcMotionGradient(mhi, 0.25, 0.05, mg_mask, mg_orientation)
                
                # Calculate global orientation
                seg_mask = np.zeros_like(gray, dtype=np.uint8)
                orientation = cv2.motempl.calcGlobalOrientation(mg_orientation, mg_mask, mhi, timestamp, mhi_duration)
            except AttributeError:
                # If both fail, we'll use a simple alternative method
                print("Warning: Motion template functions not available in this OpenCV version")
                print("Using simplified motion energy calculation")
                
                # Apply simple decay to MHI
                mhi = mhi * 0.95
                
                # Update MHI manually
                mask_255 = motion_mask * 255
                mhi = np.maximum(mhi, mask_255 * (timestamp - mhi_duration + 1))
                
                # Calculate a simple orientation estimate
                orientation = 0
                mg_mask = motion_mask
        
        # Calculate motion energy
        motion_energy = np.sum(frame_diff) / (gray.shape[0] * gray.shape[1])
        
        # Calculate active motion regions
        try:
            contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # Older OpenCV versions return 3 values
            _, contours, _ = cv2.findContours(motion_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        active_regions = len(contours)
        
        # Normalize MHI for visualization
        vis_mhi = np.clip(mhi / mhi_duration * 255, 0, 255).astype(np.uint8)
        vis_mhi_color = cv2.applyColorMap(vis_mhi, cv2.COLORMAP_JET)
        
        # Create visualization
        vis_frame = frame.copy()
        vis_frame = cv2.addWeighted(vis_frame, 0.7, vis_mhi_color, 0.3, 0)
        
        # Draw contours of motion regions
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)
        
        # Add text with metrics
        text = f"Motion Energy: {motion_energy:.2f} | Active Regions: {active_regions}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Prepare results dictionary
        result = {
            'motion_energy': motion_energy,
            'active_regions': active_regions,
            'global_orientation': orientation,
            'mhi_mean': np.mean(mhi),
            'mhi_max': np.max(mhi)
        }
        
        return result, vis_frame
    
    def _process_neuromorphic(self, prev_gray, gray, frame, event_buffer, threshold=15, decay=0.9):
        """
        Simulate event-based (neuromorphic) camera processing.
        
        Args:
            prev_gray (np.ndarray): Previous frame (grayscale)
            gray (np.ndarray): Current frame (grayscale)
            frame (np.ndarray): Current frame (color, for visualization)
            event_buffer (np.ndarray): Buffer for event accumulation
            threshold (int): Intensity change threshold for event generation
            decay (float): Decay factor for event buffer
            
        Returns:
            tuple: (result_dict, visualization_frame)
        """
        # Calculate intensity changes
        intensity_change = gray.astype(np.float32) - prev_gray.astype(np.float32)
        
        # Generate events based on threshold crossing
        pos_events = (intensity_change > threshold).astype(np.float32)
        neg_events = (intensity_change < -threshold).astype(np.float32)
        
        # Update event buffer with decay
        event_buffer = event_buffer * decay
        event_buffer += pos_events - neg_events
        
        # Calculate event statistics
        total_events = np.sum(pos_events) + np.sum(neg_events)
        event_density = total_events / (gray.shape[0] * gray.shape[1])
        pos_neg_ratio = np.sum(pos_events) / (np.sum(neg_events) + 1e-5)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Normalize event buffer for visualization
        norm_buffer = np.clip(event_buffer * 128 + 128, 0, 255).astype(np.uint8)
        event_vis = cv2.applyColorMap(norm_buffer, cv2.COLORMAP_JET)
        
        # Overlay event visualization
        vis_frame = cv2.addWeighted(vis_frame, 0.7, event_vis, 0.3, 0)
        
        # Add text with metrics
        text = f"Events: {int(total_events)} | Density: {event_density:.4f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate hotspot regions (areas with high event density)
        smoothed_events = gaussian_filter(np.abs(event_buffer), sigma=5)
        hotspots = (smoothed_events > np.max(smoothed_events) * 0.5).astype(np.uint8) * 255
        
        try:
            contours, _ = cv2.findContours(hotspots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # Older OpenCV versions return 3 values
            _, contours, _ = cv2.findContours(hotspots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw hotspot regions
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 255), 2)
        
        # Prepare results dictionary
        result = {
            'total_events': total_events,
            'event_density': event_density,
            'pos_neg_ratio': pos_neg_ratio,
            'event_buffer_mean': np.mean(np.abs(event_buffer)),
            'event_buffer_max': np.max(np.abs(event_buffer))
        }
        
        return result, vis_frame
    
    def _process_texture_analysis(self, gray, frame):
        """
        Analyze texture and gradients to detect muscle contractions and biomechanical insights.
        
        Args:
            gray (np.ndarray): Current frame (grayscale)
            frame (np.ndarray): Current frame (color, for visualization)
            
        Returns:
            tuple: (result_dict, visualization_frame)
        """
        # Calculate gradients using Sobel filters
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = cv2.magnitude(grad_x, grad_y)
        direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        
        # Apply Gabor filter bank for texture analysis
        # Detect different muscle fiber orientations
        gabor_responses = []
        gabor_energies = []
        
        # Different orientations and scales for Gabor filters
        orientations = [0, 45, 90, 135]
        for theta in orientations:
            # Create Gabor kernel
            theta_rad = theta * np.pi / 180.0
            gabor_kernel = cv2.getGaborKernel((15, 15), 4.0, theta_rad, 8.0, 0.5, 0, ktype=cv2.CV_32F)
            
            # Apply Gabor filter
            gabor_response = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel)
            gabor_energy = np.sum(np.abs(gabor_response))
            
            gabor_responses.append(gabor_response)
            gabor_energies.append(gabor_energy)
        
        # Calculate local binary patterns for texture
        def local_binary_pattern(img, radius=3):
            n_points = 8 * radius
            dx = [-1, 0, 1, 1, 1, 0, -1, -1]
            dy = [-1, -1, -1, 0, 1, 1, 1, 0]
            lbp = np.zeros_like(img)
            
            for i in range(radius, img.shape[0] - radius):
                for j in range(radius, img.shape[1] - radius):
                    center = img[i, j]
                    code = 0
                    for k in range(8):
                        code |= (img[i + dy[k], j + dx[k]] >= center) << k
                    lbp[i, j] = code
            
            return lbp
        
        lbp = local_binary_pattern(gray)
        lbp_contrast = np.std(lbp[lbp > 0])
        
        # Calculate texture metrics for biomechanical analysis
        texture_entropy = -np.sum(np.histogram(lbp, bins=256, range=(0,256))[0] * 
                                np.log(np.histogram(lbp, bins=256, range=(0,256))[0] + 1e-10))
        
        # Find regions with high gradient magnitude (potential muscle contractions)
        _, contraction_mask = cv2.threshold(magnitude, np.mean(magnitude) * 2, 255, cv2.THRESH_BINARY)
        contraction_mask = contraction_mask.astype(np.uint8)
        
        try:
            contours, _ = cv2.findContours(contraction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # Older OpenCV versions return 3 values
            _, contours, _ = cv2.findContours(contraction_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Normalize magnitude for visualization
        norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mag_color = cv2.applyColorMap(norm_magnitude, cv2.COLORMAP_HOT)
        
        # Create visualization with original frame and gradient overlay
        vis_frame = cv2.addWeighted(vis_frame, 0.7, mag_color, 0.3, 0)
        
        # Draw contours on visualization (potential muscle contractions)
        cv2.drawContours(vis_frame, contours, -1, (0, 255, 0), 2)
        
        # Add text with metrics
        texture_score = np.mean(gabor_energies) / (1.0 + lbp_contrast)  # Lower for smooth motion
        muscle_tension = np.sum(contraction_mask) / (gray.shape[0] * gray.shape[1])
        
        text = f"Texture: {texture_score:.2f} | Tension: {muscle_tension:.4f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate dominant orientation from Gabor filter responses
        dominant_orientation_idx = np.argmax(gabor_energies)
        dominant_orientation = orientations[dominant_orientation_idx]
        
        # Prepare results dictionary
        result = {
            'texture_score': texture_score,
            'muscle_tension': muscle_tension,
            'texture_entropy': texture_entropy,
            'dominant_orientation': dominant_orientation,
            'gradient_magnitude_mean': np.mean(magnitude),
            'contraction_areas': len(contours)
        }
        
        return result, vis_frame
    
    def _process_shadow_analysis(self, prev_frame, frame):
        """
        Analyze shadows and illumination changes to detect speed cues.
        
        Args:
            prev_frame (np.ndarray): Previous frame (color)
            frame (np.ndarray): Current frame (color)
            
        Returns:
            tuple: (result_dict, visualization_frame)
        """
        # Convert to HSV for better shadow detection
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
        curr_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract the V channel (illumination)
        prev_v = prev_hsv[:,:,2]
        curr_v = curr_hsv[:,:,2]
        
        # Calculate absolute difference in illumination
        v_diff = cv2.absdiff(curr_v, prev_v)
        
        # Calculate illumination ratio (for shadow detection)
        # Shadows typically have similar hue/saturation but lower value
        v_ratio = np.zeros_like(curr_v, dtype=np.float32)
        mask = prev_v > 0
        v_ratio[mask] = curr_v[mask].astype(np.float32) / prev_v[mask].astype(np.float32)
        
        # Detect shadow regions (value decreased but color similar)
        shadow_mask = np.zeros_like(curr_v, dtype=np.uint8)
        shadow_mask[(v_ratio < 0.8) & (v_ratio > 0.4)] = 255
        
        # Smooth shadow mask to reduce noise
        shadow_mask = cv2.medianBlur(shadow_mask, 5)
        
        # Compute motion mask (for moving shadows)
        _, motion_mask = cv2.threshold(v_diff, 20, 255, cv2.THRESH_BINARY)
        
        # Get moving shadow regions (intersection of shadow and motion)
        moving_shadow_mask = cv2.bitwise_and(shadow_mask, motion_mask)
        
        # Compute statistics for shadow analysis
        shadow_area = np.sum(shadow_mask > 0) / (shadow_mask.shape[0] * shadow_mask.shape[1])
        moving_shadow_area = np.sum(moving_shadow_mask > 0) / (moving_shadow_mask.shape[0] * moving_shadow_mask.shape[1])
        
        # Calculate direction of shadow movement
        # Find contours of moving shadows
        try:
            shadow_contours, _ = cv2.findContours(moving_shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            # Older OpenCV versions return 3 values
            _, shadow_contours, _ = cv2.findContours(moving_shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate average movement direction of shadows
        shadow_direction = 0.0
        if len(shadow_contours) > 0:
            # This would require tracking across frames for proper direction
            # Here we'll use a simple approximation based on contour orientation
            shadow_directions = []
            for contour in shadow_contours:
                if len(contour) >= 5:  # Need at least 5 points for ellipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        shadow_directions.append(ellipse[2])
                    except:
                        pass
            
            if shadow_directions:
                shadow_direction = np.mean(shadow_directions)
        
        # Create visualization
        vis_frame = frame.copy()
        
        # Create shadow visualization
        shadow_vis = np.zeros_like(frame)
        shadow_vis[shadow_mask > 0] = [120, 0, 255]  # Pink for shadows
        shadow_vis[moving_shadow_mask > 0] = [0, 255, 255]  # Yellow for moving shadows
        
        # Overlay shadow visualization
        vis_frame = cv2.addWeighted(vis_frame, 0.7, shadow_vis, 0.3, 0)
        
        # Draw shadow contours
        cv2.drawContours(vis_frame, shadow_contours, -1, (0, 255, 0), 2)
        
        # Add text with metrics
        text = f"Shadow Area: {shadow_area:.4f} | Moving Shadow: {moving_shadow_area:.4f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw shadow direction arrow (if available)
        if shadow_direction > 0 and len(shadow_contours) > 0:
            center_x, center_y = vis_frame.shape[1] // 2, vis_frame.shape[0] // 2
            angle_rad = shadow_direction * np.pi / 180.0
            end_x = int(center_x + 50 * np.cos(angle_rad))
            end_y = int(center_y + 50 * np.sin(angle_rad))
            cv2.arrowedLine(vis_frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
        
        # Prepare results dictionary
        result = {
            'shadow_area': shadow_area,
            'moving_shadow_area': moving_shadow_area,
            'shadow_direction': shadow_direction,
            'shadow_count': len(shadow_contours),
            'v_diff_mean': np.mean(v_diff)
        }
        
        return result, vis_frame
    
    def analyze_frequency_domain(self, video_path, roi=None, output_video=True):
        """
        Perform frequency domain analysis to identify repetitive movements.
        
        Args:
            video_path (str): Path to video file
            roi (tuple): Optional region of interest (x, y, w, h)
            output_video (bool): Whether to output visualization video
            
        Returns:
            dict: Analysis results
        """
        video_name = Path(video_path).stem
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output_video is True
        if output_video:
            output_path = os.path.join(self.visualization_dir, f"frequency_analysis/{video_name}_frequency.mp4")
            create_dir_if_not_exists(os.path.dirname(output_path))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If ROI is not specified, use the full frame
        if roi is None:
            roi = (0, 0, width, height)
        
        # Initialize signal buffer for frequency analysis
        frame_buffer = []
        signal = []
        max_buffer_size = int(fps * 5)  # 5 seconds of data
        
        # Process the video frame by frame
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract ROI
            x, y, w, h = roi
            roi_frame = frame[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate mean intensity as our signal
            mean_intensity = np.mean(gray_roi)
            
            # Add to signal buffer
            signal.append(mean_intensity)
            frame_buffer.append(frame.copy())
            
            # Keep buffer at fixed size
            if len(signal) > max_buffer_size:
                signal.pop(0)
                frame_buffer.pop(0)
            
            # Once we have enough data, perform frequency analysis
            if len(signal) >= max_buffer_size // 2:
                # Perform FFT
                signal_array = np.array(signal)
                signal_fft = fft(signal_array - np.mean(signal_array))
                freqs = fftfreq(len(signal_array), 1/fps)
                
                # Get positive frequencies only
                pos_mask = freqs > 0
                freqs = freqs[pos_mask]
                signal_fft = np.abs(signal_fft[pos_mask])
                
                # Find dominant frequency
                if len(freqs) > 0 and len(signal_fft) > 0:
                    dominant_idx = np.argmax(signal_fft)
                    dominant_freq = freqs[dominant_idx]
                    dominant_amp = signal_fft[dominant_idx]
                    
                    # Convert to Hz and cycles per second
                    dominant_freq_hz = dominant_freq
                    cycles_per_second = dominant_freq_hz
                    
                    # Create visualization
                    vis_frame = frame.copy()
                    
                    # Draw ROI
                    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Draw signal plot
                    plot_h = 150
                    plot_w = 300
                    plot_x = width - plot_w - 20
                    plot_y = 20
                    
                    # Create mini-plot of the signal
                    signal_plot = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 255
                    signal_normalized = (signal_array - np.min(signal_array)) / (np.max(signal_array) - np.min(signal_array))
                    
                    for i in range(len(signal_normalized) - 1):
                        pt1 = (int(i * plot_w / len(signal_normalized)), int(plot_h - signal_normalized[i] * plot_h))
                        pt2 = (int((i+1) * plot_w / len(signal_normalized)), int(plot_h - signal_normalized[i+1] * plot_h))
                        cv2.line(signal_plot, pt1, pt2, (0, 0, 255), 1)
                    
                    # Overlay plot on the frame
                    vis_frame[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w] = signal_plot
                    
                    # Draw frequency spectrum
                    spec_h = 150
                    spec_w = 300
                    spec_x = width - spec_w - 20
                    spec_y = plot_y + plot_h + 20
                    
                    # Create mini-plot of the frequency spectrum
                    spec_plot = np.ones((spec_h, spec_w, 3), dtype=np.uint8) * 255
                    if len(freqs) > 0 and len(signal_fft) > 0:
                        freqs_normalized = freqs / np.max(freqs)
                        fft_normalized = signal_fft / np.max(signal_fft)
                        
                        for i in range(min(len(freqs_normalized), len(fft_normalized)) - 1):
                            pt1 = (int(freqs_normalized[i] * spec_w), int(spec_h - fft_normalized[i] * spec_h))
                            pt2 = (int(freqs_normalized[i+1] * spec_w), int(spec_h - fft_normalized[i+1] * spec_h))
                            cv2.line(spec_plot, pt1, pt2, (255, 0, 0), 1)
                        
                        # Mark dominant frequency
                        if dominant_idx < len(freqs_normalized):
                            dom_x = int(freqs_normalized[dominant_idx] * spec_w)
                            dom_y = int(spec_h - fft_normalized[dominant_idx] * spec_h)
                            cv2.circle(spec_plot, (dom_x, dom_y), 5, (0, 255, 0), -1)
                    
                    # Overlay spectrum on the frame
                    vis_frame[spec_y:spec_y+spec_h, spec_x:spec_x+spec_w] = spec_plot
                    
                    # Add text with metrics
                    cv2.putText(vis_frame, f"Dominant Freq: {dominant_freq_hz:.2f} Hz", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(vis_frame, f"Cycles/Second: {cycles_per_second:.2f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Write frame if output_video
                    if output_video:
                        video_writer.write(vis_frame)
            
            frame_count += 1
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames} ({frame_count*100/total_frames:.1f}%)")
        
        # Release resources
        cap.release()
        if output_video:
            video_writer.release()
        
        # Calculate overall frequency domain results
        result = {
            'dominant_frequency': dominant_freq_hz if 'dominant_freq_hz' in locals() else 0,
            'cycles_per_second': cycles_per_second if 'cycles_per_second' in locals() else 0,
            'amplitude': dominant_amp if 'dominant_amp' in locals() else 0,
            'spectral_entropy': 0,  # Calculate spectral entropy for the full signal
            'analyzed_frames': frame_count
        }
        
        # Save results to CSV
        output_path = os.path.join(self.output_dir, f"{video_name}_frequency_analysis.csv")
        pd.DataFrame([result]).to_csv(output_path, index=False)
        
        return result

# Example usage function
def analyze_video_with_optical_methods(video_path, output_dir="results/optical_analysis", methods=None):
    """
    Analyze a video using all advanced optical methods.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save results
        methods (list): List of optical analysis methods to use
        
    Returns:
        dict: Results from analysis
    """
    if methods is None:
        methods = ['optical_flow', 'motion_energy', 'neuromorphic', 
                  'texture_analysis', 'shadow_analysis']
    
    analyzer = OpticalAnalyzer(output_dir=output_dir)
    results = analyzer.analyze_video(video_path, methods=methods)
    
    # Additionally perform frequency analysis
    freq_results = analyzer.analyze_frequency_domain(video_path)
    
    print(f"Completed optical analysis on {video_path}")
    print(f"Results saved to {output_dir}")
    
    return results 
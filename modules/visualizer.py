#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class Visualizer:
    """Visualizes tracking results and speed data"""
    
    # Color palette for tracks - visually distinct colors
    COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Green (dark)
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (255, 165, 0),  # Orange
        (255, 192, 203) # Pink
    ]
    
    # COCO keypoint connections for skeleton visualization
    SKELETON_CONNECTIONS = [
        [15, 13], [13, 11], [16, 14], [14, 12], # arms
        [11, 12], # shoulders
        [5, 11], [6, 12], # face to shoulders
        [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], # face
        [0, 1], [0, 2], [1, 3], [2, 4], # small face details (eyes, ears)
        [11, 5], [12, 6], # shoulders to hips
        [5, 6], # hips
        [5, 7], [6, 8], # hips to knees
        [7, 9], [8, 10] # knees to ankles
    ]
    
    def __init__(self, output_dir='results', show=False, save_video=True, save_plots=True, draw_skeleton=False):
        """
        Initialize the visualizer
        
        Args:
            output_dir (str): Directory to save results
            show (bool): Whether to show visualizations in real-time
            save_video (bool): Whether to save output video
            save_plots (bool): Whether to save speed plots
            draw_skeleton (bool): Whether to draw pose skeleton
        """
        self.output_dir = output_dir
        self.show = show
        self.save_video = save_video
        self.save_plots = save_plots
        self.draw_skeleton = draw_skeleton
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Video writer
        self.video_writer = None
        
        # Store speeds for plotting
        self.speed_history = {}  # {track_id: [(frame_idx, speed), ...]}
        
        # Store posture metrics for plotting
        self.posture_history = {}  # {track_id: [{metrics}, ...]}
        
        # Plot figures for visualization
        self.speed_fig = plt.figure(figsize=(10, 6))
        self.speed_ax = self.speed_fig.add_subplot(111)
        self.speed_ax.set_title('Human Speed Tracking')
        self.speed_ax.set_xlabel('Frame')
        self.speed_ax.set_ylabel('Speed (km/h)')
        
        self.posture_fig = plt.figure(figsize=(12, 8))
        self.posture_ax1 = self.posture_fig.add_subplot(311)
        self.posture_ax2 = self.posture_fig.add_subplot(312)
        self.posture_ax3 = self.posture_fig.add_subplot(313)
        self.posture_ax1.set_title('Postural Sway')
        self.posture_ax2.set_title('Locomotion Energy')
        self.posture_ax3.set_title('Stability Score')
    
    def _get_color(self, track_id):
        """Get color for a track"""
        return self.COLORS[track_id % len(self.COLORS)]
    
    def _draw_tracks(self, frame, tracks, speeds, keypoints=None, posture_metrics=None):
        """
        Draw tracks and speeds on frame
        
        Args:
            frame (numpy.ndarray): BGR image
            tracks (list): List of track states
            speeds (dict): Dictionary mapping track IDs to speeds
            keypoints (dict, optional): Dictionary mapping track IDs to keypoints
            posture_metrics (dict, optional): Dictionary mapping track IDs to posture metrics
            
        Returns:
            numpy.ndarray: Frame with visualizations
        """
        vis_frame = frame.copy()
        
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            
            # Get color for this track
            color = self._get_color(track_id)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw track ID
            cv2.putText(vis_frame, f"ID: {track_id}", (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw speed if available
            if track_id in speeds:
                speed = speeds[track_id]
                cv2.putText(vis_frame, f"{speed:.1f} km/h", (bbox[0], bbox[1] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw posture metrics if available
            y_offset = -50
            if track_id in posture_metrics:
                metrics = posture_metrics[track_id]
                
                if 'postural_sway' in metrics:
                    cv2.putText(vis_frame, f"Sway: {metrics['postural_sway']:.1f}", 
                                (bbox[0], bbox[1] + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset -= 15
                    
                if 'locomotion_energy' in metrics:
                    cv2.putText(vis_frame, f"Energy: {metrics['locomotion_energy']:.1f}", 
                                (bbox[0], bbox[1] + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    y_offset -= 15
                    
                if 'stability_score' in metrics:
                    cv2.putText(vis_frame, f"Stability: {metrics['stability_score']:.2f}", 
                                (bbox[0], bbox[1] + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw skeleton if keypoints are available and drawing is enabled
            if self.draw_skeleton and keypoints and track_id in keypoints:
                self._draw_skeleton(vis_frame, keypoints[track_id], color)
            
            # Draw trajectory
            history = track['history']
            for i in range(1, len(history)):
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                cv2.line(vis_frame, pt1, pt2, color, 2)
        
        return vis_frame
    
    def _draw_skeleton(self, frame, keypoints, color):
        """
        Draw skeleton on frame from keypoints
        
        Args:
            frame (numpy.ndarray): BGR image
            keypoints (list): List of keypoints as [x, y, confidence]
            color (tuple): BGR color
        """
        # Draw keypoints
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.5:  # Only draw high-confidence keypoints
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        
        # Draw skeleton connections
        for connection in self.SKELETON_CONNECTIONS:
            idx1, idx2 = connection
            
            # Check if both keypoints exist and have sufficient confidence
            if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5):
                
                pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
                pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
                
                cv2.line(frame, pt1, pt2, color, 2)
    
    def _update_speed_history(self, speeds, frame_idx):
        """Update speed history for plotting"""
        for track_id, speed in speeds.items():
            if track_id not in self.speed_history:
                self.speed_history[track_id] = []
            
            self.speed_history[track_id].append((frame_idx, speed))
    
    def _update_posture_history(self, posture_metrics, frame_idx):
        """Update posture history for plotting"""
        for track_id, metrics in posture_metrics.items():
            if track_id not in self.posture_history:
                self.posture_history[track_id] = []
            
            # Add frame index to metrics for plotting
            metrics_with_frame = metrics.copy()
            metrics_with_frame['frame'] = frame_idx
            
            self.posture_history[track_id].append(metrics_with_frame)
    
    def _draw_speed_plot(self):
        """Draw speed plot and convert to image"""
        self.speed_ax.clear()
        self.speed_ax.set_title('Human Speed Tracking')
        self.speed_ax.set_xlabel('Frame')
        self.speed_ax.set_ylabel('Speed (km/h)')
        
        for track_id, history in self.speed_history.items():
            if len(history) > 1:
                frames, speeds = zip(*history)
                color = [c/255 for c in self._get_color(track_id)]  # Convert to matplotlib RGB
                self.speed_ax.plot(frames, speeds, '-', color=color, label=f"ID {track_id}")
        
        self.speed_ax.legend(loc='upper right')
        self.speed_ax.grid(True)
        
        # Convert plot to image
        canvas = FigureCanvasAgg(self.speed_fig)
        canvas.draw()
        plot_img = np.array(canvas.renderer.buffer_rgba())
        
        # Convert from RGBA to BGR
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        return plot_img
    
    def _draw_posture_plots(self):
        """Draw posture metric plots and convert to image"""
        # Clear axes
        self.posture_ax1.clear()
        self.posture_ax2.clear()
        self.posture_ax3.clear()
        
        # Set titles
        self.posture_ax1.set_title('Postural Sway')
        self.posture_ax2.set_title('Locomotion Energy')
        self.posture_ax3.set_title('Stability Score')
        
        # Set labels
        self.posture_ax1.set_xlabel('Frame')
        self.posture_ax2.set_xlabel('Frame')
        self.posture_ax3.set_xlabel('Frame')
        
        self.posture_ax1.set_ylabel('Sway')
        self.posture_ax2.set_ylabel('Energy')
        self.posture_ax3.set_ylabel('Stability')
        
        # Plot data for each track
        for track_id, history in self.posture_history.items():
            if len(history) > 1:
                frames = [item['frame'] for item in history]
                sways = [item['postural_sway'] for item in history]
                energies = [item['locomotion_energy'] for item in history]
                stabilities = [item['stability_score'] for item in history]
                
                color = [c/255 for c in self._get_color(track_id)]  # Convert to matplotlib RGB
                
                self.posture_ax1.plot(frames, sways, '-', color=color, label=f"ID {track_id}")
                self.posture_ax2.plot(frames, energies, '-', color=color, label=f"ID {track_id}")
                self.posture_ax3.plot(frames, stabilities, '-', color=color, label=f"ID {track_id}")
        
        # Add legends and grid
        self.posture_ax1.legend(loc='upper right')
        self.posture_ax2.legend(loc='upper right')
        self.posture_ax3.legend(loc='upper right')
        
        self.posture_ax1.grid(True)
        self.posture_ax2.grid(True)
        self.posture_ax3.grid(True)
        
        # Ensure good spacing
        self.posture_fig.tight_layout()
        
        # Convert plot to image
        canvas = FigureCanvasAgg(self.posture_fig)
        canvas.draw()
        plot_img = np.array(canvas.renderer.buffer_rgba())
        
        # Convert from RGBA to BGR
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        
        return plot_img
    
    def visualize(self, frame, tracks, speeds, frame_idx, keypoints=None, posture_metrics=None):
        """
        Visualize tracking and speed results
        
        Args:
            frame (numpy.ndarray): BGR image
            tracks (list): List of track states
            speeds (dict): Dictionary mapping track IDs to speeds
            frame_idx (int): Current frame index
            keypoints (dict, optional): Dictionary mapping track IDs to keypoints
            posture_metrics (dict, optional): Dictionary mapping track IDs to posture metrics
        """
        # Update history
        self._update_speed_history(speeds, frame_idx)
        
        if posture_metrics:
            self._update_posture_history(posture_metrics, frame_idx)
        
        # Draw tracks on frame
        vis_frame = self._draw_tracks(frame, tracks, speeds, keypoints, posture_metrics)
        
        # Initialize video writer if not already
        if self.save_video and self.video_writer is None:
            height, width = vis_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            self.video_writer = cv2.VideoWriter(
                os.path.join(self.output_dir, 'output.mp4'),
                fourcc, 30.0, (width, height)
            )
        
        # Draw plots periodically
        if frame_idx % 30 == 0:
            # Draw speed plot if we have speed data
            if self.speed_history:
                speed_plot_img = self._draw_speed_plot()
                
                # Save plot image
                if self.save_plots:
                    cv2.imwrite(os.path.join(self.output_dir, f'speed_plot_{frame_idx:06d}.png'), speed_plot_img)
            
            # Draw posture plots if we have posture data
            if self.posture_history:
                posture_plot_img = self._draw_posture_plots()
                
                # Save plot image
                if self.save_plots:
                    cv2.imwrite(os.path.join(self.output_dir, f'posture_plot_{frame_idx:06d}.png'), posture_plot_img)
        
        # Save frame to video
        if self.save_video:
            self.video_writer.write(vis_frame)
        
        # Show frame if requested
        if self.show:
            cv2.imshow('Vibrio Human Speed Tracking', vis_frame)
            cv2.waitKey(1)
    
    def finalize(self):
        """Finalize visualization and save final results"""
        # Save final speed plot
        if self.speed_history and self.save_plots:
            speed_plot_img = self._draw_speed_plot()
            cv2.imwrite(os.path.join(self.output_dir, 'final_speed_plot.png'), speed_plot_img)
            
            # Save speed data as CSV
            with open(os.path.join(self.output_dir, 'speed_data.csv'), 'w') as f:
                f.write('track_id,frame,speed\n')
                for track_id, history in self.speed_history.items():
                    for frame_idx, speed in history:
                        f.write(f'{track_id},{frame_idx},{speed:.2f}\n')
        
        # Save final posture plots
        if self.posture_history and self.save_plots:
            posture_plot_img = self._draw_posture_plots()
            cv2.imwrite(os.path.join(self.output_dir, 'final_posture_plots.png'), posture_plot_img)
            
            # Save posture data as CSV
            with open(os.path.join(self.output_dir, 'posture_data.csv'), 'w') as f:
                f.write('track_id,frame,postural_sway,locomotion_energy,stability_score\n')
                for track_id, history in self.posture_history.items():
                    for metrics in history:
                        f.write(f'{track_id},{metrics["frame"]},{metrics["postural_sway"]:.2f},' +
                                f'{metrics["locomotion_energy"]:.2f},{metrics["stability_score"]:.2f}\n')
        
        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
        
        # Close windows
        if self.show:
            cv2.destroyAllWindows()
            
        # Close matplotlib figures
        plt.close(self.speed_fig)
        plt.close(self.posture_fig) 
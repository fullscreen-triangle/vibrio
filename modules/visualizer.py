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
    
    def __init__(self, output_dir='results', show=False, save_video=True, save_plots=True):
        """
        Initialize the visualizer
        
        Args:
            output_dir (str): Directory to save results
            show (bool): Whether to show visualizations in real-time
            save_video (bool): Whether to save output video
            save_plots (bool): Whether to save speed plots
        """
        self.output_dir = output_dir
        self.show = show
        self.save_video = save_video
        self.save_plots = save_plots
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Video writer
        self.video_writer = None
        
        # Store speeds for plotting
        self.speed_history = {}  # {track_id: [(frame_idx, speed), ...]}
        
        # Plot figure for speed visualization
        self.speed_fig = plt.figure(figsize=(10, 6))
        self.speed_ax = self.speed_fig.add_subplot(111)
        self.speed_ax.set_title('Human Speed Tracking')
        self.speed_ax.set_xlabel('Frame')
        self.speed_ax.set_ylabel('Speed (km/h)')
    
    def _get_color(self, track_id):
        """Get color for a track"""
        return self.COLORS[track_id % len(self.COLORS)]
    
    def _draw_tracks(self, frame, tracks, speeds):
        """
        Draw tracks and speeds on frame
        
        Args:
            frame (numpy.ndarray): BGR image
            tracks (list): List of track states
            speeds (dict): Dictionary mapping track IDs to speeds
            
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
            
            # Draw trajectory
            history = track['history']
            for i in range(1, len(history)):
                pt1 = (int(history[i-1][0]), int(history[i-1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                cv2.line(vis_frame, pt1, pt2, color, 2)
        
        return vis_frame
    
    def _update_speed_history(self, speeds, frame_idx):
        """Update speed history for plotting"""
        for track_id, speed in speeds.items():
            if track_id not in self.speed_history:
                self.speed_history[track_id] = []
            
            self.speed_history[track_id].append((frame_idx, speed))
    
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
    
    def visualize(self, frame, tracks, speeds, frame_idx):
        """
        Visualize tracking and speed results
        
        Args:
            frame (numpy.ndarray): BGR image
            tracks (list): List of track states
            speeds (dict): Dictionary mapping track IDs to speeds
            frame_idx (int): Current frame index
        """
        # Update speed history
        self._update_speed_history(speeds, frame_idx)
        
        # Draw tracks on frame
        vis_frame = self._draw_tracks(frame, tracks, speeds)
        
        # Initialize video writer if not already
        if self.save_video and self.video_writer is None:
            height, width = vis_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
            self.video_writer = cv2.VideoWriter(
                os.path.join(self.output_dir, 'output.mp4'),
                fourcc, 30.0, (width, height)
            )
        
        # Draw speed plot
        if frame_idx % 30 == 0 and self.speed_history:  # Update plot every 30 frames
            plot_img = self._draw_speed_plot()
            
            # Save plot image
            if self.save_plots:
                cv2.imwrite(os.path.join(self.output_dir, f'speed_plot_{frame_idx:06d}.png'), plot_img)
        
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
            plot_img = self._draw_speed_plot()
            cv2.imwrite(os.path.join(self.output_dir, 'final_speed_plot.png'), plot_img)
            
            # Save speed data as CSV
            with open(os.path.join(self.output_dir, 'speed_data.csv'), 'w') as f:
                f.write('track_id,frame,speed\n')
                for track_id, history in self.speed_history.items():
                    for frame_idx, speed in history:
                        f.write(f'{track_id},{frame_idx},{speed:.2f}\n')
        
        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
        
        # Close windows
        if self.show:
            cv2.destroyAllWindows()
            
        # Close matplotlib figure
        plt.close(self.speed_fig) 
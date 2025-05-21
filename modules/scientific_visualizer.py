#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.path import Path as mpath

class ScientificVisualizer:
    """
    Scientific visualization module for Vibrio analysis results.
    Provides publication-quality visualizations for speed and biomechanical data.
    """
    
    def __init__(self, output_dir='results', style='science', dpi=300):
        """
        Initialize the scientific visualizer
        
        Args:
            output_dir (str): Directory to save results
            style (str): Matplotlib style for plots ('science', 'ieee', 'default')
            dpi (int): DPI for output images (higher for publication quality)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plot style
        # Note: If 'science' style is not available, fall back to default
        try:
            if style == 'science':
                plt.style.use(['science', 'grid'])
            elif style == 'ieee':
                plt.style.use(['science', 'ieee'])
            else:
                plt.style.use('default')
        except:
            plt.style.use('default')
            
        # Custom color palette that's colorblind-friendly
        self.colors = sns.color_palette("colorblind", 10)
        
        # Custom theme settings
        self.theme = {
            'background': '#FFFFFF',
            'text': '#333333',
            'grid': '#CCCCCC',
            'highlight': '#FF5500'
        }
    
    def load_data(self, results_dir):
        """
        Load data from result files
        
        Args:
            results_dir (str): Directory containing result files
            
        Returns:
            dict: Dictionary containing loaded data
        """
        data = {}
        
        # Speed data
        speed_file = os.path.join(results_dir, 'speed_data.csv')
        if os.path.exists(speed_file):
            try:
                data['speed'] = pd.read_csv(speed_file)
                print(f"Loaded speed data with {len(data['speed'])} records")
            except Exception as e:
                print(f"Error loading speed data: {e}")
        
        # Posture data
        posture_file = os.path.join(results_dir, 'posture_data.csv')
        if os.path.exists(posture_file):
            try:
                data['posture'] = pd.read_csv(posture_file)
                print(f"Loaded posture data with {len(data['posture'])} records")
            except Exception as e:
                print(f"Error loading posture data: {e}")
        
        return data
    
    def visualize_single_result(self, results_dir, output_filename=None):
        """
        Create scientific visualizations for a single result directory
        
        Args:
            results_dir (str): Directory containing result files
            output_filename (str, optional): Output filename for visualizations
        """
        # Load data
        data = self.load_data(results_dir)
        if not data:
            print(f"No data found in {results_dir}")
            return
        
        # Generate visualizations
        if 'speed' in data:
            self._create_speed_visualizations(data['speed'], results_dir, output_filename)
        
        if 'posture' in data:
            self._create_posture_visualizations(data['posture'], results_dir, output_filename)
        
        if 'speed' in data and 'posture' in data:
            self._create_combined_visualizations(data, results_dir, output_filename)
    
    def visualize_comparison(self, results_dirs, labels=None, output_dir=None, output_filename="comparison"):
        """
        Create comparison visualizations between multiple result directories
        
        Args:
            results_dirs (list): List of directories containing result files
            labels (list, optional): Labels for each dataset
            output_dir (str, optional): Output directory for visualizations
            output_filename (str, optional): Output filename for visualizations
        """
        if not labels:
            labels = [os.path.basename(d) for d in results_dirs]
        
        if not output_dir:
            output_dir = self.output_dir
        
        # Load all datasets
        datasets = []
        for i, dir_path in enumerate(results_dirs):
            data = self.load_data(dir_path)
            if data:
                # Add label to data
                if 'speed' in data:
                    data['speed']['dataset'] = labels[i]
                if 'posture' in data:
                    data['posture']['dataset'] = labels[i]
                datasets.append(data)
        
        if not datasets:
            print("No valid data found for comparison")
            return
        
        # Generate comparison visualizations
        self._create_speed_comparison(datasets, output_dir, output_filename)
        self._create_posture_comparison(datasets, output_dir, output_filename)
    
    def _create_speed_visualizations(self, speed_data, output_dir, filename_prefix=None):
        """Create scientific visualizations for speed data"""
        if filename_prefix is None:
            filename_prefix = os.path.basename(output_dir)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(12, 10), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # 1. Main speed trace plot
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_speed_traces(speed_data, ax1)
        
        # 2. Speed distribution plot
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_speed_distribution(speed_data, ax2)
        
        # 3. Speed histogram
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_speed_histogram(speed_data, ax3)
        
        # 4. Statistical summary
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_speed_statistics(speed_data, ax4)
        
        # 5. Speed vs distance plot (if available)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_speed_segments(speed_data, ax5)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename_prefix}_scientific_speed_analysis.png")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved speed visualization to {output_path}")
    
    def _create_posture_visualizations(self, posture_data, output_dir, filename_prefix=None):
        """Create scientific visualizations for posture data"""
        if filename_prefix is None:
            filename_prefix = os.path.basename(output_dir)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(12, 12), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 2)
        
        # 1. Postural sway plot
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_postural_sway(posture_data, ax1)
        
        # 2. Locomotion energy plot
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_locomotion_energy(posture_data, ax2)
        
        # 3. Stability score plot
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_stability_score(posture_data, ax3)
        
        # 4. Biomechanical correlation plot
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_biomechanical_correlations(posture_data, ax4)
        
        # 5. Multi-parametric summary
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_multiparametric_summary(posture_data, ax5)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename_prefix}_scientific_posture_analysis.png")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved posture visualization to {output_path}")
    
    def _create_combined_visualizations(self, data, output_dir, filename_prefix=None):
        """Create scientific visualizations combining speed and posture data"""
        if filename_prefix is None:
            filename_prefix = os.path.basename(output_dir)
        
        # Only proceed if we have both speed and posture data
        if 'speed' not in data or 'posture' not in data:
            return
            
        speed_data = data['speed']
        posture_data = data['posture']
        
        # Merge datasets on track_id and frame
        merged_data = pd.merge(
            speed_data, 
            posture_data, 
            on=['track_id', 'frame'], 
            how='inner'
        )
        
        if merged_data.empty:
            print("No overlapping data between speed and posture metrics")
            return
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(12, 14), dpi=self.dpi)
        gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
        
        # 1. Speed vs Postural Sway
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_correlation(merged_data, 'speed', 'postural_sway', 
                             'Speed (km/h)', 'Postural Sway', ax1)
        
        # 2. Speed vs Locomotion Energy
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_correlation(merged_data, 'speed', 'locomotion_energy', 
                             'Speed (km/h)', 'Locomotion Energy', ax2)
        
        # 3. Speed vs Stability Score
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_correlation(merged_data, 'speed', 'stability_score', 
                             'Speed (km/h)', 'Stability Score', ax3)
        
        # 4. Multi-parameter performance visualization
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_multi_parameter_radar(merged_data, ax4)
        
        # 5. Temporal correlation plot
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_temporal_correlation(merged_data, ax5)
        
        # 6. Integrated metric visualization
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_integrated_performance(merged_data, ax6)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename_prefix}_scientific_integrated_analysis.png")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved integrated visualization to {output_path}")
        
    def _plot_speed_traces(self, speed_data, ax):
        """Plot speed traces over time for each track_id"""
        for track_id, group_data in speed_data.groupby('track_id'):
            # Sort by frame to ensure proper time sequence
            group_data = group_data.sort_values('frame')
            
            # Apply smoothing for better visualization
            speeds = gaussian_filter1d(group_data['speed'].values, sigma=2)
            
            # Plot with scientific styling
            color = self.colors[track_id % len(self.colors)]
            ax.plot(group_data['frame'], speeds, '-', color=color, linewidth=2, alpha=0.8, 
                    label=f"Track {track_id}")
            
            # Calculate peak speed and mark it
            peak_idx = np.argmax(speeds)
            peak_frame = group_data['frame'].iloc[peak_idx]
            peak_speed = speeds[peak_idx]
            ax.plot(peak_frame, peak_speed, 'o', color=color, markersize=8)
            ax.annotate(f"{peak_speed:.1f} km/h", 
                     xy=(peak_frame, peak_speed),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9, color=color)
        
        # Add reference lines for speed thresholds
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100 km/h')
        ax.axhline(y=200, color='gray', linestyle='--', alpha=0.5, label='200 km/h')
        
        # Styling
        ax.set_title('Speed Trace Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Speed (km/h)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, fontsize=10)

    def _plot_speed_distribution(self, speed_data, ax):
        """Plot speed distribution for each track_id"""
        all_speeds = []
        all_colors = []
        all_labels = []
        
        for track_id, group_data in speed_data.groupby('track_id'):
            speeds = group_data['speed'].values
            if len(speeds) > 0:
                all_speeds.append(speeds)
                all_colors.append(self.colors[track_id % len(self.colors)])
                all_labels.append(f"Track {track_id}")
        
        # Create violin plot if we have data
        if all_speeds:
            parts = ax.violinplot(all_speeds, showmeans=True, showmedians=True)
            
            # Customize violin plots with appropriate colors
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(all_colors[i])
                pc.set_alpha(0.7)
            
            # Add stats and labels
            for i, speeds in enumerate(all_speeds):
                ax.text(i+1, np.min(speeds), f"{np.min(speeds):.1f}", 
                     ha='center', va='bottom', fontsize=8)
                ax.text(i+1, np.max(speeds), f"{np.max(speeds):.1f}", 
                     ha='center', va='top', fontsize=8)
        
        # Styling
        ax.set_title('Speed Distribution by Track', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.set_xticks(range(1, len(all_labels) + 1))
        ax.set_xticklabels(all_labels)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    def _plot_speed_histogram(self, speed_data, ax):
        """Plot histogram of speeds across all tracks"""
        # Apply kernel density estimation for a smoother histogram
        sns.histplot(speed_data['speed'], kde=True, ax=ax, bins=30, 
                   color=self.colors[0], alpha=0.7, line_kws={'linewidth': 2})
        
        # Add statistical annotations
        mean_speed = speed_data['speed'].mean()
        median_speed = speed_data['speed'].median()
        max_speed = speed_data['speed'].max()
        
        # Draw vertical lines for statistical markers
        ax.axvline(mean_speed, color=self.colors[1], linestyle='-', linewidth=2, 
                 label=f'Mean: {mean_speed:.1f} km/h')
        ax.axvline(median_speed, color=self.colors[2], linestyle='--', linewidth=2, 
                 label=f'Median: {median_speed:.1f} km/h')
        ax.axvline(max_speed, color=self.colors[3], linestyle=':', linewidth=2, 
                 label=f'Max: {max_speed:.1f} km/h')
        
        # Styling
        ax.set_title('Speed Distribution Histogram', fontsize=12, fontweight='bold')
        ax.set_xlabel('Speed (km/h)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, fontsize=8)

    def _plot_speed_statistics(self, speed_data, ax):
        """Plot statistical summary of speed data"""
        # Calculate statistics per track
        stats = []
        labels = []
        
        for track_id, group_data in speed_data.groupby('track_id'):
            stats.append({
                'mean': group_data['speed'].mean(),
                'median': group_data['speed'].median(),
                'max': group_data['speed'].max(),
                'min': group_data['speed'].min(),
                'std': group_data['speed'].std()
            })
            labels.append(f"Track {track_id}")
        
        # Create DataFrame for plotting
        stats_df = pd.DataFrame(stats)
        
        # Plot as a bar chart with error bars
        x = np.arange(len(labels))
        width = 0.15
        
        ax.bar(x - width*2, stats_df['mean'], width, label='Mean', color=self.colors[0], alpha=0.8)
        ax.bar(x - width, stats_df['median'], width, label='Median', color=self.colors[1], alpha=0.8)
        ax.bar(x, stats_df['max'], width, label='Max', color=self.colors[2], alpha=0.8)
        ax.bar(x + width, stats_df['min'], width, label='Min', color=self.colors[3], alpha=0.8)
        
        # Add error bars showing standard deviation
        ax.errorbar(x - width*2, stats_df['mean'], yerr=stats_df['std'], fmt='none', 
                  ecolor=self.colors[4], capsize=4, label='Std Dev')
        
        # Styling
        ax.set_title('Speed Statistics Summary', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

    def _plot_speed_segments(self, speed_data, ax):
        """Plot speed segments as a heatmap"""
        # Calculate cumulative distance or use frame number as proxy
        # This is simplified since we don't have distance measurements
        for track_id, group_data in speed_data.groupby('track_id'):
            # Sort by frame
            group_data = group_data.sort_values('frame')
            
            # Skip if insufficient data points
            if len(group_data) < 5:
                continue
                
            # Normalize frame numbers to represent distance (0-100%)
            frames = group_data['frame'].values
            norm_frames = (frames - frames.min()) / (frames.max() - frames.min()) * 100
            
            # Get speeds and apply smoothing
            speeds = gaussian_filter1d(group_data['speed'].values, sigma=2)
            
            # Plot with color gradient based on speed
            points = np.array([norm_frames, speeds]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a color map normalized to speed range
            norm = plt.Normalize(speeds.min(), speeds.max())
            lc = plt.matplotlib.collections.LineCollection(segments, cmap='plasma', norm=norm)
            lc.set_array(speeds)
            lc.set_linewidth(3)
            
            line = ax.add_collection(lc)
            
            # Add colorbar for this track
            if track_id == speed_data['track_id'].unique()[0]:  # Only add one colorbar
                cbar = plt.colorbar(line, ax=ax)
                cbar.set_label('Speed (km/h)', fontsize=10)
        
        # Set axis limits
        ax.set_xlim(0, 100)
        if speed_data['speed'].max() > 0:
            ax.set_ylim(0, speed_data['speed'].max() * 1.1)
        
        # Styling
        ax.set_title('Speed Profile by Track Position', fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance (% of track)', fontsize=10)
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    def _plot_postural_sway(self, posture_data, ax):
        """Plot postural sway metrics over time for each track_id"""
        for track_id, group_data in posture_data.groupby('track_id'):
            # Sort by frame to ensure proper time sequence
            group_data = group_data.sort_values('frame')
            
            # Skip if insufficient data
            if len(group_data) < 3:
                continue
                
            # Apply smoothing for better visualization
            sway_values = gaussian_filter1d(group_data['postural_sway'].values, sigma=1.5)
            
            # Plot with scientific styling
            color = self.colors[track_id % len(self.colors)]
            ax.plot(group_data['frame'], sway_values, '-', color=color, linewidth=2, alpha=0.8,
                  label=f"Track {track_id}")
            
            # Calculate mean and highlight it
            mean_sway = np.mean(sway_values)
            ax.axhline(y=mean_sway, color=color, linestyle='--', alpha=0.5)
            ax.text(group_data['frame'].iloc[-1], mean_sway, 
                  f"Mean: {mean_sway:.1f}", color=color, fontsize=8)
        
        # Styling
        ax.set_title('Postural Sway Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Postural Sway (a.u.)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, fontsize=8)

    def _plot_locomotion_energy(self, posture_data, ax):
        """Plot locomotion energy metrics over time for each track_id"""
        for track_id, group_data in posture_data.groupby('track_id'):
            # Sort by frame to ensure proper time sequence
            group_data = group_data.sort_values('frame')
            
            # Skip if insufficient data
            if len(group_data) < 3:
                continue
                
            # Apply smoothing for better visualization
            energy_values = gaussian_filter1d(group_data['locomotion_energy'].values, sigma=1.5)
            
            # Plot with scientific styling
            color = self.colors[track_id % len(self.colors)]
            ax.plot(group_data['frame'], energy_values, '-', color=color, linewidth=2, alpha=0.8,
                  label=f"Track {track_id}")
            
            # Calculate mean and highlight it
            mean_energy = np.mean(energy_values)
            ax.axhline(y=mean_energy, color=color, linestyle='--', alpha=0.5)
            ax.text(group_data['frame'].iloc[-1], mean_energy, 
                  f"Mean: {mean_energy:.1f}", color=color, fontsize=8)
        
        # Styling
        ax.set_title('Locomotion Energy Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Locomotion Energy (a.u.)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, fontsize=8)

    def _plot_stability_score(self, posture_data, ax):
        """Plot stability score metrics over time for each track_id"""
        for track_id, group_data in posture_data.groupby('track_id'):
            # Sort by frame to ensure proper time sequence
            group_data = group_data.sort_values('frame')
            
            # Skip if insufficient data
            if len(group_data) < 3:
                continue
                
            # Apply smoothing for better visualization
            stability_values = gaussian_filter1d(group_data['stability_score'].values, sigma=1.5)
            
            # Plot with scientific styling
            color = self.colors[track_id % len(self.colors)]
            ax.plot(group_data['frame'], stability_values, '-', color=color, linewidth=2, alpha=0.8,
                  label=f"Track {track_id}")
            
            # Calculate mean and highlight it
            mean_stability = np.mean(stability_values)
            ax.axhline(y=mean_stability, color=color, linestyle='--', alpha=0.5)
            ax.text(group_data['frame'].iloc[-1], mean_stability, 
                  f"Mean: {mean_stability:.2f}", color=color, fontsize=8)
        
        # Styling
        ax.set_title('Stability Score Analysis', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Stability Score (0-1)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', frameon=True, fontsize=8)
        
        # Fixed range for stability scores
        ax.set_ylim(0, 1.0)

    def _plot_biomechanical_correlations(self, posture_data, ax):
        """Plot correlations between biomechanical metrics"""
        # Skip if insufficient data
        if len(posture_data) < 5:
            ax.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Biomechanical Correlations', fontsize=12, fontweight='bold')
            return
        
        # Create scatter plot of postural sway vs. locomotion energy
        scatter = ax.scatter(
            posture_data['postural_sway'], 
            posture_data['locomotion_energy'],
            c=posture_data['stability_score'],  # Color by stability score
            cmap='viridis',
            alpha=0.7,
            s=50,
            edgecolor='w'
        )
        
        # Add color bar for stability score
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Stability Score', fontsize=10)
        
        # Calculate and plot correlation line
        if len(posture_data) > 2:
            # Calculate correlation
            corr = posture_data['postural_sway'].corr(posture_data['locomotion_energy'])
            
            # Plot regression line
            sns.regplot(
                x='postural_sway', 
                y='locomotion_energy', 
                data=posture_data,
                scatter=False,
                ax=ax,
                line_kws={'color': 'red', 'linestyle': '--'}
            )
            
            # Add correlation coefficient text
            ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, 
                  fontsize=10, fontweight='bold', 
                  bbox=dict(facecolor='white', alpha=0.7))
        
        # Styling
        ax.set_title('Postural Sway vs. Locomotion Energy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Postural Sway (a.u.)', fontsize=10)
        ax.set_ylabel('Locomotion Energy (a.u.)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)

    def _plot_multiparametric_summary(self, posture_data, ax):
        """Create a multi-parametric summary visualization of posture metrics"""
        # Group by track_id and calculate statistics
        track_stats = []
        
        for track_id, group_data in posture_data.groupby('track_id'):
            # Calculate statistics
            stats = {
                'track_id': track_id,
                'mean_sway': group_data['postural_sway'].mean(),
                'mean_energy': group_data['locomotion_energy'].mean(),
                'mean_stability': group_data['stability_score'].mean(),
                'std_sway': group_data['postural_sway'].std(),
                'std_energy': group_data['locomotion_energy'].std(),
                'std_stability': group_data['stability_score'].std(),
                'count': len(group_data)
            }
            track_stats.append(stats)
        
        if not track_stats:
            ax.text(0.5, 0.5, "Insufficient data for multi-parametric analysis", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Multi-parametric Biomechanical Summary', fontsize=12, fontweight='bold')
            return
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(track_stats)
        
        # Number of tracks
        num_tracks = len(stats_df)
        
        # Set up bar positions
        bar_width = 0.25
        r1 = np.arange(num_tracks)
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Scale the metrics to be on similar scales for visualization
        # Normalize to 0-1 range for comparison
        stats_df['norm_sway'] = stats_df['mean_sway'] / stats_df['mean_sway'].max()
        stats_df['norm_energy'] = stats_df['mean_energy'] / stats_df['mean_energy'].max()
        # Stability is already 0-1
        
        # Create grouped bar chart
        bars1 = ax.bar(r1, stats_df['norm_sway'], width=bar_width, label='Postural Sway', 
                     color=self.colors[0], yerr=stats_df['std_sway']/stats_df['mean_sway'].max())
        bars2 = ax.bar(r2, stats_df['norm_energy'], width=bar_width, label='Locomotion Energy', 
                     color=self.colors[1], yerr=stats_df['std_energy']/stats_df['mean_energy'].max())
        bars3 = ax.bar(r3, stats_df['mean_stability'], width=bar_width, label='Stability Score', 
                     color=self.colors[2], yerr=stats_df['std_stability'])
        
        # Add biomechanical efficiency score (composite metric)
        efficiency = 0.7*stats_df['mean_stability'] - 0.15*stats_df['norm_sway'] - 0.15*stats_df['norm_energy'] 
        efficiency = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min())  # Normalize to 0-1
        
        # Plot efficiency on secondary axis
        ax2 = ax.twinx()
        ax2.plot(r2, efficiency, 'o-', color=self.colors[5], linewidth=2, markersize=8, 
               label='Efficiency Score')
        ax2.set_ylabel('Biomechanical Efficiency Score', fontsize=10, color=self.colors[5])
        ax2.tick_params(axis='y', colors=self.colors[5])
        ax2.set_ylim(0, 1.1)
        
        # Add track IDs on x-axis
        ax.set_xticks([r + bar_width for r in range(num_tracks)])
        ax.set_xticklabels([f'Track {id}' for id in stats_df['track_id']])
        
        # Styling
        ax.set_title('Multi-parametric Biomechanical Analysis', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Metric Value (0-1)', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True, fontsize=9)
        
        # Add explanatory text
        ax.text(0.01, 0.01, 
              "Note: All metrics normalized to 0-1 scale. Efficiency score is a weighted composite.",
              transform=ax.transAxes, fontsize=8, style='italic')

    def _plot_correlation(self, data, x_col, y_col, x_label, y_label, ax):
        """Plot correlation between two variables"""
        # Skip if insufficient data
        if len(data) < 5:
            ax.text(0.5, 0.5, "Insufficient data for correlation analysis", 
                  ha='center', va='center', fontsize=10)
            ax.set_title(f'{y_label} vs {x_label}', fontsize=12, fontweight='bold')
            return
            
        # Create scatter plot
        scatter = ax.scatter(
            data[x_col], 
            data[y_col],
            c=data['frame'],  # Color by frame number to show time progression
            cmap='plasma',
            alpha=0.7,
            s=40,
            edgecolor='w'
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Frame Number', fontsize=9)
        
        # Calculate and plot correlation line
        # Calculate correlation
        corr = data[x_col].corr(data[y_col])
        pearson_r, p_value = stats.pearsonr(data[x_col], data[y_col])
        
        # Plot regression line
        sns.regplot(
            x=x_col, 
            y=y_col, 
            data=data,
            scatter=False,
            ax=ax,
            line_kws={'color': 'red', 'linestyle': '--'}
        )
        
        # Add correlation coefficient text
        significance = "p < 0.05 *" if p_value < 0.05 else "p > 0.05"
        ax.text(0.05, 0.95, f"r = {pearson_r:.3f}\n{significance}", 
              transform=ax.transAxes, fontsize=9, fontweight='bold',
              bbox=dict(facecolor='white', alpha=0.7))
        
        # Styling
        ax.set_title(f'{y_label} vs {x_label}', fontsize=12, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
    
    def _plot_multi_parameter_radar(self, data, ax):
        """Plot multi-parameter radar chart for performance visualization"""
        # For radar chart, we need a specialized polar plot
        # First, group by track_id and calculate average metrics
        track_means = data.groupby('track_id').agg({
            'speed': 'mean',
            'postural_sway': 'mean',
            'locomotion_energy': 'mean',
            'stability_score': 'mean'
        }).reset_index()
        
        if len(track_means) == 0:
            ax.text(0.5, 0.5, "Insufficient data for radar chart", 
                  ha='center', va='center', fontsize=10)
            ax.set_title('Performance Radar Chart', fontsize=12, fontweight='bold')
            return
        
        # Normalize metrics to 0-1 scale for radar chart
        metrics = ['speed', 'postural_sway', 'locomotion_energy', 'stability_score']
        for metric in metrics:
            if track_means[metric].max() > 0:
                track_means[f'{metric}_norm'] = track_means[metric] / track_means[metric].max()
            else:
                track_means[f'{metric}_norm'] = track_means[metric]
        
        # For radar chart, postural sway and locomotion energy need to be inverted
        # Lower values are better, so 1 - normalized value
        track_means['postural_sway_norm'] = 1 - track_means['postural_sway_norm']
        track_means['locomotion_energy_norm'] = 1 - track_means['locomotion_energy_norm']
        
        # Number of variables
        N = 4
        
        # Angles for each metric (evenly spaced)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Set up the figure as a polar plot
        ax.set_theta_offset(np.pi / 2)  # Start at top
        ax.set_theta_direction(-1)  # Go clockwise
        
        # Label positions (adjusted for readability)
        labels = ['Speed', 'Postural\nStability', 'Energy\nEfficiency', 'Stability\nScore']
        
        # Draw labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        
        # Draw ylabels (concentric circles)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8)
        ax.set_rlabel_position(0)  # Move labels away from center
        
        # Plot data for each track_id
        for i, row in track_means.iterrows():
            track_id = int(row['track_id'])
            color = self.colors[track_id % len(self.colors)]
            
            # Extract metrics for this track
            values = [
                row['speed_norm'],
                row['postural_sway_norm'],
                row['locomotion_energy_norm'],
                row['stability_score_norm']
            ]
            values += values[:1]  # Close the loop
            
            # Plot metrics
            ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, 
                  label=f"Track {track_id}")
            ax.fill(angles, values, color=color, alpha=0.2)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=8)
        
        # Title
        ax.set_title('Performance Radar Chart', fontsize=12, fontweight='bold', pad=15)
        
        # Gridlines
        ax.grid(True, linestyle='--', alpha=0.5)
    
    def _plot_temporal_correlation(self, data, ax):
        """Plot temporal correlation of speed and biomechanical parameters"""
        # Skip if insufficient data
        if len(data) < 5:
            ax.text(0.5, 0.5, "Insufficient data for temporal correlation", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Temporal Correlation Analysis', fontsize=12, fontweight='bold')
            return
        
        # Sort by frame to ensure proper time sequence
        data = data.sort_values('frame')
        
        # Create figure with 3 metrics on one plot, with shared x-axis (frame number)
        # Primary axis: Speed
        ax.plot(data['frame'], data['speed'], 'b-', linewidth=2, label='Speed (km/h)')
        ax.set_xlabel('Frame Number', fontsize=10)
        ax.set_ylabel('Speed (km/h)', fontsize=10, color='b')
        ax.tick_params(axis='y', colors='b')
        
        # Secondary axis 1: Postural Sway
        ax2 = ax.twinx()
        ax2.plot(data['frame'], data['postural_sway'], 'r-', linewidth=2, label='Postural Sway')
        ax2.set_ylabel('Postural Sway', fontsize=10, color='r')
        ax2.tick_params(axis='y', colors='r')
        
        # Secondary axis 2: Stability Score (on a separate spine)
        ax3 = ax.twinx()
        # Move this spine to the right
        ax3.spines["right"].set_position(("axes", 1.1))
        ax3.plot(data['frame'], data['stability_score'], 'g-', linewidth=2, label='Stability Score')
        ax3.set_ylabel('Stability Score', fontsize=10, color='g')
        ax3.tick_params(axis='y', colors='g')
        
        # Set title
        ax.set_title('Temporal Correlation of Speed and Biomechanical Parameters', 
                   fontsize=14, fontweight='bold')
        
        # Combine legends from all three axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                ncol=3, frameon=True, fontsize=9)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.4)
    
    def _plot_integrated_performance(self, data, ax):
        """Plot integrated performance metrics"""
        # Skip if insufficient data
        if len(data) < 5:
            ax.text(0.5, 0.5, "Insufficient data for integrated performance analysis", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Integrated Performance Analysis', fontsize=14, fontweight='bold')
            return
        
        # Group by track_id
        grouped = data.groupby('track_id')
        
        # Get unique track IDs
        track_ids = data['track_id'].unique()
        
        # Get labels and metrics
        num_tracks = len(track_ids)
        x_pos = np.arange(num_tracks)
        width = 0.35
        
        # Calculate performance metrics
        max_speeds = []
        avg_speeds = []
        min_sway = []
        biomech_score = []
        
        for track_id in track_ids:
            track_data = data[data['track_id'] == track_id]
            
            max_speed = track_data['speed'].max()
            avg_speed = track_data['speed'].mean()
            
            # Calculate biomechanical score
            # Lower postural sway is better, higher stability score is better
            # Normalize both to 0-1 scale and combine
            if len(track_data) > 0:
                mean_sway = track_data['postural_sway'].mean()
                mean_stability = track_data['stability_score'].mean()
                mean_energy = track_data['locomotion_energy'].mean()
                
                # Composite biomechanical score (higher is better)
                # Weight factors can be adjusted based on domain knowledge
                score = 0.5 * mean_stability - 0.25 * (mean_sway / data['postural_sway'].max()) - 0.25 * (mean_energy / data['locomotion_energy'].max())
                score = (score + 0.5) * 100  # Scale to 0-100 range
            else:
                score = 0
            
            max_speeds.append(max_speed)
            avg_speeds.append(avg_speed)
            min_sway.append(track_data['postural_sway'].min())
            biomech_score.append(score)
        
        # Create bar chart comparing tracks
        ax.bar(x_pos - width/2, max_speeds, width, label='Max Speed (km/h)', color=self.colors[0])
        ax.bar(x_pos + width/2, biomech_score, width, label='Biomechanical Score (0-100)', color=self.colors[1])
        
        # Add labels
        ax.set_xlabel('Track ID', fontsize=10)
        ax.set_ylabel('Metrics', fontsize=10)
        ax.set_title('Integrated Performance Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Track {id}' for id in track_ids])
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.4, axis='y')
        
        # Add text summary
        summary_text = "Performance Summary:\n"
        for i, track_id in enumerate(track_ids):
            summary_text += f"Track {track_id}: Max Speed = {max_speeds[i]:.1f} km/h, Biomech. Score = {biomech_score[i]:.1f}/100\n"
            
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, fontsize=9,
              bbox=dict(facecolor='white', alpha=0.8, boxstyle="round,pad=0.5"))
    
    def _create_speed_comparison(self, datasets, output_dir, filename_prefix="comparison"):
        """Create comparison visualizations for speed data"""
        # Extract speed datasets from all datasets
        speed_data = []
        labels = []
        
        for i, data in enumerate(datasets):
            if 'speed' in data:
                speed_data.append(data['speed'])
                # Get dataset label if available
                if 'dataset' in data['speed'].columns:
                    label = data['speed']['dataset'].iloc[0]
                else:
                    label = f"Dataset {i+1}"
                labels.append(label)
        
        if not speed_data:
            print("No speed data available for comparison")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # 1. Speed distribution comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_speed_distribution_comparison(speed_data, labels, ax1)
        
        # 2. Max speed comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_max_speed_comparison(speed_data, labels, ax2)
        
        # 3. Speed profile comparison
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_speed_profile_comparison(speed_data, labels, ax3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename_prefix}_speed_comparison.png")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved speed comparison visualization to {output_path}")
    
    def _create_posture_comparison(self, datasets, output_dir, filename_prefix="comparison"):
        """Create comparison visualizations for posture data"""
        # Extract posture datasets from all datasets
        posture_data = []
        labels = []
        
        for i, data in enumerate(datasets):
            if 'posture' in data:
                posture_data.append(data['posture'])
                # Get dataset label if available
                if 'dataset' in data['posture'].columns:
                    label = data['posture']['dataset'].iloc[0]
                else:
                    label = f"Dataset {i+1}"
                labels.append(label)
        
        if not posture_data:
            print("No posture data available for comparison")
            return
            
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 12), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # 1. Postural sway comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_postural_sway_comparison(posture_data, labels, ax1)
        
        # 2. Stability score comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_stability_score_comparison(posture_data, labels, ax2)
        
        # 3. Biomechanical summary comparison
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_biomechanical_summary_comparison(posture_data, labels, ax3)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f"{filename_prefix}_posture_comparison.png")
        fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved posture comparison visualization to {output_path}")
    
    def _plot_speed_distribution_comparison(self, speed_data_list, labels, ax):
        """Plot speed distribution comparison across datasets"""
        # Prepare data for box plot
        data_for_plot = []
        
        for data in speed_data_list:
            data_for_plot.append(data['speed'].values)
        
        # Create violin plot
        parts = ax.violinplot(data_for_plot, showmeans=True, showmedians=True)
        
        # Customize violin plots with appropriate colors
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.7)
        
        # Add labels
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        
        # Add stats
        for i, speeds in enumerate(data_for_plot):
            mean_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            ax.text(i+1, np.min(speeds), f"{np.min(speeds):.1f}", 
                  ha='center', va='bottom', fontsize=8)
            ax.text(i+1, max_speed, f"{max_speed:.1f}", 
                  ha='center', va='top', fontsize=8)
        
        # Styling
        ax.set_title('Speed Distribution Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    def _plot_max_speed_comparison(self, speed_data_list, labels, ax):
        """Plot max speed comparison across datasets"""
        max_speeds = []
        avg_speeds = []
        
        for data in speed_data_list:
            max_speeds.append(data['speed'].max())
            avg_speeds.append(data['speed'].mean())
        
        # Create bar chart
        x_pos = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x_pos - width/2, max_speeds, width, label='Max Speed', color=self.colors[0])
        ax.bar(x_pos + width/2, avg_speeds, width, label='Avg Speed', color=self.colors[1])
        
        # Add labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels)
        
        # Add speed values above bars
        for i, (max_speed, avg_speed) in enumerate(zip(max_speeds, avg_speeds)):
            ax.text(i - width/2, max_speed + 1, f"{max_speed:.1f}", 
                  ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, avg_speed + 1, f"{avg_speed:.1f}", 
                  ha='center', va='bottom', fontsize=9)
        
        # Styling
        ax.set_title('Maximum and Average Speed Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    def _plot_speed_profile_comparison(self, speed_data_list, labels, ax):
        """Plot speed profile comparison across datasets"""
        for i, data in enumerate(speed_data_list):
            # Group by track_id
            for track_id, group_data in data.groupby('track_id'):
                # Sort by frame
                group_data = group_data.sort_values('frame')
                
                # Skip if insufficient data points
                if len(group_data) < 5:
                    continue
                    
                # Normalize frame numbers to represent distance (0-100%)
                frames = group_data['frame'].values
                if len(frames) > 1 and frames.max() > frames.min():
                    norm_frames = (frames - frames.min()) / (frames.max() - frames.min()) * 100
                    
                    # Get speeds and apply smoothing
                    speeds = gaussian_filter1d(group_data['speed'].values, sigma=2)
                    
                    # Plot with different line style for each dataset
                    color = self.colors[i % len(self.colors)]
                    ax.plot(norm_frames, speeds, '-', color=color, linewidth=2, alpha=0.8,
                          label=f"{labels[i]} - Track {track_id}")
        
        # Styling
        ax.set_title('Speed Profile Comparison', fontsize=12, fontweight='bold')
        ax.set_xlabel('Normalized Distance (%)', fontsize=10)
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best', frameon=True, fontsize=8)
    
    def _plot_postural_sway_comparison(self, posture_data_list, labels, ax):
        """Plot postural sway comparison across datasets"""
        sway_stats = []
        
        for i, data in enumerate(posture_data_list):
            # Calculate statistics for each track
            for track_id, group_data in data.groupby('track_id'):
                mean_sway = group_data['postural_sway'].mean()
                std_sway = group_data['postural_sway'].std()
                min_sway = group_data['postural_sway'].min()
                max_sway = group_data['postural_sway'].max()
                
                sway_stats.append({
                    'dataset': labels[i],
                    'track_id': track_id,
                    'mean_sway': mean_sway,
                    'std_sway': std_sway,
                    'min_sway': min_sway,
                    'max_sway': max_sway
                })
        
        if not sway_stats:
            ax.text(0.5, 0.5, "No postural sway data available for comparison", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Postural Sway Comparison', fontsize=12, fontweight='bold')
            return
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(sway_stats)
        
        # Create bar chart grouped by dataset
        datasets = stats_df['dataset'].unique()
        track_ids = stats_df['track_id'].unique()
        
        # Set up positions
        x_positions = []
        bar_width = 0.8 / len(track_ids)
        for i in range(len(datasets)):
            for j in range(len(track_ids)):
                x_positions.append(i + (j - len(track_ids)/2 + 0.5) * bar_width)
        
        # Create bars
        sway_values = []
        std_values = []
        colors = []
        
        for dataset in datasets:
            for track_id in track_ids:
                subset = stats_df[(stats_df['dataset'] == dataset) & (stats_df['track_id'] == track_id)]
                if not subset.empty:
                    sway_values.append(subset['mean_sway'].values[0])
                    std_values.append(subset['std_sway'].values[0])
                    colors.append(self.colors[int(track_id) % len(self.colors)])
                else:
                    sway_values.append(0)
                    std_values.append(0)
                    colors.append(self.colors[0])
        
        # Plot bars
        bars = ax.bar(x_positions, sway_values, width=bar_width, yerr=std_values, 
                    color=colors, capsize=3)
        
        # Create legend
        legend_elements = []
        for i, track_id in enumerate(track_ids):
            color = self.colors[int(track_id) % len(self.colors)]
            legend_elements.append(mpatches.Patch(facecolor=color, label=f'Track {track_id}'))
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
        
        # Add dataset labels
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(datasets)
        
        # Styling
        ax.set_title('Postural Sway Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Postural Sway (a.u.)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    def _plot_stability_score_comparison(self, posture_data_list, labels, ax):
        """Plot stability score comparison across datasets"""
        stability_stats = []
        
        for i, data in enumerate(posture_data_list):
            # Calculate statistics for each track
            for track_id, group_data in data.groupby('track_id'):
                mean_stability = group_data['stability_score'].mean()
                std_stability = group_data['stability_score'].std()
                
                stability_stats.append({
                    'dataset': labels[i],
                    'track_id': track_id,
                    'mean_stability': mean_stability,
                    'std_stability': std_stability
                })
        
        if not stability_stats:
            ax.text(0.5, 0.5, "No stability score data available for comparison", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Stability Score Comparison', fontsize=12, fontweight='bold')
            return
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(stability_stats)
        
        # Create bar chart grouped by dataset
        datasets = stats_df['dataset'].unique()
        track_ids = stats_df['track_id'].unique()
        
        # Set up positions
        x_positions = []
        bar_width = 0.8 / len(track_ids)
        for i in range(len(datasets)):
            for j in range(len(track_ids)):
                x_positions.append(i + (j - len(track_ids)/2 + 0.5) * bar_width)
        
        # Create bars
        stability_values = []
        std_values = []
        colors = []
        
        for dataset in datasets:
            for track_id in track_ids:
                subset = stats_df[(stats_df['dataset'] == dataset) & (stats_df['track_id'] == track_id)]
                if not subset.empty:
                    stability_values.append(subset['mean_stability'].values[0])
                    std_values.append(subset['std_stability'].values[0])
                    colors.append(self.colors[int(track_id) % len(self.colors)])
                else:
                    stability_values.append(0)
                    std_values.append(0)
                    colors.append(self.colors[0])
        
        # Plot bars
        bars = ax.bar(x_positions, stability_values, width=bar_width, yerr=std_values, 
                    color=colors, capsize=3)
        
        # Create legend
        legend_elements = []
        for i, track_id in enumerate(track_ids):
            color = self.colors[int(track_id) % len(self.colors)]
            legend_elements.append(mpatches.Patch(facecolor=color, label=f'Track {track_id}'))
        
        ax.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=8)
        
        # Add dataset labels
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(datasets)
        
        # Styling
        ax.set_title('Stability Score Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel('Stability Score (0-1)', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    def _plot_biomechanical_summary_comparison(self, posture_data_list, labels, ax):
        """Plot biomechanical summary comparison across datasets"""
        # Calculate aggregate biomechanical metrics for each dataset and track
        bio_stats = []
        
        for i, data in enumerate(posture_data_list):
            # Group by track_id
            for track_id, group_data in data.groupby('track_id'):
                # Calculate metrics
                mean_sway = group_data['postural_sway'].mean()
                mean_energy = group_data['locomotion_energy'].mean()
                mean_stability = group_data['stability_score'].mean()
                
                # Calculate biomechanical score (higher is better)
                # Invert sway and energy (lower is better)
                if len(posture_data_list) > 1:
                    # Get max values across all datasets for normalization
                    all_sway = pd.concat([d['postural_sway'] for d in posture_data_list])
                    all_energy = pd.concat([d['locomotion_energy'] for d in posture_data_list])
                    
                    max_sway = all_sway.max()
                    max_energy = all_energy.max()
                else:
                    max_sway = data['postural_sway'].max()
                    max_energy = data['locomotion_energy'].max()
                
                # Normalize metrics
                norm_sway = mean_sway / max_sway if max_sway > 0 else 0
                norm_energy = mean_energy / max_energy if max_energy > 0 else 0
                
                # Calculate composite score (higher is better)
                bio_score = mean_stability - 0.5 * norm_sway - 0.5 * norm_energy
                bio_score = (bio_score + 1) / 2  # Scale to 0-1 range
                
                bio_stats.append({
                    'dataset': labels[i],
                    'track_id': track_id,
                    'biomech_score': bio_score * 100,  # Scale to 0-100 for readability
                    'stability': mean_stability * 100,
                    'sway': (1 - norm_sway) * 100,  # Invert so higher is better
                    'energy': (1 - norm_energy) * 100  # Invert so higher is better
                })
        
        if not bio_stats:
            ax.text(0.5, 0.5, "No biomechanical data available for comparison", 
                  ha='center', va='center', fontsize=12)
            ax.set_title('Biomechanical Performance Comparison', fontsize=14, fontweight='bold')
            return
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(bio_stats)
        
        # Set up the x-positions
        datasets = stats_df['dataset'].unique()
        x_pos = np.arange(len(datasets))
        
        # Set up colors for each track
        track_ids = stats_df['track_id'].unique()
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
        
        # Plot biomechanical scores for each track across datasets
        for i, track_id in enumerate(track_ids):
            track_data = stats_df[stats_df['track_id'] == track_id]
            
            # Extract scores for each dataset
            scores = []
            for dataset in datasets:
                dataset_score = track_data[track_data['dataset'] == dataset]['biomech_score']
                scores.append(dataset_score.iloc[0] if not dataset_score.empty else 0)
            
            # Plot with connecting lines
            color = self.colors[i % len(self.colors)]
            marker = markers[i % len(markers)]
            ax.plot(x_pos, scores, marker=marker, markersize=10, linewidth=2, 
                  color=color, label=f'Track {track_id}')
            
            # Add score labels
            for j, score in enumerate(scores):
                ax.text(x_pos[j], score + 2, f"{score:.1f}", ha='center', fontsize=9, 
                      color=color, fontweight='bold')
        
        # Styling
        ax.set_title('Biomechanical Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(datasets)
        ax.set_ylabel('Biomechanical Score (0-100)', fontsize=10)
        ax.set_ylim(0, 105)  # Leave room for text labels
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        ax.legend(loc='upper right', frameon=True, fontsize=9) 
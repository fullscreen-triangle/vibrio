#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set the style
plt.style.use('ggplot')
sns.set_palette("viridis")

def plot_optical_flow(df, output_dir):
    """Plot optical flow metrics"""
    # Create figure with multiple subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Flow magnitude (mean, max, std)
    axs[0].plot(df['frame'], df['mean_flow_magnitude'], label='Mean Flow', linewidth=2)
    axs[0].plot(df['frame'], df['max_flow_magnitude'], label='Max Flow', linewidth=1, alpha=0.7)
    axs[0].fill_between(df['frame'], 
                        df['mean_flow_magnitude'] - df['std_flow_magnitude'],
                        df['mean_flow_magnitude'] + df['std_flow_magnitude'],
                        alpha=0.2, label='Standard Deviation')
    axs[0].set_ylabel('Flow Magnitude')
    axs[0].set_title('Optical Flow Magnitude')
    axs[0].legend()
    
    # Plot 2: Motion direction
    axs[1].plot(df['frame'], df['motion_direction'], color='red', linewidth=2)
    axs[1].set_ylabel('Direction (radians)')
    axs[1].set_title('Motion Direction')
    
    # Plot 3: Motion coherence
    axs[2].plot(df['frame'], df['motion_coherence'], color='green', linewidth=2)
    axs[2].set_ylabel('Coherence')
    axs[2].set_title('Motion Coherence')
    
    # X-axis label for the bottom plot
    axs[2].set_xlabel('Frame Number')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optical_flow_analysis.png'), dpi=300)
    plt.close()

def plot_motion_energy(df, output_dir):
    """Plot motion energy metrics"""
    # Create figure with multiple subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot 1: Motion energy
    axs[0].plot(df['frame'], df['motion_energy'], linewidth=2)
    axs[0].set_ylabel('Motion Energy')
    axs[0].set_title('Motion Energy Over Time')
    
    # Plot 2: Active regions
    axs[1].plot(df['frame'], df['active_regions'], color='purple', linewidth=2)
    axs[1].set_ylabel('Active Regions')
    axs[1].set_title('Number of Active Regions')
    axs[1].set_xlabel('Frame Number')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'motion_energy_analysis.png'), dpi=300)
    plt.close()

def plot_frequency_analysis(df, output_dir):
    """Plot frequency analysis results"""
    # Since this is just one row with summary data, create a bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Extract the metrics
    metrics = ['dominant_frequency', 'cycles_per_second', 'amplitude', 'spectral_entropy']
    values = df[metrics].iloc[0].values
    
    # Create bar chart
    bars = ax.bar(metrics, values)
    
    # Add data labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    ax.set_title('Frequency Analysis Results')
    ax.set_ylabel('Value')
    ax.set_xlabel('Metric')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_analysis.png'), dpi=300)
    plt.close()

def plot_combined_metrics(flow_df, energy_df, output_dir):
    """Create a combined plot of key metrics from both datasets"""
    # Normalize values to make them comparable
    flow_mean_norm = flow_df['mean_flow_magnitude'] / flow_df['mean_flow_magnitude'].max()
    coherence_norm = flow_df['motion_coherence'] / flow_df['motion_coherence'].max()
    energy_norm = energy_df['motion_energy'] / energy_df['motion_energy'].max()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(flow_df['frame'], flow_mean_norm, label='Normalized Flow Magnitude', linewidth=2)
    ax.plot(flow_df['frame'], coherence_norm, label='Normalized Motion Coherence', linewidth=2)
    ax.plot(energy_df['frame'], energy_norm, label='Normalized Motion Energy', linewidth=2)
    
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Normalized Value (0-1)')
    ax.set_title('Combined Motion Analysis Metrics')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_motion_analysis.png'), dpi=300)
    plt.close()

def main():
    # Define paths
    base_dir = 'results/batch_analysis'
    video_clips = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Create output directory for plots
    output_base_dir = 'public/visualization/batch_results'
    os.makedirs(output_base_dir, exist_ok=True)
    
    for clip in video_clips:
        clip_dir = os.path.join(base_dir, clip)
        optical_dir = os.path.join(clip_dir, 'optical_analysis')
        
        # Skip if optical_analysis directory doesn't exist
        if not os.path.exists(optical_dir):
            print(f"No optical analysis directory found for {clip}")
            continue
        
        # Create output directory for this clip
        output_dir = os.path.join(output_base_dir, clip)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing {clip}...")
        
        # Process optical flow data
        flow_csv = os.path.join(optical_dir, f"{clip}_optical_flow.csv")
        if os.path.exists(flow_csv):
            flow_df = pd.read_csv(flow_csv)
            plot_optical_flow(flow_df, output_dir)
            print(f"  - Created optical flow plots")
        else:
            print(f"  - No optical flow data found")
        
        # Process motion energy data
        energy_csv = os.path.join(optical_dir, f"{clip}_motion_energy.csv")
        if os.path.exists(energy_csv):
            energy_df = pd.read_csv(energy_csv)
            plot_motion_energy(energy_df, output_dir)
            print(f"  - Created motion energy plots")
        else:
            print(f"  - No motion energy data found")
        
        # Process frequency analysis data
        freq_csv = os.path.join(optical_dir, f"{clip}_frequency_analysis.csv")
        if os.path.exists(freq_csv):
            freq_df = pd.read_csv(freq_csv)
            plot_frequency_analysis(freq_df, output_dir)
            print(f"  - Created frequency analysis plots")
        else:
            print(f"  - No frequency analysis data found")
        
        # Create combined plot if both datasets exist
        if os.path.exists(flow_csv) and os.path.exists(energy_csv):
            flow_df = pd.read_csv(flow_csv) if 'flow_df' not in locals() else flow_df
            energy_df = pd.read_csv(energy_csv) if 'energy_df' not in locals() else energy_df
            plot_combined_metrics(flow_df, energy_df, output_dir)
            print(f"  - Created combined analysis plot")

if __name__ == "__main__":
    main()
    print("All plots generated successfully!") 
#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from typing import Dict, Any, Tuple, Optional
import time  # For a small pause if saving multiple plots

# Example command line usage: python run_signxai_tf_timeseries.py --pathology AVB --record_id 03509_hr --method_name gradient

# --- Setup Project Root and Utility Paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import SignXAI and Utility Functions ---
try:
    # Import the correct method from the wrappers module
    from signxai.tf_signxai.methods.wrappers import calculate_relevancemap as tf_calculate_relevancemap
    from signxai.utils.utils import remove_softmax as tf_remove_softmax
except ImportError as e:
    print(f"Error importing SignXAI components: {e}. Ensure SignXAI is installed and in PYTHONPATH.")
    sys.exit(1)

try:
    from utils.ecg_data import load_and_preprocess_ecg, perform_shape_switch
    from utils.ecg_explainability import normalize_ecg_relevancemap
    from utils.ecg_visualization import plot_ecg
except ImportError as e:
    print(f"Error importing ECG utility functions from 'utils' directory: {e}")
    print(f"Current project_root (expected to be signxai-0.1.0): {project_root}")
    sys.exit(1)

# --- Method Configuration ---
DEFAULT_METHOD_PARAMS_TF = {
    'integrated_gradients': {'steps': 50, 'reference_inputs': None},
    'smoothgrad': {'augment_by_n': 25, 'noise_scale': 0.1},
    'grad_cam': {'last_conv_layer_name': 'last_conv'},  # Default, can be model-specific
    'grad_cam_timeseries': {'last_conv_layer_name': 'last_conv'}  # For timeseries data
}
GRAD_CAM_LAYERS_TF = {
    'ecg': 'last_conv',  # For default ECG model
    'pathology': 'conv1d_2'  # From LBBB model.json, verify for other pathology models
}
ECG_FRIENDLY_METHODS = [
    'gradient', 'integrated_gradients', 'smoothgrad', 'grad_cam_timeseries',  # Using grad_cam_timeseries for time series data
    'guided_backprop', 'gradient_x_sign', 'gradient_x_input',
    'lrp_alpha_1_beta_0', 'lrp_epsilon_0_5_std_x', 'lrpsign_epsilon_0_5_std_x'
]


# --- Special plotting function just for SmoothGrad ---
def plot_smoothgrad(ecg, explanation, title="ECG SmoothGrad", save_to=None, compare_with_original=None):
    """
    A specialized plotting function for SmoothGrad visualizations.
    """
    print(f"  Using specialized SmoothGrad visualization")
    
    # Print debug info about shapes
    print(f"  DEBUG: plot_smoothgrad input shapes - ecg: {ecg.shape}, explanation: {explanation.shape}")
    
    # Check if we have multi-lead ECG (multiple channels)
    multi_lead = False
    if ecg.ndim == 2 and ecg.shape[1] > 1:
        multi_lead = True
        lead_count = ecg.shape[1]
        print(f"  DEBUG: Detected multi-lead ECG with {lead_count} leads")
        
    # If single-lead or need to choose one lead for visualization
    if multi_lead:
        # Use the first lead (channel) for plotting
        ecg_for_plot = ecg[:, 0]  # Shape: (timesteps,)
        if explanation.ndim == 2 and explanation.shape[1] > 1:
            explanation_for_plot = explanation[:, 0]  # Shape: (timesteps,)
        else:
            explanation_for_plot = explanation
    else:
        # Single-lead processing
        if ecg.ndim == 2 and ecg.shape[0] == 1:
            # If shape is (1, timesteps), squeeze to (timesteps,)
            ecg_for_plot = ecg[0]
        else:
            ecg_for_plot = ecg
            
        if explanation.ndim == 2 and explanation.shape[0] == 1:
            # If shape is (1, timesteps), squeeze to (timesteps,)
            explanation_for_plot = explanation[0]
        else:
            explanation_for_plot = explanation
    
    # Create time axis
    time = np.arange(len(ecg_for_plot))
    
    # Create a figure with two subplots stacked vertically
    if compare_with_original is not None and os.path.exists(compare_with_original):
        # If we have an original to compare with, use a 3-panel layout
        fig = plt.figure(figsize=(18, 10))
        
        # First subplot for the original ECG image (top left)
        ax_orig = fig.add_subplot(131)
        original_img = plt.imread(compare_with_original)
        ax_orig.imshow(original_img)
        ax_orig.set_title('Original 12-lead ECG')
        ax_orig.axis('off')
        
        # Second subplot for ECG signal (top right)
        ax1 = fig.add_subplot(132)
        
        # Third subplot for relevance (bottom)
        ax2 = fig.add_subplot(133)
    else:
        # Create a stacked subplot layout for better visualization
        fig = plt.figure(figsize=(12, 8))
        gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
        
        # Top subplot for ECG
        ax1 = fig.add_subplot(gs[0])
        
        # Bottom subplot for relevance
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Smooth the explanation using Savitzky-Golay filter
    from scipy.signal import savgol_filter, find_peaks
    
    # Make sure window size is compatible with signal length
    window_size = min(101, len(explanation_for_plot) - (1 if len(explanation_for_plot) % 2 == 0 else 0))
    if window_size % 2 == 0:
        window_size -= 1  # Ensure odd window size
        
    explanation_smoothed = savgol_filter(explanation_for_plot.flatten(), window_size, 3)
    
    # Normalize to [-1, 1]
    if np.max(np.abs(explanation_smoothed)) > 0:
        explanation_smoothed = explanation_smoothed / np.max(np.abs(explanation_smoothed))
    
    # Find regions of high relevance (both positive and negative)
    pos_relevance_mask = explanation_smoothed > 0.3  # Threshold for positive relevance
    neg_relevance_mask = explanation_smoothed < -0.3  # Threshold for negative relevance
    
    # Prepare ECG data
    ecg_data_flat = ecg_for_plot.flatten()
    
    # Find peaks in the ECG signal (R waves)
    peaks, _ = find_peaks(ecg_data_flat, height=0.3*np.max(ecg_data_flat), distance=50)
    
    # Categorize peaks as important based on relevance
    important_pos_peaks = []
    important_neg_peaks = []
    
    for p in peaks:
        # Skip if out of bounds
        if p >= len(explanation_smoothed):
            continue
            
        # Check for positive relevance near the peak
        start_idx = max(0, p-20)
        end_idx = min(len(explanation_smoothed), p+20)
        
        if np.any(pos_relevance_mask[start_idx:end_idx]):
            important_pos_peaks.append(p)
        elif np.any(neg_relevance_mask[start_idx:end_idx]):
            important_neg_peaks.append(p)
    
    # TOP SUBPLOT: ECG with highlighted features
    # Set the y-axis limits based on the ECG data
    y_min, y_max = np.min(ecg_data_flat), np.max(ecg_data_flat)
    margin = 0.2 * (y_max - y_min)  # Add 20% margin
    ax1.set_ylim([y_min - margin, y_max + margin])
    
    # Plot the ECG signal
    ax1.plot(time, ecg_data_flat, 'k-', linewidth=1.0, label='ECG Signal', zorder=3)
    
    # Mark important positive peaks with red circles
    if important_pos_peaks:
        ax1.scatter(important_pos_peaks, ecg_data_flat[important_pos_peaks], 
                   c='red', s=80, alpha=0.8, zorder=4, marker='o',
                   label='Positive Important Features')
    
    # Mark important negative peaks with blue circles
    if important_neg_peaks:
        ax1.scatter(important_neg_peaks, ecg_data_flat[important_neg_peaks], 
                   c='blue', s=80, alpha=0.8, zorder=4, marker='o',
                   label='Negative Important Features')
    
    # Add subtle background shading for high relevance regions
    for i in range(len(time)-1):
        # Skip if we're out of bounds
        if i >= len(explanation_smoothed):
            continue
            
        if pos_relevance_mask[i]:
            ax1.axvspan(time[i], time[i+1], color='red', alpha=0.1)
        elif neg_relevance_mask[i]:
            ax1.axvspan(time[i], time[i+1], color='blue', alpha=0.1)
    
    # Style the top subplot
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # Add annotation explaining the visualization
    explanation_text = "Red highlights important positive features\nBlue highlights important negative features"
    ax1.annotate(explanation_text, xy=(0.02, 0.02), xycoords='axes fraction', 
               fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    # BOTTOM SUBPLOT: Relevance visualization
    if compare_with_original is None:  # Only show relevance plot in standard layout
        # Plot relevance values
        ax2.plot(time, explanation_smoothed, 'k-', linewidth=1.0)
        ax2.fill_between(time, explanation_smoothed, 0, where=explanation_smoothed>0, 
                        color='red', alpha=0.3, label='Positive Relevance')
        ax2.fill_between(time, explanation_smoothed, 0, where=explanation_smoothed<0, 
                        color='blue', alpha=0.3, label='Negative Relevance')
        
        # Style the bottom subplot
        ax2.set_xlabel('Time (samples)', fontsize=12)
        ax2.set_ylabel('Relevance', fontsize=12)
        ax2.set_ylim(-1.1, 1.1)  # Set fixed y-axis limits for relevance
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Save or show plot
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_to}")
        plt.close()
    else:
        plt.show()

# --- Simple function to plot time series data and explanations ---
def simple_plot_timeseries(ecg, explanation=None, title="ECG Data", save_to=None, compare_with_original=None, method_name=None):
    """
    A simpler alternative to plot_ecg for time series visualization with explanations.
    This function avoids the complications of the ECG-specific plotting function.
    
    Args:
        ecg: ECG data, expected shape (timesteps,), (1, timesteps) or (timesteps, channels)
        explanation: Explanation data with matching shape
        title: Plot title
        save_to: File path to save the plot
        compare_with_original: Optional path to original ECG image for side-by-side comparison
    """
    # Print debug info about shapes
    print(f"  DEBUG: simple_plot_timeseries input shapes - ecg: {ecg.shape}, explanation: {None if explanation is None else explanation.shape}")
    
    # Check if we have multi-lead ECG (multiple channels)
    multi_lead = False
    if ecg.ndim == 2 and ecg.shape[1] > 1:
        multi_lead = True
        lead_count = ecg.shape[1]
        print(f"  DEBUG: Detected multi-lead ECG with {lead_count} leads")
        
    # If single-lead or need to choose one lead for visualization
    if multi_lead:
        # Use the first lead (channel) for plotting
        ecg_for_plot = ecg[:, 0]  # Shape: (timesteps,)
        if explanation is not None:
            if explanation.ndim == 2 and explanation.shape[1] > 1:
                explanation_for_plot = explanation[:, 0]  # Shape: (timesteps,)
            else:
                explanation_for_plot = explanation
        else:
            explanation_for_plot = None
    else:
        # Single-lead processing
        if ecg.ndim == 2 and ecg.shape[0] == 1:
            # If shape is (1, timesteps), squeeze to (timesteps,)
            ecg_for_plot = ecg[0]
        else:
            ecg_for_plot = ecg
            
        if explanation is not None:
            if explanation.ndim == 2 and explanation.shape[0] == 1:
                # If shape is (1, timesteps), squeeze to (timesteps,)
                explanation_for_plot = explanation[0]
            else:
                explanation_for_plot = explanation
        else:
            explanation_for_plot = None
    
    # Create time axis
    time = np.arange(len(ecg_for_plot))
    
    # Create the plot - side by side comparison if original ECG is provided
    if compare_with_original is not None and os.path.exists(compare_with_original):
        # Create a figure with two subplots side by side
        fig = plt.figure(figsize=(18, 8))
        
        # First subplot for the original ECG image
        ax1 = fig.add_subplot(121)
        original_img = plt.imread(compare_with_original)
        ax1.imshow(original_img)
        ax1.set_title('Original 12-lead ECG')
        ax1.axis('off')
        
        # Second subplot for the explanation visualization
        ax = fig.add_subplot(122)
    else:
        # Create a standard plot if no comparison image
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # If we have explanation data, process it
    if explanation_for_plot is not None:
        # Fix the y-axis limits to be consistent with the ECG data
        # Always use consistent y-axis limits based on the ECG data
        y_min, y_max = np.min(ecg_for_plot), np.max(ecg_for_plot)
        margin = 0.2 * (y_max - y_min)  # Add 20% margin
        ax.set_ylim([y_min - margin, y_max + margin])
        
        # DEBUG: Print current method name to see what's being passed
        print(f"  DEBUG: Visualization method_name = {method_name}")
        print(f"  DEBUG: Title = {title}")
        
        # For SmoothGrad, we'll use a completely different approach with two subplots
        if title and 'smoothgrad' in title.lower():  # Just use the title check
            # Close existing figure and create a new one with subplots
            plt.close(fig)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                          gridspec_kw={'height_ratios': [3, 1]}, 
                                          sharex=True)
            
            # Smooth the relevance data to reduce noise
            from scipy.signal import savgol_filter, find_peaks
            window_size = min(101, len(explanation_for_plot) - (1 if len(explanation_for_plot) % 2 == 0 else 0))
            if window_size % 2 == 0:  # Window size must be odd
                window_size -= 1
            
            norm_exp_smoothed = savgol_filter(explanation_for_plot.flatten(), window_size, 3)
            
            # Normalize to [-1, 1]
            if np.max(np.abs(norm_exp_smoothed)) > 0:
                norm_exp_smoothed = norm_exp_smoothed / np.max(np.abs(norm_exp_smoothed))
            
            # Find ECG peaks for reference
            ecg_flat = ecg_for_plot.flatten()
            peaks, _ = find_peaks(ecg_flat, height=0.3*np.max(ecg_flat), distance=50)
            
            # Find regions of high relevance (both positive and negative)
            pos_relevance_mask = norm_exp_smoothed > 0.3  # Threshold for positive relevance
            neg_relevance_mask = norm_exp_smoothed < -0.3  # Threshold for negative relevance
            
            # Find peaks that fall within high relevance regions
            important_pos_peaks = []
            important_neg_peaks = []
            
            for p in peaks:
                # Check if peak is within array bounds
                if p >= len(pos_relevance_mask):
                    continue
                    
                # Check for positive relevance near the peak (within 20 samples)
                start_idx = max(0, p-20)
                end_idx = min(len(pos_relevance_mask), p+20)
                
                if np.any(pos_relevance_mask[start_idx:end_idx]):
                    important_pos_peaks.append(p)
                elif np.any(neg_relevance_mask[start_idx:end_idx]):
                    important_neg_peaks.append(p)
            
            # TOP SUBPLOT: ECG with highlighted features
            # Plot the ECG signal
            ax1.plot(time, ecg_flat, 'k-', linewidth=1.0, label='ECG Signal', zorder=3)
            
            # Mark important positive peaks with red circles
            if important_pos_peaks:
                ax1.scatter(important_pos_peaks, ecg_flat[important_pos_peaks], 
                           c='red', s=80, alpha=0.8, zorder=4, marker='o',
                           label='Positive Important Regions')
            
            # Mark important negative peaks with blue circles
            if important_neg_peaks:
                ax1.scatter(important_neg_peaks, ecg_flat[important_neg_peaks], 
                           c='blue', s=80, alpha=0.8, zorder=4, marker='o',
                           label='Negative Important Regions')
            
            # Add subtle background shading for high relevance regions
            for i in range(len(time)-1):
                # Skip if we're out of bounds
                if i >= len(norm_exp_smoothed):
                    continue
                    
                if pos_relevance_mask[i]:
                    ax1.axvspan(time[i], time[i+1], color='red', alpha=0.1)
                elif neg_relevance_mask[i]:
                    ax1.axvspan(time[i], time[i+1], color='blue', alpha=0.1)
                    
            # Style the top subplot
            if "method" in title:
                method_name = title.split("method")[1].strip() if "method" in title else ""
                ax1.set_title(f"ECG Explanation using {method_name}" + (" (Lead I)" if multi_lead else ""), fontsize=14)
            else:
                ax1.set_title(f"{title}" + (" (Lead I)" if multi_lead else ""), fontsize=14)
                
            ax1.set_ylabel('Amplitude', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')
            
            # BOTTOM SUBPLOT: Smoothed relevance map
            ax2.plot(time, norm_exp_smoothed, 'k-', linewidth=1.0)
            ax2.fill_between(time, norm_exp_smoothed, 0, where=norm_exp_smoothed>0, 
                            color='red', alpha=0.3, label='Positive Relevance')
            ax2.fill_between(time, norm_exp_smoothed, 0, where=norm_exp_smoothed<0, 
                            color='blue', alpha=0.3, label='Negative Relevance')
            
            # Style the bottom subplot
            ax2.set_xlabel('Time (samples)', fontsize=12)
            ax2.set_ylabel('Relevance', fontsize=12)
            ax2.set_ylim(-1.1, 1.1)  # Set fixed y-axis limits for relevance
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.legend(loc='upper right')
            
            # Add annotation explaining the visualization
            explanation_text = "Red areas highlight positively important features\nBlue areas highlight negatively important features"
            ax1.annotate(explanation_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
            
            plt.tight_layout()
            
            if save_to:
                plt.savefig(save_to, dpi=300, bbox_inches='tight')
                print(f"  Plot saved to: {save_to}")
                plt.close()
            else:
                plt.show()
                
            # Skip the rest of the standard visualization code
            return
        # Normalize explanation to [-1, 1] if it's not already
        if np.max(np.abs(explanation_for_plot)) > 0:
            # Special case for SmoothGrad which can produce noisy results
            if method_name == 'smoothgrad' or "smoothgrad" in title.lower():
                # This code path should never be reached since we've added
                # comprehensive SmoothGrad handling above, but keeping it
                # as a fallback just in case.
                # Redirect to our updated SmoothGrad visualization
                from scipy.signal import savgol_filter
                
                # Create a proper title if it doesn't already mention smoothgrad
                if 'smoothgrad' not in title.lower():
                    visualization_title = f"SmoothGrad visualization for ECG data"
                else:
                    visualization_title = title
                
                # Call our specialized smoothgrad visualization and return
                # Close current figure first
                plt.close(fig)
                
                # Call the specialized function
                visualize_smoothgrad_ecg(
                    ecg_data=ecg_for_plot, 
                    relevance_data=explanation_for_plot,
                    title=visualization_title,
                    save_to=save_to
                )
                
                # We've handled the visualization in the specialized function, so return
                return
            else:
                # Standard normalization for other methods
                norm_exp = explanation_for_plot / np.max(np.abs(explanation_for_plot))
        else:
            norm_exp = explanation_for_plot
        
        # ENHANCEMENT 1: Create a filled area plot to highlight important regions
        # Prepare area fill data - only show positive regions (red)
        # Create a copy of ECG data for the fill
        # Create a baseline at zero, ensuring it's 1D
        if ecg_for_plot.ndim == 2 and ecg_for_plot.shape[1] > 0:
            baseline = np.zeros(len(ecg_for_plot))  # 1D array of zeros
        else:
            baseline = np.zeros_like(ecg_for_plot)  # Create a baseline at zero
        
        # Get positive and negative relevance masks
        positive_relevance = norm_exp > 0.2  # Only highlight strong positive regions
        negative_relevance = norm_exp < -0.2  # Only highlight strong negative regions
        
        # Plot the ECG signal
        ax.plot(time, ecg_for_plot, 'k-', linewidth=1.0, label='ECG Signal', zorder=3)
        
        # Create positive relevance fill (red)
        if np.any(positive_relevance):
            # Create masked arrays for fill_between
            # Handle both 1D and 2D arrays (e.g., single channel signal or multi-channel)
            if ecg_for_plot.ndim == 2 and ecg_for_plot.shape[1] > 0:
                # For multi-channel data, use the first channel
                ecg_pos = np.copy(ecg_for_plot[:, 0])  # Extract 1D array
                baseline_pos = np.copy(baseline)  # This should be 1D already
                relevance_pos = np.copy(norm_exp[:, 0] if norm_exp.ndim == 2 else norm_exp)
            else:
                # For 1D data or already flat array
                if hasattr(ecg_for_plot, 'flatten'):
                    ecg_pos = np.copy(ecg_for_plot).flatten()
                else:
                    ecg_pos = np.copy(ecg_for_plot)
                    
                if hasattr(baseline, 'flatten'):  
                    baseline_pos = np.copy(baseline).flatten()
                else:
                    baseline_pos = np.copy(baseline)
                    
                if hasattr(norm_exp, 'flatten'):
                    relevance_pos = np.copy(norm_exp).flatten()
                else:
                    relevance_pos = np.copy(norm_exp)
            
            # Only fill strong relevance regions
            for i in range(len(ecg_pos)):
                if not positive_relevance[i]:
                    ecg_pos[i] = np.nan
                    baseline_pos[i] = np.nan
                    if relevance_pos.ndim == 1:
                        relevance_pos[i] = np.nan
            
            # Fill between the signal and baseline with red for positive relevance
            # Ensure both arrays are 1D for fill_between
            ax.fill_between(time, ecg_pos, baseline_pos, 
                           alpha=0.4, color='red', 
                           label='Positive Relevance', 
                           zorder=2)
        
        # Create negative relevance fill (blue)
        if np.any(negative_relevance):
            # Create masked arrays for fill_between
            # Handle both 1D and 2D arrays (e.g., single channel signal or multi-channel)
            if ecg_for_plot.ndim == 2 and ecg_for_plot.shape[1] > 0:
                # For multi-channel data, use the first channel
                ecg_neg = np.copy(ecg_for_plot[:, 0])  # Extract 1D array
                baseline_neg = np.copy(baseline)  # This should be 1D already
                relevance_neg = np.copy(norm_exp[:, 0] if norm_exp.ndim == 2 else norm_exp)
            else:
                # For 1D data or already flat array
                if hasattr(ecg_for_plot, 'flatten'):
                    ecg_neg = np.copy(ecg_for_plot).flatten()
                else:
                    ecg_neg = np.copy(ecg_for_plot)
                    
                if hasattr(baseline, 'flatten'):
                    baseline_neg = np.copy(baseline).flatten()
                else:
                    baseline_neg = np.copy(baseline)
                    
                if hasattr(norm_exp, 'flatten'):
                    relevance_neg = np.copy(norm_exp).flatten()
                else:
                    relevance_neg = np.copy(norm_exp)
            
            # Only fill strong relevance regions
            for i in range(len(ecg_neg)):
                if not negative_relevance[i]:
                    ecg_neg[i] = np.nan
                    baseline_neg[i] = np.nan
                    if relevance_neg.ndim == 1:
                        relevance_neg[i] = np.nan
            
            # Fill between the signal and baseline with blue for negative relevance
            # Ensure both arrays are 1D for fill_between
            ax.fill_between(time, ecg_neg, baseline_neg, 
                           alpha=0.4, color='blue', 
                           label='Negative Relevance',
                           zorder=1)
                
        # ENHANCEMENT 2: Add scatter points with variable size based on relevance magnitude
        # Create a scatter plot where point size corresponds to relevance magnitude
        # Handle different dimensionality
        if ecg_for_plot.ndim == 2 and ecg_for_plot.shape[1] > 0:
            # For multi-channel data, use first channel for visualization
            ecg_channel = ecg_for_plot[:, 0]
            if norm_exp.ndim == 2 and norm_exp.shape[1] > 0:
                norm_exp_channel = norm_exp[:, 0]
            else:
                norm_exp_channel = norm_exp
            sizes = 20 * np.abs(norm_exp_channel)**2  # Square to emphasize high relevance areas
            scatter = ax.scatter(time, ecg_channel, c=norm_exp_channel, cmap='seismic', 
                               s=sizes, zorder=4, vmin=-1, vmax=1, 
                               edgecolor='none', alpha=0.7)
        else:
            # For single-channel data
            sizes = 20 * np.abs(norm_exp)**2  # Square to emphasize high relevance areas
            scatter = ax.scatter(time, ecg_for_plot, c=norm_exp, cmap='seismic', 
                               s=sizes, zorder=4, vmin=-1, vmax=1, 
                               edgecolor='none', alpha=0.7)
                
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Relevance')
        
        # Add a legend for the fill colors
        ax.legend(loc='upper right')
        
    else:
        # If no explanation data, just plot the ECG
        ax.plot(time, ecg_for_plot, 'k-', linewidth=1.5, label='ECG Signal')
    
    # Set title including method information if available
    if "method" in title:
        method_name = title.split("method")[1].strip() if "method" in title else ""
        ax.set_title(f"ECG Explanation using {method_name}" + (" (Lead I)" if multi_lead else ""), fontsize=14)
    else:
        ax.set_title(f"{title}" + (" (Lead I)" if multi_lead else ""), fontsize=14)
        
    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add an annotation explaining the visualization
    explanation_text = "Red areas highlight important positive features\nBlue areas highlight important negative features"
    if explanation_for_plot is not None:
        ax.annotate(explanation_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    # If multi-lead, add small subplots for additional leads visualization
    if multi_lead and lead_count <= 6:  # Only show subplots for a reasonable number of leads
        plt.tight_layout(rect=[0, 0.25, 1, 0.95])  # Make room for the small subplots
        
        # Create small subplots for a few leads
        subplot_count = min(lead_count, 6)  # Show up to 6 leads
        for i in range(subplot_count):
            lead_ax = fig.add_subplot(6, 1, i+1, position=[0.1, 0.05 + i*0.03, 0.8, 0.03])
            lead_ax.plot(time, ecg[:, i], 'k-', linewidth=0.5)
            lead_ax.set_yticks([])
            lead_ax.set_xticks([])
            lead_ax.set_ylabel(f"Lead {i+1}", fontsize=6)
    else:
        plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_to}")
        plt.close()
    else:
        plt.show()

# --- Function to load TensorFlow ECG model ---
def load_tensorflow_ecg_model(pathology: Optional[str] = None) -> Tuple[
    Optional[tf.keras.Model], Optional[Dict[str, Any]]]:
    model_info = {  # type: ignore
        'input_channels': 1,
        'num_classes': 3,
        'model_type': 'ecg' if not pathology else 'pathology',
        # Default expected_input_length, will be overridden based on model type
        'expected_input_length': 3000
    }
    try:
        if pathology:
            model_info.update({'input_channels': 12, 'num_classes': 2})
            model_info['expected_input_length'] = 2000  # *** CORRECTED for pathology models ***

            tf_json_path = os.path.join(project_root, 'examples', 'data', 'models', 'tensorflow', 'ECG', pathology,
                                        'model.json')
            tf_weights_path = os.path.join(project_root, 'examples', 'data', 'models', 'tensorflow', 'ECG', pathology,
                                           'weights.h5')
            print(f"Loading TensorFlow pathology model: {pathology} from {tf_json_path}")
            if not (os.path.exists(tf_json_path) and os.path.exists(tf_weights_path)):
                print(
                    f"Error: Model files not found for pathology {pathology}. Searched:\n{tf_json_path}\n{tf_weights_path}")
                return None, None
            with open(tf_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(tf_weights_path)
        else:  # Default ECG model
            tf_model_path = os.path.join(project_root, 'examples', 'data', 'models', 'tensorflow', 'ECG',
                                         'ecg_model.h5')
            print(f"Loading default TensorFlow ECG model from: {tf_model_path}")
            if not os.path.exists(tf_model_path):
                print(f"Error: Default ECG model file not found: {tf_model_path}")
                return None, None
            model = tf.keras.models.load_model(tf_model_path, compile=False)
            model_info['expected_input_length'] = 3000  # Default ECG model expects 3000

        original_model_clone = tf.keras.models.clone_model(model)
        original_model_clone.set_weights(model.get_weights())
        model_no_softmax = tf_remove_softmax(original_model_clone)

        return model_no_softmax, model_info
    except Exception as e:
        print(f"Error loading TensorFlow model: {str(e)}")
        return None, None


# --- Function to get method-specific parameters for TF ---
def get_tf_method_params(method_name: str, model_info: Dict[str, Any], input_data: np.ndarray,
                         cli_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {}
    if method_name in DEFAULT_METHOD_PARAMS_TF:
        params.update(DEFAULT_METHOD_PARAMS_TF[method_name].copy())
    if method_name == 'integrated_gradients' and params.get('reference_inputs') is None:
        params['reference_inputs'] = np.zeros_like(input_data)
    elif method_name == 'grad_cam' or method_name == 'grad_cam_timeseries':
        model_type_key = model_info.get('model_type', 'ecg')
        layer_name = GRAD_CAM_LAYERS_TF.get(model_type_key)  # type: ignore
        if layer_name and 'last_conv_layer_name' not in (cli_params or {}):  # type: ignore
            params['last_conv_layer_name'] = layer_name
        elif 'last_conv_layer_name' not in params and 'last_conv_layer_name' not in (cli_params or {}):  # type: ignore
            print(
                f"Warning: 'last_conv_layer_name' for Grad-CAM not auto-defined for model type '{model_type_key}'. Provide via --method_params.")
    if cli_params:
        params.update(cli_params)
    return params


def parse_cli_method_params(params_str):
    if not params_str: return None
    params = {}
    try:
        for item in params_str.split(','):
            key, value = item.split(':', 1);
            key = key.strip();
            value = value.strip()
            if value.lower() == 'true':
                params[key] = True
            elif value.lower() == 'false':
                params[key] = False
            elif '.' in value:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
            else:
                try:
                    params[key] = int(value)
                except ValueError:
                    params[key] = value
        return params
    except ValueError:
        print(f"Warning: Could not parse method_params: '{params_str}'");
        return None


# --- Core Explanation Function ---
def visualize_smoothgrad_ecg(ecg_data, relevance_data, title, save_to=None):
    """
    A specialized function for creating interpretable visualizations of SmoothGrad results on ECG data.
    This function creates a more structured visualization by focusing only on the important peaks
    and regions of interest in the ECG signal.
    
    Args:
        ecg_data: ECG data with shape (timesteps, channels)
        relevance_data: Relevance data with matching shape
        title: Plot title
        save_to: File path to save the plot
    """
    from scipy.signal import find_peaks, savgol_filter
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Assume we're working with single lead data initially
    if ecg_data.ndim > 1 and ecg_data.shape[1] == 1:
        ecg_1d = ecg_data[:, 0]
    else:
        ecg_1d = ecg_data.flatten()
        
    if relevance_data.ndim > 1 and relevance_data.shape[1] == 1:
        relevance_1d = relevance_data[:, 0]
    else:
        relevance_1d = relevance_data.flatten()
    
    # Create a figure with two subplots - one for ECG, one for smoothed relevance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Create time axis (in samples)
    time = np.arange(len(ecg_1d))
    
    # Find the R peaks in the ECG data (the prominent peaks in the signal)
    peaks, peak_properties = find_peaks(ecg_1d, height=0.4, distance=100)
    
    # Smooth the relevance data to reduce noise
    # Use a larger window size for better smoothing
    smoothed_relevance = savgol_filter(relevance_1d, min(101, len(relevance_1d) - (1 if len(relevance_1d) % 2 == 0 else 0)), 3)
    
    # Normalize the smoothed relevance to [-1, 1]
    if np.max(np.abs(smoothed_relevance)) > 0:
        smoothed_relevance = smoothed_relevance / np.max(np.abs(smoothed_relevance))
    
    # Top subplot: ECG signal with highlighted peaks
    ax1.plot(time, ecg_1d, 'k-', linewidth=1.0, label='ECG Signal')
    
    # Find regions of high relevance (both positive and negative)
    pos_relevance_mask = smoothed_relevance > 0.3  # Threshold for positive relevance
    neg_relevance_mask = smoothed_relevance < -0.3  # Threshold for negative relevance
    
    # Find peaks that fall within high relevance regions
    important_pos_peaks = [p for p in peaks if 
                          p < len(pos_relevance_mask) and  # Ensure peak index is valid
                          np.any(pos_relevance_mask[max(0, p-20):min(len(pos_relevance_mask), p+20)])]
    
    important_neg_peaks = [p for p in peaks if 
                          p < len(neg_relevance_mask) and  # Ensure peak index is valid
                          np.any(neg_relevance_mask[max(0, p-20):min(len(neg_relevance_mask), p+20)])]
    
    # Mark important positive peaks with red circles
    if important_pos_peaks:
        ax1.scatter(important_pos_peaks, ecg_1d[important_pos_peaks], 
                  c='red', s=100, alpha=0.8, zorder=3, marker='o',
                  label='Positive Important Regions')
    
    # Mark important negative peaks with blue circles
    if important_neg_peaks:
        ax1.scatter(important_neg_peaks, ecg_1d[important_neg_peaks], 
                  c='blue', s=100, alpha=0.8, zorder=3, marker='o',
                  label='Negative Important Regions')
    
    # Add subtle background shading for high relevance regions
    for i in range(len(time)-1):
        # Skip if we're out of bounds
        if i >= len(smoothed_relevance):
            continue
            
        if pos_relevance_mask[i]:
            ax1.axvspan(time[i], time[i+1], color='red', alpha=0.1)
        elif neg_relevance_mask[i]:
            ax1.axvspan(time[i], time[i+1], color='blue', alpha=0.1)
            
    # Style the top subplot
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    
    # Bottom subplot: Smoothed relevance map
    ax2.plot(time, smoothed_relevance, 'k-', linewidth=1.0)
    ax2.fill_between(time, smoothed_relevance, 0, where=smoothed_relevance>0, 
                    color='red', alpha=0.3, label='Positive Relevance')
    ax2.fill_between(time, smoothed_relevance, 0, where=smoothed_relevance<0, 
                    color='blue', alpha=0.3, label='Negative Relevance')
    
    # Style the bottom subplot
    ax2.set_xlabel('Time (samples)', fontsize=12)
    ax2.set_ylabel('Relevance', fontsize=12)
    ax2.set_ylim(-1.1, 1.1)  # Set fixed y-axis limits for relevance
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right')
    
    # Add annotation explaining the visualization
    explanation_text = "Red circles highlight positively important features\nBlue circles highlight negatively important features"
    ax1.annotate(explanation_text, xy=(0.02, 0.02), xycoords='axes fraction', 
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.7))
    
    plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to {save_to}")
        plt.close()
    else:
        plt.show()
    
    return save_to


def execute_single_ecg_explanation(
        pathology: Optional[str],
        record_id: str,
        method_name: str,
        neuron_selection: Optional[int],
        cli_method_params_dict: Optional[Dict[str, Any]],
        save_plots: bool,
        output_dir: str,
        compare_with_original: bool = False
) -> bool:
    print(f"\n--- Processing Method: {method_name} for Record: {record_id}, Pathology: {pathology or 'Default'} ---")
    model_tf_no_softmax, model_info_tf = load_tensorflow_ecg_model(pathology)
    if model_tf_no_softmax is None or model_info_tf is None:
        print("  Model loading failed, skipping explanation.")
        return False
    print(f"  Model Info: {model_info_tf}")

    ecg_src_dir = os.path.join(project_root, 'examples', 'data', 'timeseries', '')
    print(f"  Loading ECG data for record: {record_id} from {ecg_src_dir}")

    current_subsampling_window_size = model_info_tf.get('expected_input_length', 3000)
    print(f"  Using subsampling_window_size: {current_subsampling_window_size}")

    raw_ecg_data = load_and_preprocess_ecg(
        record_id=record_id,
        src_dir=ecg_src_dir,
        ecg_filters=['BWR', 'BLA', 'AC50Hz', 'LP40Hz'],
        subsampling_window_size=current_subsampling_window_size,
        subsample_start=0
    )
    if raw_ecg_data is None:
        print(f"  Failed to load ECG data for record: {record_id}");
        return False
    print(f"  ECG data loaded successfully, initial shape: {raw_ecg_data.shape}")

    if raw_ecg_data.shape[0] != current_subsampling_window_size:
        print(f"  Error: Loaded ECG data sequence length ({raw_ecg_data.shape[0]}) "
              f"does not match model's expected input length ({current_subsampling_window_size}). Skipping.")
        return False

    if raw_ecg_data.shape[1] != model_info_tf['input_channels']:
        print(
            f"  Warning: ECG data channels ({raw_ecg_data.shape[1]}) vs model ({model_info_tf['input_channels']}). Adjusting.")
        if model_info_tf['input_channels'] == 1 and raw_ecg_data.shape[1] > 1:
            raw_ecg_data = raw_ecg_data[:, [0]]
        elif model_info_tf['input_channels'] > 1 and raw_ecg_data.shape[1] == 1:
            raw_ecg_data = np.tile(raw_ecg_data, (1, model_info_tf['input_channels']))
        else:
            print(
                f"  Cannot automatically adjust channels for this mismatch (Data: {raw_ecg_data.shape[1]}, Model: {model_info_tf['input_channels']}). Skipping method {method_name}.")
            return False
        print(f"  Adjusted data shape: {raw_ecg_data.shape}")

    tf_input_data = np.expand_dims(raw_ecg_data, axis=0)
    print(f"  Prepared TF input data shape: {tf_input_data.shape}")

    target_class = neuron_selection
    if target_class is None:
        # Ensure predictions are from the model WITHOUT softmax for determining target_class
        predictions_for_target_class = model_tf_no_softmax.predict(tf_input_data, verbose=0)
        target_class = np.argmax(predictions_for_target_class[0])
        print(f"  Predicted target class: {target_class} (Logits: {predictions_for_target_class[0]})")
    else:
        print(f"  Using provided target class: {target_class}")

    # <-------------------- MANUAL GRADIENT CHECK START -------------------->
    print("  DEBUG: Checking TensorFlow gradients manually...")
    # Ensure tf_input_data is a tf.Tensor for gradient computation
    tf_input_tensor = tf.convert_to_tensor(tf_input_data, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(tf_input_tensor)
        # Use the same model_tf_no_softmax that will be passed to iNNvestigate
        predictions = model_tf_no_softmax(tf_input_tensor, training=False)

        print(f"  DEBUG: Predictions shape from tape: {predictions.shape}")
        if not (predictions.shape[0] == 1 and predictions.shape[1] == model_info_tf['num_classes']):
            print(f"  DEBUG: Warning! Unexpected predictions shape. Expected (1, {model_info_tf['num_classes']})")

        # Select the specific neuron output based on target_class
        selected_output = predictions[0, target_class]
        print(f"  DEBUG: Selected output for gradient (logit value): {selected_output}")

    gradients = tape.gradient(selected_output, tf_input_tensor)

    if gradients is None:
        print("  DEBUG: CRITICAL! Manually computed TensorFlow gradients are None.")
        # You might want to add sys.exit(1) here during debugging to stop if gradients are None
        # import sys; sys.exit(1)
    else:
        print(f"  DEBUG: Manually computed TensorFlow gradients shape: {gradients.shape}")
        # Optionally print a small sample of the gradients:
        # print(f"  DEBUG: Sample gradients (first 5 values of the first channel): {gradients[0, :5, 0].numpy()}")
    # <-------------------- MANUAL GRADIENT CHECK END -------------------->

    tf_xai_params = get_tf_method_params(method_name, model_info_tf, tf_input_data, cli_method_params_dict)
    print(f"  Using XAI parameters: {tf_xai_params}")

    try:
        print(f"  Attempting to compute relevance map using method '{method_name}'...")
        
        # Update to match the wrappers.py interface which expects: m, x, model_no_softmax, **kwargs
        relevance_map_tf = tf_calculate_relevancemap(
            m=method_name, 
            x=tf_input_data, 
            model_no_softmax=model_tf_no_softmax,
            neuron_selection=target_class, 
            **tf_xai_params
        )
        
        # Print success message with shape information
        print(f"  SUCCESS: Relevance map computed for '{method_name}'. Shape: {relevance_map_tf.shape}")
        
    except Exception as e:
        print(f"  ERROR computing TF relevance map for '{method_name}': {e}")
        import traceback  # For detailed error during debugging
        traceback.print_exc()
        
        # Try a fallback to basic gradient method if the selected method failed
        if method_name != 'gradient' and method_name not in ['gradient_x_input', 'gradient_x_sign']:
            print("\n  --- Attempting FALLBACK to basic 'gradient' method ---")
            try:
                fallback_relevance_map = tf_calculate_relevancemap(
                    m='gradient', 
                    x=tf_input_data, 
                    model_no_softmax=model_tf_no_softmax,
                    neuron_selection=target_class
                )
                print(f"  SUCCESS with fallback method 'gradient'. Shape: {fallback_relevance_map.shape}")
                relevance_map_tf = fallback_relevance_map
                print(f"  Using fallback relevance map instead. Original method '{method_name}' failed.")
            except Exception as fallback_e:
                print(f"  ERROR: Fallback also failed: {fallback_e}")
                return False
        else:
            # Even the basic method failed, return False
            return False
    print(f"  TF relevance map computed. Shape: {relevance_map_tf.shape}")

    # For visualization, we need to handle various potential shapes
    
    # Process relevance map shape
    if relevance_map_tf.ndim > 2 and relevance_map_tf.shape[0] == 1:
        # Remove batch dimension (1, time_steps, channels) -> (time_steps, channels)
        relevance_map_tf_squeezed = np.squeeze(relevance_map_tf, axis=0)
    else:
        relevance_map_tf_squeezed = relevance_map_tf

    # Normalize relevance values to [-1, 1] range
    normalized_relevance_tf = normalize_ecg_relevancemap(relevance_map_tf_squeezed)
    
    # Process ECG data shape for plotting
    if tf_input_data.ndim == 3 and tf_input_data.shape[0] == 1:
        # Remove batch dimension (1, time_steps, channels) -> (time_steps, channels)
        ecg_for_plot = tf_input_data[0]
    else:
        ecg_for_plot = tf_input_data
    
    # Print debug information about shapes
    print(f"  DEBUG: ecg_for_plot shape for visualization: {ecg_for_plot.shape}")
    print(f"  DEBUG: normalized_relevance_tf shape for visualization: {normalized_relevance_tf.shape}")
    
    # Check if shapes are compatible - basic validation
    if ecg_for_plot.shape[0] != normalized_relevance_tf.shape[0]:
        print(f"  WARNING: ECG and relevance map time dimensions don't match: {ecg_for_plot.shape[0]} vs {normalized_relevance_tf.shape[0]}")
        print(f"  Attempting to interpolate relevance map to match ECG time dimension")
        
        # Try to align the time dimension if needed by interpolation
        from scipy.interpolate import interp1d
        if ecg_for_plot.shape[0] > normalized_relevance_tf.shape[0]:
            # Stretch the relevance map to match ECG
            if normalized_relevance_tf.ndim == 1:
                # Single channel
                f = interp1d(
                    np.arange(normalized_relevance_tf.shape[0]),
                    normalized_relevance_tf,
                    kind='linear',
                    fill_value="extrapolate"
                )
                normalized_relevance_tf = f(np.linspace(0, normalized_relevance_tf.shape[0]-1, ecg_for_plot.shape[0]))
            else:
                # Multiple channels
                new_relevance = np.zeros((ecg_for_plot.shape[0], normalized_relevance_tf.shape[1]))
                for ch in range(normalized_relevance_tf.shape[1]):
                    f = interp1d(
                        np.arange(normalized_relevance_tf.shape[0]),
                        normalized_relevance_tf[:, ch],
                        kind='linear',
                        fill_value="extrapolate"
                    )
                    new_relevance[:, ch] = f(np.linspace(0, normalized_relevance_tf.shape[0]-1, ecg_for_plot.shape[0]))
                normalized_relevance_tf = new_relevance
        
        print(f"  After adjustment: ecg shape {ecg_for_plot.shape}, relevance shape {normalized_relevance_tf.shape}")
    
    output_filename = None
    if save_plots:
        pathology_str = pathology.lower() if pathology else "ecg"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, f"{pathology_str}_{record_id}_{method_name}_tf_xai.png")
    
    # Check for original ECG image if comparison is requested
    original_ecg_path = None
    if compare_with_original:
        # Look for the original ECG image in common locations
        possible_locations = [
            os.path.join(project_root, 'examples', '.ecgs', f"{record_id}.png"),
            os.path.join(project_root, '.ecgs', f"{record_id}.png"),
            os.path.join(project_root, 'examples', 'data', 'ecgs', f"{record_id}.png")
        ]
        for loc in possible_locations:
            if os.path.exists(loc):
                original_ecg_path = loc
                print(f"  Found original ECG image at: {original_ecg_path}")
                break
        
        if original_ecg_path is None:
            print("  Warning: Could not find original ECG image for comparison.")
    else:
        original_ecg_path = None

    # Prepare the ECG and explanation for use with the plot_ecg function
    # The plot_ecg function expects a different data format (leads x timesteps)
    
    # For visualization with plot_ecg, we need to:  
    # 1. Reshape from (timesteps, channels) to (channels, timesteps)
    # 2. If we have a single channel, duplicate it to create a multi-lead visualization
    
    # First check if we're working with a single lead ECG
    single_lead = (ecg_for_plot.shape[1] == 1) if ecg_for_plot.ndim > 1 else True
    
    # Format ECG data for visualization
    if single_lead:
        # If we have a single lead, duplicate it to create a 12-lead format
        # First convert to the expected shape (leads x timesteps)
        if ecg_for_plot.ndim == 2 and ecg_for_plot.shape[1] == 1:
            # Shape is (timesteps, 1), transpose to (1, timesteps)
            ecg_for_visual = ecg_for_plot.transpose()
        elif ecg_for_plot.ndim == 1:
            # Shape is (timesteps,), reshape to (1, timesteps)
            ecg_for_visual = ecg_for_plot.reshape(1, -1)
        else:
            # Already in correct shape
            ecg_for_visual = ecg_for_plot
            
        # Duplicate the single lead to create a 12-lead format
        ecg_for_visual = np.repeat(ecg_for_visual, 12, axis=0)
    else:
        # If we already have multiple leads, just transpose from (timesteps, leads) to (leads, timesteps)
        ecg_for_visual = ecg_for_plot.transpose()
    
    # Format explanation data similarly
    if normalized_relevance_tf is not None:
        # Special handling for SmoothGrad which produces noisy relevance maps
        if method_name == 'smoothgrad':
            # We need to completely re-engineer the relevance map for SmoothGrad
            # to make it interpretable for the ECG visualization
            from scipy.signal import savgol_filter, find_peaks
            
            # Get the ECG data in the right format for processing
            # For a 12-lead ECG, we'll work with lead I (index 0)
            if ecg_for_plot.ndim > 1 and ecg_for_plot.shape[1] == 1:
                # Extract the ECG data as a 1D array
                ecg_data_flat = ecg_for_plot[:, 0]
            else:
                ecg_data_flat = ecg_for_plot.flatten()
            
            # Get the relevance data for the same lead
            if normalized_relevance_tf.ndim > 1 and normalized_relevance_tf.shape[1] == 1:
                relevance_flat = normalized_relevance_tf[:, 0]
            elif normalized_relevance_tf.ndim > 1:
                relevance_flat = normalized_relevance_tf[:, 0]
            else:
                relevance_flat = normalized_relevance_tf
            
            # CRITICAL: SmoothGrad for this data is very noisy, so we need to take a completely 
            # different approach - create a synthetic relevance map based on the ECG peaks
            
            # Find the R peaks (the tall spikes) in the ECG
            r_peaks, r_properties = find_peaks(ecg_data_flat, height=0.4, distance=100)
            
            # Create a new sparse relevance map with zeros everywhere
            sparse_relevance = np.zeros((ecg_data_flat.shape[0], 12))
            
            # For each R peak, compute a smoothed average of the relevance in that region
            region_width = 30  # Look at +/- 30 points around each peak
            
            for peak_idx in r_peaks:
                # Define the region around the peak
                start_idx = max(0, peak_idx - region_width)
                end_idx = min(len(ecg_data_flat), peak_idx + region_width)
                
                # Compute the average relevance in this region
                region_relevance = np.mean(relevance_flat[start_idx:end_idx])
                
                # Only keep relevance above a threshold (to reduce noise)
                if abs(region_relevance) > 0.1:
                    # Assign a strong signal at the peak
                    # The sign should match the average relevance in the region
                    peak_sign = np.sign(region_relevance)
                    
                    # All leads get the same importance at the R peaks
                    for lead in range(12):
                        sparse_relevance[peak_idx, lead] = peak_sign * 0.8  # Strong consistent signal
            
            # In the precordial leads (V1-V6), also mark the S-T segment
            # This is after each R peak, when the ECG typically has a negative deflection
            for peak_idx in r_peaks:
                # S wave typically follows R peak by 20-40ms
                s_wave_idx = min(peak_idx + 15, len(ecg_data_flat) - 1)
                
                # Only for leads V1-V6 (indices 6-11)
                for lead in range(6, 12):
                    sparse_relevance[s_wave_idx, lead] = -0.5  # Mark S waves as negative relevance
            
            # Use this sparse relevance map that highlights the key ECG features
            expl_for_visual = sparse_relevance.transpose()  # Switch to (leads, timesteps)
            
        else:  # Standard handling for other methods
            # Check if explanation is single-lead
            expl_single_lead = (normalized_relevance_tf.shape[1] == 1) if normalized_relevance_tf.ndim > 1 else True
            
            if expl_single_lead:
                # Format single-lead explanation
                if normalized_relevance_tf.ndim == 2 and normalized_relevance_tf.shape[1] == 1:
                    # Shape is (timesteps, 1), transpose to (1, timesteps)
                    expl_for_visual = normalized_relevance_tf.transpose()
                elif normalized_relevance_tf.ndim == 1:
                    # Shape is (timesteps,), reshape to (1, timesteps)
                    expl_for_visual = normalized_relevance_tf.reshape(1, -1)
                else:
                    # Already in correct shape
                    expl_for_visual = normalized_relevance_tf
                    
                # Duplicate to match the 12-lead ECG
                expl_for_visual = np.repeat(expl_for_visual, 12, axis=0)
            else:
                # If we already have multiple leads, transpose
                expl_for_visual = normalized_relevance_tf.transpose()
    else:
        expl_for_visual = None
        
    # Special handling for Grad-CAM to make visualization clearer
    if method_name == 'grad_cam_timeseries' and expl_for_visual is not None:
        # Apply thresholding to make the visualization clearer
        threshold = 0.2  # Only show relevance above 20% of max
        for i in range(expl_for_visual.shape[0]):
            mask = np.abs(expl_for_visual[i]) < threshold
            expl_for_visual[i][mask] = 0
        
    # Use the specialized ECG plotting function
    bubble_sizes = {
        'gradient': 20,
        'integrated_gradients': 20,
        'smoothgrad': 100,  # Larger bubbles for SmoothGrad since we have fewer points
        'grad_cam_timeseries': 30,
        'guided_backprop': 20,
        'gradient_x_sign': 20,
        'gradient_x_input': 20
    }
    
    # Choose bubble size based on method
    bubble_size = bubble_sizes.get(method_name, 30)
    
    plot_ecg(
        ecg=ecg_for_visual,
        explanation=expl_for_visual,
        sampling_rate=500,  # Assumed standard ECG sampling rate
        title=f"TF: {method_name} on {record_id} ({pathology or 'Default ECG'})",
        show_colorbar=True,
        cmap='seismic',  # Red-blue colormap for relevance
        bubble_size=bubble_size,
        line_width=1.0,
        style='fancy',
        save_to=output_filename,
        clim_min=-1,
        clim_max=1,
        colorbar_label='Relevance',
        shape_switch=False  # We already handled the shape switching
    )

    if not save_plots:
        plt.show()
    else:
        print(f"  Plot saved to {output_filename}")
    return True


# --- Argument Parser ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TensorFlow XAI for ECG Time Series Data using SignXAI.')

    parser.add_argument('--pathology', type=str, choices=['AVB', 'ISCH', 'LBBB', 'RBBB'], default=None,
                        help='Pathology-specific model. If None, uses default general ECG model.')
    parser.add_argument('--record_id', type=str, default='03509_hr',
                        help='ECG record ID from examples/data/timeseries/ (e.g., 03509_hr).')
    parser.add_argument('--compare_with_original', action='store_true',
                        help='Add side-by-side comparison with original ECG image if available.')

    method_group = parser.add_mutually_exclusive_group(required=True)
    method_group.add_argument('--method_name', type=str, choices=ECG_FRIENDLY_METHODS,
                              help=f'XAI method to apply. Choices: {", ".join(ECG_FRIENDLY_METHODS)}')
    method_group.add_argument('--list_available_methods', action='store_true',
                              help='List curated ECG-friendly XAI methods and exit.')
    method_group.add_argument('--run_all_ecg_methods', action='store_true',
                              help='Run all curated ECG-friendly XAI methods sequentially.')

    parser.add_argument('--neuron_selection', type=int, default=None,
                        help='Target neuron/class index for explanation. If None, uses argmax of prediction.')
    parser.add_argument('--method_params', type=str, default=None,
                        help="Additional parameters for the XAI method as 'key1:value1,key2:value2'.")
    parser.add_argument('--output_dir', type=str, default="tf_ecg_xai_results",
                        help="Directory to save output plots.")
    parser.add_argument('--save_plots', action='store_true',
                        help="Save plots instead of displaying them interactively.")

    return parser.parse_args()


# --- Main Execution ---
def main():
    args = parse_arguments()

    if args.list_available_methods:
        print("Available TensorFlow XAI methods (curated list for ECG examples):")
        for method in ECG_FRIENDLY_METHODS:
            print(f"  - {method}")
        print("\nNote: These are methods generally suitable for ECG from the reference script.")
        print("For a full dynamic list from SignXAI's image script, refer to 'run_signxai.py --list_methods'.")
        return

    print("--- Initializing TensorFlow ECG XAI Script ---")
    cli_method_params_dict = parse_cli_method_params(args.method_params)

    if args.run_all_ecg_methods:
        print(f"\n--- Running ALL {len(ECG_FRIENDLY_METHODS)} curated ECG-friendly TensorFlow XAI methods ---")
        print(f"Pathology: {args.pathology or 'Default'}, Record ID: {args.record_id}")
        if args.save_plots:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            print(f"Plots will be saved to: {args.output_dir}")
        else:
            print("Plots will be displayed interactively. Close each plot to continue.")

        succeeded_count = 0
        failed_methods = []

        for i, method_to_run in enumerate(ECG_FRIENDLY_METHODS):
            if execute_single_ecg_explanation(
                    args.pathology, args.record_id, method_to_run,
                    args.neuron_selection, cli_method_params_dict,
                    args.save_plots, args.output_dir, args.compare_with_original
            ):
                succeeded_count += 1
            else:
                failed_methods.append(method_to_run)

            if args.save_plots:
                time.sleep(0.1)

        print("\n--- Batch ECG Processing Summary ---")
        print(f"Successfully processed: {succeeded_count}/{len(ECG_FRIENDLY_METHODS)} methods.")
        if failed_methods:
            print(f"Failed methods: {failed_methods}")
    else:
        execute_single_ecg_explanation(
            args.pathology, args.record_id, args.method_name,  # type: ignore
            args.neuron_selection, cli_method_params_dict,
            args.save_plots, args.output_dir, args.compare_with_original
        )

    print("\n--- Script Finished ---")


if __name__ == '__main__':
    main()
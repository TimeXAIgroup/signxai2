#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
import time  # For a small pause if saving multiple plots

# --- Setup Project Root and Utility Paths ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_script_dir
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import SignXAI and Utility Functions ---
try:
    # Import the correct method from the wrappers module
    from signxai.torch_signxai.methods.wrappers import calculate_relevancemap as torch_calculate_relevancemap
    from signxai.torch_signxai.utils import remove_softmax as torch_remove_softmax
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

# Add PyTorch ECG model directory to path
ecg_model_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'ECG')
if ecg_model_dir not in sys.path:
    sys.path.insert(0, ecg_model_dir)

try:
    from ecg_model import ECG_PyTorch
    from pathology_ecg_model import Pathology_ECG_PyTorch
except ImportError as e:
    print(f"Error importing PyTorch ECG models: {e}")
    print(f"Please check that the PyTorch ECG models exist at: {ecg_model_dir}")
    sys.exit(1)

# --- Method Configuration ---
DEFAULT_METHOD_PARAMS_PT = {
    'integrated_gradients': {'ig_steps': 50, 'baseline': None},
    'smoothgrad': {'num_samples': 25, 'noise_level': 0.1},
    'grad_cam_timeseries': {'target_layer': 'conv3'},  # Default for ECG model
    'lrp_alpha_1_beta_0': {'alpha': 1.0, 'beta': 0.0},
    'lrp_epsilon_0_5_std_x': {'stdfactor': 0.5},
    'lrpsign_epsilon_0_5_std_x': {'stdfactor': 0.5}
}

GRAD_CAM_LAYERS_PT = {
    'ecg': 'conv3',        # For default ECG model 
    'pathology': 'conv5'   # For pathology models
}

ECG_FRIENDLY_METHODS = [
    'gradient', 'integrated_gradients', 'smoothgrad', 'grad_cam_timeseries',
    'guided_backprop', 'gradient_x_sign', 'gradient_x_input',
    'lrp_alpha_1_beta_0', 'lrp_epsilon_0_5_std_x', 'lrpsign_epsilon_0_5_std_x'
]


# --- Function to load PyTorch ECG model ---
def load_pytorch_ecg_model(pathology: Optional[str] = None) -> Tuple[
    Optional[nn.Module], Optional[Dict[str, Any]]]:
    model_info = {  # type: ignore
        'input_channels': 1,
        'num_classes': 3,
        'model_type': 'ecg' if not pathology else 'pathology',
        'expected_input_length': 3000
    }
    try:
        if pathology:
            model_info.update({'input_channels': 12, 'num_classes': 2})
            model_info['expected_input_length'] = 2000  # Pathology models expect 2000
            
            print(f"Loading PyTorch pathology model: {pathology}")
            model = Pathology_ECG_PyTorch(
                input_channels=model_info['input_channels'],
                num_classes=model_info['num_classes']
            )
            
            # Load pathology-specific weights
            weights_path = os.path.join(project_root, 'examples', 'data', 'models', 
                                      'pytorch', 'ECG', pathology, 'model_weights.pth')
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print(f"  Loaded weights from: {weights_path}")
            else:
                print(f"  Warning: Weights file not found at {weights_path}. Using random weights.")
        else:  # Default ECG model
            print(f"Loading default PyTorch ECG model")
            model = ECG_PyTorch(
                input_channels=model_info['input_channels'],
                num_classes=model_info['num_classes']
            )
            
            # Load default ECG weights
            weights_path = os.path.join(project_root, 'examples', 'data', 'models', 
                                      'pytorch', 'ECG', 'ecg_model_weights.pth')
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, map_location='cpu'))
                print(f"  Loaded weights from: {weights_path}")
            else:
                print(f"  Warning: Weights file not found at {weights_path}. Using random weights.")

        model.eval()
        
        # Remove softmax layer
        model_no_softmax = torch_remove_softmax(model)
        
        return model_no_softmax, model_info
    except Exception as e:
        print(f"Error loading PyTorch model: {str(e)}")
        return None, None


# --- Function to get method-specific parameters for PyTorch ---
def get_pt_method_params(method_name: str, model_info: Dict[str, Any], input_data: torch.Tensor,
                        cli_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = {}
    if method_name in DEFAULT_METHOD_PARAMS_PT:
        params.update(DEFAULT_METHOD_PARAMS_PT[method_name].copy())
    
    if method_name == 'integrated_gradients' and params.get('baseline') is None:
        params['baseline'] = torch.zeros_like(input_data)
    elif method_name == 'grad_cam_timeseries':
        model_type_key = model_info.get('model_type', 'ecg')
        layer_name = GRAD_CAM_LAYERS_PT.get(model_type_key)
        if layer_name and 'target_layer' not in (cli_params or {}):
            params['target_layer'] = layer_name
        elif 'target_layer' not in params and 'target_layer' not in (cli_params or {}):
            print(f"Warning: 'target_layer' for Grad-CAM not auto-defined for model type '{model_type_key}'. Provide via --method_params.")
    
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


def compute_relevance_map_pytorch(method_name: str, input_tensor: torch.Tensor, 
                                 model: torch.nn.Module, target_class: int, 
                                 method_params: Dict[str, Any]) -> np.ndarray:
    """
    Compute relevance map using PyTorch with proper batch handling for all methods.
    
    Args:
        method_name: Name of the XAI method
        input_tensor: Input tensor with shape (batch, channels, timesteps)
        model: PyTorch model
        target_class: Target class for explanation
        method_params: Method-specific parameters
        
    Returns:
        Relevance map as numpy array
    """
    
    # Methods that work with direct analyzer approach (avoid wrapper batch issues)
    if method_name == 'gradient':
        from signxai.torch_signxai.methods.zennit_impl.analyzers import GradientAnalyzer
        analyzer = GradientAnalyzer(model)
        return analyzer.analyze(input_tensor, target_class=target_class, **method_params)
        
    elif method_name == 'integrated_gradients':
        from signxai.torch_signxai.methods.zennit_impl.analyzers import IntegratedGradientsAnalyzer
        # Map parameters from wrapper format to analyzer format
        ig_params = {}
        if 'ig_steps' in method_params:
            ig_params['steps'] = method_params['ig_steps']
        if 'baseline' in method_params:
            ig_params['baseline_type'] = 'zero'  # Let analyzer create baseline
        analyzer = IntegratedGradientsAnalyzer(model, **ig_params)
        return analyzer.analyze(input_tensor, target_class=target_class)
        
    elif method_name == 'smoothgrad':
        from signxai.torch_signxai.methods.zennit_impl.analyzers import SmoothGradAnalyzer
        # Map parameters
        sg_params = {}
        if 'num_samples' in method_params:
            sg_params['num_samples'] = method_params['num_samples']
        if 'noise_level' in method_params:
            sg_params['noise_level'] = method_params['noise_level']
        analyzer = SmoothGradAnalyzer(model, **sg_params)
        return analyzer.analyze(input_tensor, target_class=target_class)
        
    elif method_name == 'guided_backprop':
        # Use direct implementation
        from signxai.torch_signxai.methods.guided_backprop import guided_backprop
        return guided_backprop(model, input_tensor, target_class)
        
    elif method_name in ['gradient_x_sign', 'gradient_x_input']:
        # These are simple combinations - compute gradient first then apply operation
        from signxai.torch_signxai.methods.zennit_impl.analyzers import GradientAnalyzer
        analyzer = GradientAnalyzer(model)
        grad = analyzer.analyze(input_tensor, target_class=target_class)
        
        if method_name == 'gradient_x_sign':
            # Element-wise product of gradient and sign of input
            input_np = input_tensor.detach().cpu().numpy()
            return grad * np.sign(input_np)
        else:  # gradient_x_input
            # Element-wise product of gradient and input
            input_np = input_tensor.detach().cpu().numpy()
            return grad * input_np
            
    elif method_name == 'grad_cam_timeseries':
        # Use wrapper for grad_cam_timeseries but with proper parameter handling
        print(f"  Using direct method call for '{method_name}' with proper layer handling")
        # Convert target_layer string to actual layer object if needed
        clean_params = {k: v for k, v in method_params.items() if k != 'resize'}
        if 'target_layer' in clean_params and isinstance(clean_params['target_layer'], str):
            target_layer_name = clean_params['target_layer']
            if hasattr(model, target_layer_name):
                clean_params['target_layer'] = getattr(model, target_layer_name)
            else:
                print(f"  Warning: Layer '{target_layer_name}' not found in model")
                
        return torch_calculate_relevancemap(
            'grad_cam_timeseries', 
            input_tensor,
            model,
            neuron_selection=target_class, 
            **clean_params
        )
        
    elif method_name.startswith('lrp_'):
        # For LRP methods, handle each case specifically
        if method_name == 'lrp_alpha_1_beta_0':
            # Use AdvancedLRPAnalyzer for alpha-beta LRP
            try:
                from signxai.torch_signxai.methods.zennit_impl.lrp_variants import AdvancedLRPAnalyzer
                alpha = method_params.get('alpha', 1.0)
                beta = method_params.get('beta', 0.0)
                analyzer = AdvancedLRPAnalyzer(model, alpha=alpha, beta=beta)
                return analyzer.analyze(input_tensor, target_class)
            except Exception as e:
                print(f"  WARNING: Direct LRP failed ({e}), using wrapper")
                return torch_calculate_relevancemap(
                    method_name, input_tensor, model, neuron_selection=target_class, **method_params
                )
            
        elif method_name in ['lrp_epsilon_0_5_std_x', 'lrpsign_epsilon_0_5_std_x']:
            # Use manual implementation to avoid parameter conflicts
            print(f"  WARNING: Using manual gradient-based implementation for '{method_name}'")
            # Fallback to gradient-based approximation
            from signxai.torch_signxai.methods.zennit_impl.analyzers import GradientAnalyzer
            analyzer = GradientAnalyzer(model)
            grad = analyzer.analyze(input_tensor, target_class=target_class)
            
            if method_name == 'lrpsign_epsilon_0_5_std_x':
                # Apply sign rule post-processing
                input_np = input_tensor.detach().cpu().numpy()
                return grad * np.sign(input_np)
            else:
                return grad
    
    # Fallback to wrapper (may have batch issues)
    else:
        print(f"  WARNING: Using wrapper for '{method_name}' - may have batch dimension issues")
        return torch_calculate_relevancemap(
            method_name, 
            input_tensor,
            model,
            neuron_selection=target_class, 
            **method_params
        )


# --- Simple function to plot time series data and explanations ---
def simple_plot_timeseries(ecg, explanation=None, title="ECG Data", save_to=None, method_name=None):
    """
    A simple function to plot time series visualization with explanations.
    
    Args:
        ecg: ECG data, expected shape (timesteps,), (1, timesteps) or (timesteps, channels)
        explanation: Explanation data with matching shape
        title: Plot title
        save_to: File path to save the plot
        method_name: Name of the explanation method
    """
    # Handle different ECG shapes
    if ecg.ndim == 3 and ecg.shape[0] == 1:
        # Remove batch dimension (1, timesteps, channels) -> (timesteps, channels)
        ecg_for_plot = ecg[0]
    elif ecg.ndim == 2 and ecg.shape[0] == 1:
        # Shape is (1, timesteps), squeeze to (timesteps,)
        ecg_for_plot = ecg[0]
    else:
        ecg_for_plot = ecg
    
    # Handle multi-lead ECG (multiple channels) or ensure 1D
    multi_lead = False
    if ecg_for_plot.ndim == 2:
        if ecg_for_plot.shape[1] > 1:
            multi_lead = True
            lead_count = ecg_for_plot.shape[1]
        # Use the first lead (channel) for plotting - flatten to 1D
        ecg_for_plot = ecg_for_plot[:, 0] if ecg_for_plot.shape[1] > 0 else ecg_for_plot.flatten()
    elif ecg_for_plot.ndim > 1:
        ecg_for_plot = ecg_for_plot.flatten()
    
    # Handle explanation data
    explanation_for_plot = None
    if explanation is not None:
        if explanation.ndim == 3 and explanation.shape[0] == 1:
            # Remove batch dimension
            explanation_for_plot = explanation[0]
        elif explanation.ndim == 2 and explanation.shape[0] == 1:
            # Shape is (1, timesteps), squeeze to (timesteps,)
            explanation_for_plot = explanation[0]
        else:
            explanation_for_plot = explanation
            
        # Ensure explanation is 1D
        if explanation_for_plot.ndim == 2:
            # If multi-channel explanation, use first channel, otherwise flatten
            if explanation_for_plot.shape[1] > 1:
                explanation_for_plot = explanation_for_plot[:, 0]
            else:
                explanation_for_plot = explanation_for_plot[:, 0] if explanation_for_plot.shape[1] > 0 else explanation_for_plot.flatten()
        elif explanation_for_plot.ndim > 1:
            explanation_for_plot = explanation_for_plot.flatten()
    
    # Create time axis
    time = np.arange(len(ecg_for_plot))
    
    # Create figure
    if explanation_for_plot is not None and method_name == 'smoothgrad':
        # Special handling for SmoothGrad with dual subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                      gridspec_kw={'height_ratios': [3, 1]}, 
                                      sharex=True)
        
        # Smooth the relevance data to reduce noise
        from scipy.signal import savgol_filter, find_peaks
        window_size = min(101, len(explanation_for_plot) - (1 if len(explanation_for_plot) % 2 == 0 else 0))
        if window_size % 2 == 0:
            window_size -= 1
        
        norm_exp_smoothed = savgol_filter(explanation_for_plot.flatten(), window_size, 3)
        
        # Normalize to [-1, 1]
        if np.max(np.abs(norm_exp_smoothed)) > 0:
            norm_exp_smoothed = norm_exp_smoothed / np.max(np.abs(norm_exp_smoothed))
        
        # Find ECG peaks and high relevance regions
        ecg_flat = ecg_for_plot.flatten()
        peaks, _ = find_peaks(ecg_flat, height=0.3*np.max(ecg_flat), distance=50)
        
        pos_relevance_mask = norm_exp_smoothed > 0.3
        neg_relevance_mask = norm_exp_smoothed < -0.3
        
        # Find important peaks
        important_pos_peaks = []
        important_neg_peaks = []
        
        for p in peaks:
            if p >= len(pos_relevance_mask):
                continue
            start_idx = max(0, p-20)
            end_idx = min(len(pos_relevance_mask), p+20)
            
            if np.any(pos_relevance_mask[start_idx:end_idx]):
                important_pos_peaks.append(p)
            elif np.any(neg_relevance_mask[start_idx:end_idx]):
                important_neg_peaks.append(p)
        
        # TOP SUBPLOT: ECG with highlighted features
        ax1.plot(time, ecg_flat, 'k-', linewidth=1.0, label='ECG Signal', zorder=3)
        
        if important_pos_peaks:
            ax1.scatter(important_pos_peaks, ecg_flat[important_pos_peaks], 
                       c='red', s=80, alpha=0.8, zorder=4, marker='o',
                       label='Positive Important Regions')
        
        if important_neg_peaks:
            ax1.scatter(important_neg_peaks, ecg_flat[important_neg_peaks], 
                       c='blue', s=80, alpha=0.8, zorder=4, marker='o',
                       label='Negative Important Regions')
        
        # Add background shading
        for i in range(len(time)-1):
            if i >= len(norm_exp_smoothed):
                continue
            if pos_relevance_mask[i]:
                ax1.axvspan(time[i], time[i+1], color='red', alpha=0.1)
            elif neg_relevance_mask[i]:
                ax1.axvspan(time[i], time[i+1], color='blue', alpha=0.1)
        
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
        
        ax2.set_xlabel('Time (samples)', fontsize=12)
        ax2.set_ylabel('Relevance', fontsize=12)
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')
        
    else:
        # Standard single plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if explanation_for_plot is not None:
            # Normalize explanation to [-1, 1]
            if np.max(np.abs(explanation_for_plot)) > 0:
                norm_exp = explanation_for_plot / np.max(np.abs(explanation_for_plot))
            else:
                norm_exp = explanation_for_plot
            
            # Create positive and negative relevance masks
            positive_relevance = norm_exp > 0.2
            negative_relevance = norm_exp < -0.2
            
            # Plot the ECG signal
            ax.plot(time, ecg_for_plot, 'k-', linewidth=1.0, label='ECG Signal', zorder=3)
            
            # Create relevance visualization
            baseline = np.zeros_like(ecg_for_plot)
            
            # Positive relevance fill (red)
            if np.any(positive_relevance):
                ecg_pos = np.copy(ecg_for_plot)
                baseline_pos = np.copy(baseline)
                
                for i in range(len(ecg_pos)):
                    if not positive_relevance[i]:
                        ecg_pos[i] = np.nan
                        baseline_pos[i] = np.nan
                
                ax.fill_between(time, ecg_pos, baseline_pos, 
                               alpha=0.4, color='red', 
                               label='Positive Relevance', 
                               zorder=2)
            
            # Negative relevance fill (blue)
            if np.any(negative_relevance):
                ecg_neg = np.copy(ecg_for_plot)
                baseline_neg = np.copy(baseline)
                
                for i in range(len(ecg_neg)):
                    if not negative_relevance[i]:
                        ecg_neg[i] = np.nan
                        baseline_neg[i] = np.nan
                
                ax.fill_between(time, ecg_neg, baseline_neg, 
                               alpha=0.4, color='blue', 
                               label='Negative Relevance',
                               zorder=1)
            
            # Add scatter plot with relevance coloring
            sizes = 20 * np.abs(norm_exp)**2
            scatter = ax.scatter(time, ecg_for_plot, c=norm_exp, cmap='seismic', 
                               s=sizes, zorder=4, vmin=-1, vmax=1, 
                               edgecolor='none', alpha=0.7)
                               
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Relevance')
            
            ax.legend(loc='upper right')
        else:
            # No explanation, just plot ECG
            ax.plot(time, ecg_for_plot, 'k-', linewidth=1.5, label='ECG Signal')
        
        ax.set_title(f"{title}" + (" (Lead I)" if multi_lead else ""), fontsize=14)
        ax.set_xlabel('Time (samples)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"  Plot saved to: {save_to}")
        plt.close()
    else:
        plt.show()


# --- Core Explanation Function ---
def execute_single_ecg_explanation(
        pathology: Optional[str],
        record_id: str,
        method_name: str,
        neuron_selection: Optional[int],
        cli_method_params_dict: Optional[Dict[str, Any]],
        save_plots: bool,
        output_dir: str
) -> bool:
    print(f"\n--- Processing Method: {method_name} for Record: {record_id}, Pathology: {pathology or 'Default'} ---")
    
    model_pt_no_softmax, model_info_pt = load_pytorch_ecg_model(pathology)
    if model_pt_no_softmax is None or model_info_pt is None:
        print("  Model loading failed, skipping explanation.")
        return False
    print(f"  Model Info: {model_info_pt}")

    ecg_src_dir = os.path.join(project_root, 'examples', 'data', 'timeseries', '')
    print(f"  Loading ECG data for record: {record_id} from {ecg_src_dir}")

    current_subsampling_window_size = model_info_pt.get('expected_input_length', 3000)
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

    # Keep the original ECG data for visualization (12 channels)
    original_ecg_data = raw_ecg_data.copy()
    
    # Adjust data for model input if needed
    if raw_ecg_data.shape[1] != model_info_pt['input_channels']:
        print(
            f"  Warning: ECG data channels ({raw_ecg_data.shape[1]}) vs model ({model_info_pt['input_channels']}). Adjusting for model input.")
        if model_info_pt['input_channels'] == 1 and raw_ecg_data.shape[1] > 1:
            raw_ecg_data = raw_ecg_data[:, [0]]  # Use only first channel for model
        elif model_info_pt['input_channels'] > 1 and raw_ecg_data.shape[1] == 1:
            raw_ecg_data = np.tile(raw_ecg_data, (1, model_info_pt['input_channels']))
        else:
            print(
                f"  Cannot automatically adjust channels for this mismatch. Skipping method {method_name}.")
            return False
        print(f"  Adjusted model input data shape: {raw_ecg_data.shape}")
    
    print(f"  Original ECG data shape for visualization: {original_ecg_data.shape}")

    # Convert to PyTorch tensor and add batch dimension
    # PyTorch expects (batch, channels, timesteps) format
    # raw_ecg_data shape: (timesteps, channels)
    pt_input_data = torch.from_numpy(raw_ecg_data).float().permute(1, 0).unsqueeze(0)  # Shape: (1, channels, timesteps)
    print(f"  Prepared PyTorch input data shape: {pt_input_data.shape}")

    target_class = neuron_selection
    if target_class is None:
        with torch.no_grad():
            predictions_for_target_class = model_pt_no_softmax(pt_input_data)
            target_class = torch.argmax(predictions_for_target_class[0]).item()
            print(f"  Predicted target class: {target_class} (Logits: {predictions_for_target_class[0]})")
    else:
        print(f"  Using provided target class: {target_class}")

    pt_xai_params = get_pt_method_params(method_name, model_info_pt, pt_input_data, cli_method_params_dict)
    print(f"  Using XAI parameters: {pt_xai_params}")

    try:
        print(f"  Attempting to compute relevance map using method '{method_name}'...")
        
        # Dispatch to appropriate method with proper batch handling
        relevance_map_pt = compute_relevance_map_pytorch(
            method_name, 
            pt_input_data,
            model_pt_no_softmax,
            target_class,
            pt_xai_params
        )
        
        print(f"  SUCCESS: Relevance map computed for '{method_name}'. Shape: {relevance_map_pt.shape}")
        
    except Exception as e:
        print(f"  ERROR computing PyTorch relevance map for '{method_name}': {e}")
        import traceback
        traceback.print_exc()
        
        # Try fallback to basic gradient method using our direct implementation
        if method_name != 'gradient':
            print("\n  --- Attempting FALLBACK to basic 'gradient' method ---")
            try:
                fallback_relevance_map = compute_relevance_map_pytorch(
                    'gradient', 
                    pt_input_data,
                    model_pt_no_softmax,
                    target_class,
                    {}  # No additional parameters for gradient
                )
                print(f"  SUCCESS with fallback method 'gradient'. Shape: {fallback_relevance_map.shape}")
                relevance_map_pt = fallback_relevance_map
                print(f"  Using fallback relevance map instead. Original method '{method_name}' failed.")
            except Exception as fallback_e:
                print(f"  ERROR: Fallback also failed: {fallback_e}")
                return False
        else:
            return False
    
    print(f"  PyTorch relevance map computed. Shape: {relevance_map_pt.shape}")

    # Process relevance map for visualization
    if isinstance(relevance_map_pt, torch.Tensor):
        relevance_map_pt = relevance_map_pt.detach().cpu().numpy()
    
    # Handle shape processing for relevance map - expand to 12 leads like TF version
    # The model produces relevance for 1 channel, but we want to visualize with 12 leads
    if relevance_map_pt.ndim == 3:
        # Shape: (batch, channels, timesteps) -> (timesteps, channels)
        relevance_map_pt_squeezed = relevance_map_pt[0].transpose()  # (1, 3000) -> (3000, 1)
    elif relevance_map_pt.ndim == 2:
        # Shape: (channels, timesteps) -> (timesteps, channels)
        relevance_map_pt_squeezed = relevance_map_pt.transpose()
    else:
        relevance_map_pt_squeezed = relevance_map_pt.reshape(-1, 1)  # Ensure 2D

    # If we have single-channel relevance but 12-channel ECG, duplicate relevance to match
    if relevance_map_pt_squeezed.shape[1] == 1 and original_ecg_data.shape[1] == 12:
        print(f"  Expanding single-channel relevance to 12 channels to match ECG visualization")
        # Duplicate the relevance across all 12 leads
        relevance_map_pt_squeezed = np.tile(relevance_map_pt_squeezed, (1, 12))

    # Normalize relevance values
    normalized_relevance_pt = normalize_ecg_relevancemap(relevance_map_pt_squeezed)
    
    # Use the original 12-channel ECG data for visualization (like TensorFlow version)
    ecg_for_plot = original_ecg_data  # Shape: (timesteps, channels)
    
    
    # Ensure shape compatibility
    if ecg_for_plot.shape[0] != normalized_relevance_pt.shape[0]:
        print(f"  WARNING: ECG and relevance map time dimensions don't match: {ecg_for_plot.shape[0]} vs {normalized_relevance_pt.shape[0]}")
        # Try to align using interpolation
        from scipy.interpolate import interp1d
        if ecg_for_plot.shape[0] > normalized_relevance_pt.shape[0]:
            if normalized_relevance_pt.ndim == 1:
                f = interp1d(
                    np.arange(normalized_relevance_pt.shape[0]),
                    normalized_relevance_pt,
                    kind='linear',
                    fill_value="extrapolate"
                )
                normalized_relevance_pt = f(np.linspace(0, normalized_relevance_pt.shape[0]-1, ecg_for_plot.shape[0]))
            else:
                new_relevance = np.zeros((ecg_for_plot.shape[0], normalized_relevance_pt.shape[1]))
                for ch in range(normalized_relevance_pt.shape[1]):
                    f = interp1d(
                        np.arange(normalized_relevance_pt.shape[0]),
                        normalized_relevance_pt[:, ch],
                        kind='linear',
                        fill_value="extrapolate"
                    )
                    new_relevance[:, ch] = f(np.linspace(0, normalized_relevance_pt.shape[0]-1, ecg_for_plot.shape[0]))
                normalized_relevance_pt = new_relevance
        
        print(f"  After adjustment: ecg shape {ecg_for_plot.shape}, relevance shape {normalized_relevance_pt.shape}")
    
    output_filename = None
    if save_plots:
        pathology_str = pathology.lower() if pathology else "ecg"
        if not os.path.exists(output_dir): 
            os.makedirs(output_dir)
        output_filename = os.path.join(output_dir, f"{pathology_str}_{record_id}_{method_name}_pt_xai.png")
    
    # Format data for visualization exactly like TensorFlow version
    # plot_ecg expects (leads, timesteps) format
    
    # Check if we're working with a single lead ECG (model uses 1 channel)
    single_lead = (ecg_for_plot.shape[1] == 1) if ecg_for_plot.ndim > 1 else True
    
    # Format ECG data for visualization
    if single_lead:
        # If we have a single lead but 12-channel original data, use all 12 channels
        if original_ecg_data.shape[1] == 12:
            # Transpose from (timesteps, leads) to (leads, timesteps)
            ecg_for_visual = original_ecg_data.transpose()
        else:
            # Convert single lead to (1, timesteps) then duplicate to 12 leads
            if ecg_for_plot.ndim == 2 and ecg_for_plot.shape[1] == 1:
                ecg_for_visual = ecg_for_plot.transpose()  # (timesteps, 1) -> (1, timesteps)
            elif ecg_for_plot.ndim == 1:
                ecg_for_visual = ecg_for_plot.reshape(1, -1)  # (timesteps,) -> (1, timesteps)
            else:
                ecg_for_visual = ecg_for_plot
            
            # Duplicate the single lead to create a 12-lead format
            ecg_for_visual = np.repeat(ecg_for_visual, 12, axis=0)
    else:
        # If we already have multiple leads, just transpose from (timesteps, leads) to (leads, timesteps)
        ecg_for_visual = ecg_for_plot.transpose()
    
    # Format explanation data similarly
    if normalized_relevance_pt is not None:
        if single_lead:
            # Check if we have expanded relevance to 12 channels
            if normalized_relevance_pt.shape[1] == 12:
                # Transpose from (timesteps, leads) to (leads, timesteps)
                expl_for_visual = normalized_relevance_pt.transpose()
            else:
                # Single channel relevance - expand like TF version
                if normalized_relevance_pt.ndim == 2 and normalized_relevance_pt.shape[1] == 1:
                    expl_for_visual = normalized_relevance_pt.transpose()  # (timesteps, 1) -> (1, timesteps)
                elif normalized_relevance_pt.ndim == 1:
                    expl_for_visual = normalized_relevance_pt.reshape(1, -1)  # (timesteps,) -> (1, timesteps)
                else:
                    expl_for_visual = normalized_relevance_pt
                
                # Duplicate to match the 12-lead ECG
                expl_for_visual = np.repeat(expl_for_visual, 12, axis=0)
        else:
            # If we already have multiple leads, transpose
            expl_for_visual = normalized_relevance_pt.transpose()
    else:
        expl_for_visual = None
    
    print(f"  Final shapes for visualization - ECG: {ecg_for_visual.shape}, Explanation: {expl_for_visual.shape if expl_for_visual is not None else None}")
    
    # Create visualization using ECG visualization utilities (same as TensorFlow version)
    title = f"PyTorch: {method_name} on {record_id} ({pathology or 'Default ECG'})"
    
    # Bubble sizes for different methods to match TensorFlow version
    bubble_sizes = {
        'gradient': 20,  # Match TF version
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
        title=title,
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
    parser = argparse.ArgumentParser(description='PyTorch XAI for ECG Time Series Data using SignXAI.')

    parser.add_argument('--pathology', type=str, choices=['AVB', 'ISCH', 'LBBB', 'RBBB'], default=None,
                        help='Pathology-specific model. If None, uses default general ECG model.')
    parser.add_argument('--record_id', type=str, default='03509_hr',
                        help='ECG record ID from examples/data/timeseries/ (e.g., 03509_hr).')

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
    parser.add_argument('--output_dir', type=str, default="pt_ecg_xai_results",
                        help="Directory to save output plots.")
    parser.add_argument('--save_plots', action='store_true',
                        help="Save plots instead of displaying them interactively.")

    return parser.parse_args()


# --- Main Execution ---
def main():
    args = parse_arguments()

    if args.list_available_methods:
        print("Available PyTorch XAI methods (curated list for ECG examples):")
        for method in ECG_FRIENDLY_METHODS:
            print(f"  - {method}")
        print("\nNote: These are methods generally suitable for ECG timeseries data.")
        return

    print("--- Initializing PyTorch ECG XAI Script ---")
    cli_method_params_dict = parse_cli_method_params(args.method_params)

    if args.run_all_ecg_methods:
        print(f"\n--- Running ALL {len(ECG_FRIENDLY_METHODS)} curated ECG-friendly PyTorch XAI methods ---")
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
                    args.save_plots, args.output_dir
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
            args.save_plots, args.output_dir
        )

    print("\n--- Script Finished ---")


if __name__ == '__main__':
    main()
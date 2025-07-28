#!/usr/bin/env python3
"""
Time Series XAI Method Comparison Script for ECG Data

This script compares the outputs of TensorFlow and PyTorch implementations
of XAI methods on ECG data and produces visualizations with highlighted areas.

Usage:
    python run_timeseries_method_comparison.py --pathology [MODEL] --method [METHOD] --record_id [RECORD_ID]

Example:
    python run_timeseries_method_comparison.py --pathology AVB --method gradient --record_id 03509_hr
    python run_timeseries_method_comparison.py --pathology LBBB --method integrated_gradients
    python run_timeseries_method_comparison.py --method smoothgrad  # Uses default ECG model
"""

import numpy as np
import tensorflow as tf
import torch
import os
import matplotlib.pyplot as plt
import sys
import argparse
from tensorflow.keras.models import model_from_json
from typing import Dict, Any, Tuple, Optional

# Add project root to sys.path BEFORE any local imports
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import ECG utilities (these require project_root in sys.path)
from utils.ecg_data import load_and_preprocess_ecg, perform_shape_switch
from utils.ecg_explainability import normalize_ecg_relevancemap
from utils.ecg_visualization import plot_ecg

# --- Import packages ---
from signxai.tf_signxai import calculate_relevancemap as tf_calculate_relevancemap
from signxai.utils.utils import remove_softmax as tf_remove_softmax
from signxai.torch_signxai import calculate_relevancemap as torch_calculate_relevancemap
from signxai.torch_signxai.utils import remove_softmax as torch_remove_softmax

# Add PyTorch ECG model directory to path
ecg_model_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'ECG')
if ecg_model_dir not in sys.path:
    sys.path.insert(0, ecg_model_dir)

# Import both model types
from ecg_model import ECG_PyTorch
from pathology_ecg_model import Pathology_ECG_PyTorch

# Configure method parameter mappings between TF and PyTorch
METHOD_PARAM_MAPPING = {
    'integrated_gradients': {
        'tf_params': ['steps', 'reference_inputs'],
        'pt_params': ['ig_steps', 'baseline']
    },
    'smoothgrad': {
        'tf_params': ['num_samples', 'noise_level'],
        'pt_params': ['num_samples', 'noise_level']
    },
    'grad_cam': {
        'tf_params': ['last_conv_layer_name'],
        'pt_params': ['target_layer']
    }
}

# Define default parameters for methods
DEFAULT_METHOD_PARAMS = {
    'integrated_gradients': {
        'steps': 50,
        'reference_inputs': None  # Will be set to zeros dynamically
    },
    'smoothgrad': {
        'augment_by_n': 25,  # TensorFlow parameter name
        'noise_scale': 0.1   # TensorFlow parameter name
    },
    'grad_cam': {
        'target_layer': None  # Will be set dynamically based on model
    }
}

# Configure target layers for Grad-CAM based on model type
GRAD_CAM_LAYERS = {
    'ecg': {
        'tf': 'last_conv',
        'pt': 'conv3'  # This should match the attribute name in ECG_PyTorch
    },
    'pathology': {
        'tf': 'conv1d_4',  # The final conv layer in pathology models
        'pt': 'conv5'  # This should match the attribute name in Pathology_ECG_PyTorch
    }
}

# Default LRP parameters for different methods
DEFAULT_LRP_PARAMS = {
    'lrp_epsilon': {
        'epsilon': 0.1
    },
    'lrp_alpha_1_beta_0': {
        'alpha': 1.0,
        'beta': 0.0
    },
    'lrp_alpha_2_beta_1': {
        'alpha': 2.0,
        'beta': 1.0
    },
    'lrp_z': {
        'rule_name': 'zplus',
        'epsilon': 1e-7
    },
    'lrp_flat': {
        'rule_name': 'flat',
        'epsilon': 1e-7
    }
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Time Series XAI Method Comparison for ECG Data')

    parser.add_argument('--pathology', type=str, choices=['AVB', 'ISCH', 'LBBB', 'RBBB'],
                        help='Pathology-specific model (if not specified, uses default ECG model)')

    parser.add_argument('--method', type=str, required=True,
                        choices=['gradient', 'integrated_gradients', 'smoothgrad', 'guided_backprop', 
                                 'input_t_gradient', 'deconvnet', 'grad_cam', 'lrp_epsilon',
                                 'lrp_alpha_1_beta_0', 'lrp_alpha_2_beta_1', 'lrp_z', 'lrp_flat'],
                        help='XAI method to apply')
                      
    parser.add_argument('--record_id', type=str, default='03509_hr',
                        help='ECG record ID to use (e.g., 03509_hr, 12131_hr, 14493_hr, 02906_hr)')

    parser.add_argument('--output_dir', type=str, default=os.path.join(current_script_dir, 'results'),
                        help='Directory to save results')
                        
    parser.add_argument('--posthresh', type=float, default=0.2,
                        help='Threshold for highlighting positive relevance values (0.0-1.0)')
                        
    parser.add_argument('--cmap_adjust', type=float, default=0.3,
                        help='Amplification factor for highlighted relevance values')
                        
    parser.add_argument('--comparison_only', action='store_true',
                        help='Only show comparison plots without ECG visualizations with red marks')
                        
    parser.add_argument('--combined_view', action='store_true',
                        help='Generate a combined visualization showing both TF and PyTorch results on the same plot')

    return parser.parse_args()


def load_tensorflow_model(pathology: Optional[str] = None) -> Tuple[tf.keras.models.Model, tf.keras.models.Model, Dict[str, Any]]:
    """
    Load TensorFlow model based on pathology.

    Args:
        pathology: Pathology type or None for default ECG model

    Returns:
        Tuple of (model, model_no_softmax, model_info)
    """
    model_info = {
        'input_channels': 1,
        'num_classes': 3,
        'model_type': 'ecg'
    }

    try:
        if pathology:
            # Pathology-specific model (AVB, ISCH, LBBB, RBBB)
            model_info.update({'input_channels': 12, 'num_classes': 2, 'model_type': 'pathology'})

            tf_json_path = os.path.join(project_root, 'examples', 'data', 'models',
                                        'tensorflow', 'ECG', pathology, 'model.json')
            tf_weights_path = os.path.join(project_root, 'examples', 'data', 'models',
                                          'tensorflow', 'ECG', pathology, 'weights.h5')

            print(f"Loading TensorFlow model from: {tf_json_path} and {tf_weights_path}")

            # Check if files exist
            if not os.path.exists(tf_json_path):
                raise FileNotFoundError(f"TensorFlow model JSON file not found: {tf_json_path}")
            if not os.path.exists(tf_weights_path):
                raise FileNotFoundError(f"TensorFlow model weights file not found: {tf_weights_path}")

            with open(tf_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(tf_weights_path)

            # Create a copy for no_softmax version
            with open(tf_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model_no_softmax = model_from_json(loaded_model_json)
            model_no_softmax.load_weights(tf_weights_path)
            model_no_softmax = tf_remove_softmax(model_no_softmax)
        else:
            # Default ECG model
            tf_model_path = os.path.join(project_root, 'examples', 'data', 'models',
                                         'tensorflow', 'ECG', 'ecg_model.h5')
            print(f"Loading TensorFlow model from: {tf_model_path}")

            # Check if file exists
            if not os.path.exists(tf_model_path):
                raise FileNotFoundError(f"TensorFlow model file not found: {tf_model_path}")

            model = tf.keras.models.load_model(tf_model_path, compile=False)
            model_no_softmax = tf_remove_softmax(tf.keras.models.load_model(tf_model_path, compile=False))

        return model, model_no_softmax, model_info
    
    except FileNotFoundError as e:
        print(f"Error loading TensorFlow model: {str(e)}")
        print("Please ensure the model files exist and that the pathology argument is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading TensorFlow model: {str(e)}")
        sys.exit(1)


def load_pytorch_model(pathology: Optional[str] = None) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[str, Any]]:
    """
    Load PyTorch model based on pathology.

    Args:
        pathology: Pathology type or None for default ECG model

    Returns:
        Tuple of (model, model_no_softmax, model_info)
    """
    model_info = {
        'input_channels': 1,
        'num_classes': 3,
        'model_type': 'ecg'
    }

    try:
        if pathology:
            # Pathology-specific model (AVB, ISCH, LBBB, RBBB)
            model_info.update({'input_channels': 12, 'num_classes': 2, 'model_type': 'pathology'})

            pt_weights_path = os.path.join(project_root, 'examples', 'data', 'models',
                                        'pytorch', 'ECG', pathology, f"{pathology.lower()}_ported_weights.pt")

            # Fallback path if not in specific directory
            if not os.path.exists(pt_weights_path):
                pt_weights_path = os.path.join(project_root, 'examples', 'data', 'models',
                                            'pytorch', 'ECG', f"{pathology.lower()}_ported_weights.pt")

            print(f"Loading PyTorch model weights from: {pt_weights_path}")

            # Check if file exists
            if not os.path.exists(pt_weights_path):
                raise FileNotFoundError(f"PyTorch model weights file not found: {pt_weights_path}")

            # Use the Pathology_ECG_PyTorch class
            pt_model = Pathology_ECG_PyTorch(input_channels=model_info['input_channels'],
                                            num_classes=model_info['num_classes'])
            pt_model.load_state_dict(torch.load(pt_weights_path, map_location=torch.device('cpu')))
            pt_model.eval()

            # Create a copy for no_softmax version
            pt_model_no_softmax = Pathology_ECG_PyTorch(input_channels=model_info['input_channels'],
                                                        num_classes=model_info['num_classes'])
            pt_model_no_softmax.load_state_dict(torch.load(pt_weights_path, map_location=torch.device('cpu')))
            pt_model_no_softmax.eval()
            torch_remove_softmax(pt_model_no_softmax)
        else:
            # Default ECG model
            pt_weights_path = os.path.join(project_root, 'examples', 'data', 'models',
                                        'pytorch', 'ECG', 'ecg_ported_weights.pt')
            print(f"Loading PyTorch model weights from: {pt_weights_path}")
            
            # Check if file exists
            if not os.path.exists(pt_weights_path):
                raise FileNotFoundError(f"PyTorch model weights file not found: {pt_weights_path}")

            # Use the ECG_PyTorch class
            pt_model = ECG_PyTorch(input_channels=model_info['input_channels'],
                                num_classes=model_info['num_classes'])
            pt_model.load_state_dict(torch.load(pt_weights_path, map_location=torch.device('cpu')))
            pt_model.eval()

            # Create a copy for no_softmax version
            pt_model_no_softmax = ECG_PyTorch(input_channels=model_info['input_channels'],
                                            num_classes=model_info['num_classes'])
            pt_model_no_softmax.load_state_dict(torch.load(pt_weights_path, map_location=torch.device('cpu')))
            pt_model_no_softmax.eval()
            torch_remove_softmax(pt_model_no_softmax)

        return pt_model, pt_model_no_softmax, model_info
        
    except FileNotFoundError as e:
        print(f"Error loading PyTorch model: {str(e)}")
        print("Please ensure the model files exist and that the pathology argument is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading PyTorch model: {str(e)}")
        sys.exit(1)


def load_ecg_data(record_id: str, model_info: Dict[str, Any]) -> np.ndarray:
    """
    Load and preprocess ECG data.
    
    Args:
        record_id: ECG record ID to use
        model_info: Information about the model
        
    Returns:
        Preprocessed ECG data
    """
    try:
        # Set the source directory to ensure we're loading from examples/data/timeseries
        src_dir = os.path.join(project_root, 'examples', 'data', 'timeseries', '')
        
        # Check if data file exists
        data_file = os.path.join(src_dir, f"{record_id}.dat")
        header_file = os.path.join(src_dir, f"{record_id}.hea")
        
        if not os.path.exists(data_file) or not os.path.exists(header_file):
            raise FileNotFoundError(f"ECG data files not found: {data_file} or {header_file}")
            
        # Load ECG data using utility function (use 2000 to match model expectations)
        ecg_data = load_and_preprocess_ecg(
            record_id=record_id,
            ecg_filters=['BWR', 'BLA', 'AC50Hz', 'LP40Hz'],
            subsampling_window_size=2000,
            subsample_start=0,
            src_dir=src_dir
        )
        
        print(f"Loaded ECG data with shape: {ecg_data.shape}")
        
        # Keep original ECG data for visualization (like individual scripts)
        original_ecg_data = ecg_data.copy()
        
        # Check if we need to adapt data for model input
        input_channels = model_info['input_channels']
        if ecg_data.shape[1] != input_channels:
            print(f"Adapting ECG data from {ecg_data.shape[1]} to {input_channels} channels for model")
            # Use the provided channels if possible
            if ecg_data.shape[1] < input_channels:
                # Duplicate existing channels
                ecg_data_orig = ecg_data.copy()
                ecg_data = np.zeros((ecg_data.shape[0], input_channels))
                for i in range(input_channels):
                    channel_idx = i % ecg_data_orig.shape[1]
                    ecg_data[:, i] = ecg_data_orig[:, channel_idx]
                print(f"Expanded model input ECG data to shape: {ecg_data.shape}")
            elif ecg_data.shape[1] > input_channels:
                # Take only the first N channels for model
                print(f"Taking only the first {input_channels} channels from ECG data for model")
                ecg_data = ecg_data[:, :input_channels]
                print(f"Reduced model input ECG data to shape: {ecg_data.shape}")
        
        print(f"Original ECG data shape for visualization: {original_ecg_data.shape}")
        return ecg_data, original_ecg_data
    
    except FileNotFoundError as e:
        print(f"Error loading ECG data: {str(e)}")
        print(f"Please ensure the ECG record ID '{record_id}' exists in the 'examples/data/timeseries/' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading ECG data: {str(e)}")
        sys.exit(1)


def prepare_ecg_data(ecg_data: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Prepare ECG data for TensorFlow and PyTorch models.

    Args:
        ecg_data: Raw ECG data

    Returns:
        Tuple of (tensorflow_input, pytorch_input)
    """
    # Prepare data for TensorFlow (shape: [batch_size, sequence_length, channels])
    tf_input = np.expand_dims(ecg_data, axis=0)  # Add batch dimension
    
    # Prepare data for PyTorch (shape: [batch_size, channels, sequence_length])
    pt_input = torch.from_numpy(tf_input.copy()).float()
    pt_input = pt_input.permute(0, 2, 1)  # Convert from [batch, seq, channels] to [batch, channels, seq]

    print(f"TensorFlow input shape: {tf_input.shape}")
    print(f"PyTorch input shape: {pt_input.shape}")

    return tf_input, pt_input


def get_method_params(method: str, model_info: Dict[str, Any], tf_input: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """
    Get parameters for the XAI method for both TensorFlow and PyTorch.

    Args:
        method: XAI method name
        model_info: Information about the model
        tf_input: TensorFlow input tensor (for reference values)

    Returns:
        Dictionary with 'tf' and 'pt' keys containing parameters for each framework
    """
    params = {'tf': {}, 'pt': {}}

    # Copy default parameters if available
    if method in DEFAULT_METHOD_PARAMS:
        for k, v in DEFAULT_METHOD_PARAMS[method].items():
            params['tf'][k] = v

    # Special handling for certain methods
    if method == 'integrated_gradients':
        # Create defaults for both frameworks to ensure consistency
        if params['tf'].get('reference_inputs') is None:
            params['tf']['reference_inputs'] = np.zeros_like(tf_input)
        
        if params['tf'].get('steps') is None:
            params['tf']['steps'] = 50
            
        # Map TF parameters to PyTorch parameters with exactly matching values
        params['pt']['ig_steps'] = params['tf']['steps']
        
        # PyTorch expects baseline as a tensor with correct dimension order
        params['pt']['baseline'] = torch.zeros_like(torch.from_numpy(tf_input).permute(0, 2, 1).float())

    elif method == 'grad_cam':
        # Set appropriate target layers based on model type
        model_type = model_info['model_type']  # 'ecg' or 'pathology'
        if model_type in GRAD_CAM_LAYERS:
            params['tf']['layer_name'] = GRAD_CAM_LAYERS[model_type]['tf']
            params['pt']['target_layer'] = GRAD_CAM_LAYERS[model_type]['pt']

    elif method == 'smoothgrad':
        # Create defaults for both frameworks to ensure consistency
        if params['tf'].get('augment_by_n') is None:
            params['tf']['augment_by_n'] = 25
        if params['tf'].get('noise_scale') is None:
            params['tf']['noise_scale'] = 0.1
            
        # Map TF parameters to PyTorch parameters with exactly matching values
        params['pt']['num_samples'] = params['tf']['augment_by_n']
        params['pt']['noise_level'] = params['tf']['noise_scale']
    
    # Handle LRP methods with default parameters
    elif method in DEFAULT_LRP_PARAMS:
        # Some TF methods have built-in parameters, don't override them
        if method in ['lrp_alpha_2_beta_1', 'lrp_alpha_1_beta_0', 'lrp_z', 'lrp_flat']:
            # These TF methods have built-in parameters
            # Only pass parameters to PyTorch
            for k, v in DEFAULT_LRP_PARAMS[method].items():
                params['pt'][k] = v
        else:
            # Add default LRP parameters for both frameworks
            for k, v in DEFAULT_LRP_PARAMS[method].items():
                params['tf'][k] = v
                params['pt'][k] = v

    return params


def compute_relevance_maps(
        method: str,
        tf_model: tf.keras.Model,
        pt_model: torch.nn.Module,
        tf_input: np.ndarray,
        pt_input: torch.Tensor,
        target_class_tf: int,
        target_class_pt: int,
        method_params: Dict[str, Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute relevance maps using the specified method for both frameworks.

    Args:
        method: XAI method name
        tf_model: TensorFlow model with softmax removed
        pt_model: PyTorch model with softmax removed
        tf_input: TensorFlow input tensor
        pt_input: PyTorch input tensor
        target_class_tf: TensorFlow target class
        target_class_pt: PyTorch target class
        method_params: Method parameters for both frameworks

    Returns:
        Tuple of (tensorflow_relevance_map, pytorch_relevance_map)
    """
    print(f"\nComputing relevance maps using method: {method}")

    # Add specific parameters for tensor dimension ordering for gradient-based methods
    if method in ['gradient_x_sign', 'gradient_x_input'] and 'mu' not in method_params['tf']:
        method_params['tf']['mu'] = 0.0  # Default mu value for sign thresholding

    # Convert method name to format expected by each framework
    tf_method_name = method
    pt_method_name = method

    # Some methods might have different names in different frameworks
    method_name_mapping = {
        'gradient': {'tf': 'gradient', 'pt': 'gradient'},
        'integrated_gradients': {'tf': 'integrated_gradients', 'pt': 'integrated_gradients'},
        'smoothgrad': {'tf': 'smoothgrad', 'pt': 'smoothgrad'},
        'guided_backprop': {'tf': 'guided_backprop', 'pt': 'guided_backprop'},
        'input_t_gradient': {'tf': 'input_t_gradient', 'pt': 'gradient_x_input'},
        'deconvnet': {'tf': 'deconvnet', 'pt': 'deconvnet'},
        'grad_cam': {'tf': 'grad_cam', 'pt': 'grad_cam'},
        'lrp_epsilon': {'tf': 'lrp.epsilon', 'pt': 'lrp.epsilon'},
        'lrp_alpha_1_beta_0': {'tf': 'lrp.alpha_1_beta_0', 'pt': 'lrp.alphabeta'},
        'lrp_alpha_2_beta_1': {'tf': 'lrp.alpha_2_beta_1', 'pt': 'lrp.alphabeta'},
        'lrp_z': {'tf': 'lrp.z', 'pt': 'lrp.zplus'},
        'lrp_flat': {'tf': 'lrp.flat', 'pt': 'lrp.flat'}
    }

    if method in method_name_mapping:
        tf_method_name = method_name_mapping[method]['tf']
        pt_method_name = method_name_mapping[method]['pt']

    # TensorFlow relevance map
    tf_relevance_params = method_params['tf'].copy()
    print(f"TensorFlow relevance map parameters: {tf_relevance_params}")

    # TensorFlow relevance map - no fallbacks, solid implementation or failure
    tf_relevance_map = tf_calculate_relevancemap(
        method=tf_method_name,
        x=tf_input,
        model=tf_model,
        neuron_selection=target_class_tf,
        **tf_relevance_params
    )
    print(f"TensorFlow relevance map shape: {tf_relevance_map.shape}")

    # PyTorch relevance map
    pt_relevance_params = method_params['pt'].copy()
    print(f"PyTorch relevance map parameters: {pt_relevance_params}")

    # Handle special case for GradCAM where we need to pass an attribute, not a string
    if method == 'grad_cam' and 'target_layer' in pt_relevance_params:
        target_layer_name = pt_relevance_params.pop('target_layer')
        if target_layer_name and hasattr(pt_model, target_layer_name):
            pt_relevance_params['target_layer'] = getattr(pt_model, target_layer_name)

    # PyTorch relevance map - no fallbacks, solid implementation or failure
    pt_relevance_map = torch_calculate_relevancemap(
        model=pt_model,
        input_tensor=pt_input,
        method=pt_method_name,
        target_class=target_class_pt,
        **pt_relevance_params
    )
    print(f"PyTorch relevance map shape: {pt_relevance_map.shape}")

    return tf_relevance_map, pt_relevance_map


def plot_ecg_comparison_12leads(
        ecg_data: np.ndarray,
        tf_relevance_map: np.ndarray,
        pt_relevance_map: np.ndarray,
        method_name: str,
        pathology: str,
        record_id: str,
        output_dir: str,
        sampling_rate: int = 500
) -> str:
    """
    Create a 12-lead ECG comparison plot with highlighted ECG lines and color-coded legend.
    
    Colors:
    - Red: Both TF and PyTorch match (perfect agreement)
    - Green: Only TensorFlow relevance 
    - Blue: Only PyTorch relevance
    
    Args:
        ecg_data: ECG signal data (2000, 12) or (12, 2000)
        tf_relevance_map: TensorFlow relevance map (2000, 12) or (12, 2000) 
        pt_relevance_map: PyTorch relevance map (2000, 12) or (12, 2000)
        method_name: XAI method name
        pathology: Pathology type
        record_id: ECG record ID
        output_dir: Output directory
        sampling_rate: ECG sampling rate
        
    Returns:
        Path to saved comparison plot
    """
    # Ensure data is in (leads, timesteps) format for plot_ecg function
    if ecg_data.shape[0] == 2000 and ecg_data.shape[1] == 12:
        ecg_data = ecg_data.T  # (12, 2000)
    if tf_relevance_map.shape[0] == 2000 and tf_relevance_map.shape[1] == 12:
        tf_relevance_map = tf_relevance_map.T  # (12, 2000)
    
    # Handle PyTorch relevance map shape conversion
    if pt_relevance_map.ndim == 3:
        # Remove batch dimension and convert to (leads, timesteps)
        if pt_relevance_map.shape[0] == 1:  # Batch dimension
            pt_relevance_map = pt_relevance_map[0]  # Now (12, 2000) or (2000, 12)
            if pt_relevance_map.shape[0] == 2000 and pt_relevance_map.shape[1] == 12:
                pt_relevance_map = pt_relevance_map.T  # (12, 2000)
    elif pt_relevance_map.shape[0] == 2000 and pt_relevance_map.shape[1] == 12:
        pt_relevance_map = pt_relevance_map.T  # (12, 2000)
    
    # Calculate correlation per lead to determine color coding
    lead_correlations = []
    comparison_explanation = np.zeros_like(tf_relevance_map)
    
    for lead_idx in range(12):
        tf_lead = tf_relevance_map[lead_idx].flatten()
        pt_lead = pt_relevance_map[lead_idx].flatten()
        
        # Calculate correlation
        if np.std(tf_lead) > 1e-10 and np.std(pt_lead) > 1e-10 and len(tf_lead) > 1:
            corr = np.corrcoef(tf_lead, pt_lead)[0, 1]
        else:
            corr = 0.0
        lead_correlations.append(corr)
        
        # Create comparison explanation based on correlation for color-coding
        if corr > 0.95:  # Perfect match - use combined relevance (will show as RED)
            comparison_explanation[lead_idx] = (tf_relevance_map[lead_idx] + pt_relevance_map[lead_idx]) / 2
        elif corr > 0.5:  # Moderate match - blend but favor differences
            comparison_explanation[lead_idx] = tf_relevance_map[lead_idx] * 0.8 + pt_relevance_map[lead_idx] * 0.2
        else:  # Poor match - show TF only (will show as GREEN for TF-only)
            comparison_explanation[lead_idx] = tf_relevance_map[lead_idx]
    
    # Calculate overall correlation for title
    overall_correlation = np.mean(lead_correlations)
    
    # Create title with correlation info
    title = f'{pathology} ECG Comparison - {method_name.upper()} ({record_id}) - Correlation: {overall_correlation:.3f}'
    
    # Determine colormap based on overall performance
    if overall_correlation > 0.95:
        # Perfect match - use red colormap to show perfect agreement
        cmap = 'Reds'
    else:
        # Use seismic colormap to show differences (red=positive, blue=negative, white=neutral)
        cmap = 'seismic'
    
    # Normalize explanation for better visualization
    from utils.ecg_explainability import normalize_ecg_relevancemap
    comparison_explanation_normalized = normalize_ecg_relevancemap(comparison_explanation)
    
    # First create the ECG plot without legend
    import os
    temp_filename = os.path.join(output_dir, f'temp_{pathology.lower()}_{method_name}_12leads_{record_id}.png')
    
    try:
        # Create the ECG plot with highlighted relevance using the same plot_ecg function
        plot_ecg(
            ecg=ecg_data,  # (12, 2000)
            explanation=comparison_explanation_normalized,  # (12, 2000)
            sampling_rate=sampling_rate,
            title=title,
            show_colorbar=True,
            cmap=cmap,
            bubble_size=25,  # Good size for visibility
            line_width=1.0,
            style='fancy',
            save_to=temp_filename,
            clim_min=-1,
            clim_max=1,
            colorbar_label='Relevance',
            shape_switch=False,  # Already in correct format
            dpi=300
        )
        
        # Now create the final plot with legend
        output_filename = os.path.join(output_dir, f'{pathology.lower()}_{method_name}_12leads_comparison_{record_id}.png')
        
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        # Create figure for final plot with legend
        fig = plt.figure(figsize=(18, 10))  # Wider to accommodate legend
        
        # Load the ECG plot image
        ecg_img = plt.imread(temp_filename)
        
        # Create main plot area (shifted right to make room for legend)
        ax_main = fig.add_axes([0.18, 0.1, 0.75, 0.8])  # Main plot area
        ax_main.imshow(ecg_img)
        ax_main.axis('off')
        
        # Create legend on the left side
        ax_legend = fig.add_axes([0.02, 0.15, 0.15, 0.7])  # Left side legend area
        ax_legend.axis('off')
        
        # Color legend
        legend_elements = [
            Patch(facecolor='red', label='Perfect Match'),
            Patch(facecolor='green', label='TensorFlow Only'), 
            Patch(facecolor='blue', label='PyTorch Only'),
            Patch(facecolor='white', edgecolor='black', label='No Relevance')
        ]
        
        # Add overall correlation info
        if overall_correlation > 0.95:
            legend_elements.insert(0, Patch(facecolor='darkred', label=f'Overall: PERFECT ({overall_correlation:.3f})'))
        elif overall_correlation > 0.7:
            legend_elements.insert(0, Patch(facecolor='orange', label=f'Overall: GOOD ({overall_correlation:.3f})'))
        else:
            legend_elements.insert(0, Patch(facecolor='lightblue', label=f'Overall: POOR ({overall_correlation:.3f})'))
        
        # Add per-lead correlation info 
        legend_elements.append(Patch(facecolor='lightgray', label='--- Per Lead ---'))
        lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for lead_name, corr in zip(lead_names, lead_correlations):
            if corr > 0.95:
                color = 'red'
            elif corr > 0.7:
                color = 'orange'
            else:
                color = 'lightblue'
            legend_elements.append(Patch(facecolor=color, label=f'{lead_name}: {corr:.3f}'))
        
        # Create the legend
        legend = ax_legend.legend(handles=legend_elements, loc='upper left', fontsize=9, 
                                frameon=True, fancybox=True, shadow=True)
        legend.set_title('TF vs PyTorch\nComparison', prop={'size': 11, 'weight': 'bold'})
        
        # Save the final plot with legend
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        
    except Exception as e:
        print(f"Error creating 12-lead comparison plot: {e}")
        # Fallback: create simpler version
        output_filename = os.path.join(output_dir, f'{pathology.lower()}_{method_name}_12leads_comparison_simple_{record_id}.png')
        plot_ecg(
            ecg=ecg_data,
            explanation=comparison_explanation_normalized,
            sampling_rate=sampling_rate,
            title=title,
            show_colorbar=True,
            cmap=cmap,
            bubble_size=30,
            save_to=output_filename,
            shape_switch=False,
            dpi=300
        )
    
    return output_filename


def normalize_and_compare_relevance_maps(
        tf_relevance_map: np.ndarray,
        pt_relevance_map: np.ndarray,
        model_info: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Normalize and prepare relevance maps for comparison with improved handling.
    
    This function performs critical operations to make TensorFlow and PyTorch relevance maps comparable.
    It ensures that both maps have the same dimensionality and alignment before computing metrics.
    
    The main differences to handle:
    1. TensorFlow format: [batch, sequence, channels] vs PyTorch format: [batch, channels, sequence]
    2. Different normalization methods potentially used in each framework
    3. Proper tensor conversion and shape standardization

    Args:
        tf_relevance_map: TensorFlow relevance map
        pt_relevance_map: PyTorch relevance map  
        model_info: Information about the model

    Returns:
        Tuple of (tf_flat, pt_flat, correlation, mae)
    """
    print(f"\nNormalizing relevance maps for comparison...")
    print(f"TF input shape: {tf_relevance_map.shape}")
    
    # Convert PyTorch tensor to numpy if needed
    if isinstance(pt_relevance_map, torch.Tensor):
        pt_relevance_map = pt_relevance_map.detach().cpu().numpy()
        print(f"Converted PT tensor to numpy: {pt_relevance_map.shape}")
    
    print(f"PT input shape: {pt_relevance_map.shape}")
    
    # Ensure TF relevance map is also numpy
    if hasattr(tf_relevance_map, 'numpy'):
        tf_relevance_map = tf_relevance_map.numpy()
    elif isinstance(tf_relevance_map, torch.Tensor):
        tf_relevance_map = tf_relevance_map.detach().cpu().numpy()
    
    # Handle batch dimensions
    if tf_relevance_map.ndim == 3 and tf_relevance_map.shape[0] == 1:
        tf_relevance_map = tf_relevance_map[0]
        print(f"Removed TF batch dimension: {tf_relevance_map.shape}")
    
    if pt_relevance_map.ndim == 3 and pt_relevance_map.shape[0] == 1:
        pt_relevance_map = pt_relevance_map[0]
        print(f"Removed PT batch dimension: {pt_relevance_map.shape}")
    
    # Now handle the core format conversion for time series data
    # Expected: TF format [Seq, Channels], PT format [Channels, Seq]
    
    if pt_relevance_map.ndim == 3:
        # Still has batch or extra dimension - handle carefully
        if pt_relevance_map.shape[0] == model_info.get('input_channels', 12):
            # [Channels, Seq, Extra] or [Channels, Batch, Seq]
            pt_relevance_map = pt_relevance_map.mean(axis=-1)  # Average extra dimension
            print(f"Averaged PT extra dimension: {pt_relevance_map.shape}")
        elif pt_relevance_map.shape[-1] == model_info.get('input_channels', 12):
            # [Extra, Seq, Channels] or [Batch, Seq, Channels] 
            pt_relevance_map = pt_relevance_map.mean(axis=0)  # Average first dimension
            print(f"Averaged PT first dimension: {pt_relevance_map.shape}")
        else:
            # Unknown format - take middle slice
            pt_relevance_map = pt_relevance_map[pt_relevance_map.shape[0]//2]
            print(f"Took middle slice of PT: {pt_relevance_map.shape}")
    
    if tf_relevance_map.ndim == 3:
        # Similar handling for TF
        if tf_relevance_map.shape[-1] == model_info.get('input_channels', 12):
            # [Extra, Seq, Channels] or [Batch, Seq, Channels] - take mean of extra dimension
            tf_relevance_map = tf_relevance_map.mean(axis=0)
            print(f"Averaged TF first dimension: {tf_relevance_map.shape}")
        else:
            # Unknown format - take middle slice
            tf_relevance_map = tf_relevance_map[tf_relevance_map.shape[0]//2]
            print(f"Took middle slice of TF: {tf_relevance_map.shape}")
    
    # Now both should be 2D - convert PT from [C, Seq] to [Seq, C] if needed
    if pt_relevance_map.ndim == 2 and tf_relevance_map.ndim == 2:
        # Determine format based on which dimension matches expected channels
        expected_channels = model_info.get('input_channels', 12)
        expected_sequence = 2000  # Typical sequence length
        
        # Check TF format
        if tf_relevance_map.shape[-1] == expected_channels:
            tf_format = "[Seq, Channels]"
        elif tf_relevance_map.shape[0] == expected_channels:
            tf_format = "[Channels, Seq]"
            tf_relevance_map = tf_relevance_map.T  # Transpose to [Seq, Channels]
            print(f"Transposed TF to [Seq, Channels]: {tf_relevance_map.shape}")
        else:
            tf_format = "Unknown - assuming [Seq, Channels]"
        
        # Check PT format  
        if pt_relevance_map.shape[0] == expected_channels:
            pt_format = "[Channels, Seq]"
            pt_relevance_map = pt_relevance_map.T  # Transpose to [Seq, Channels]
            print(f"Transposed PT to [Seq, Channels]: {pt_relevance_map.shape}")
        elif pt_relevance_map.shape[-1] == expected_channels:
            pt_format = "[Seq, Channels]"
        else:
            pt_format = "Unknown - assuming [Seq, Channels]"
        
        print(f"Detected formats - TF: {tf_format}, PT: {pt_format}")
    
    # Ensure shapes match for comparison
    if tf_relevance_map.shape != pt_relevance_map.shape:
        print(f"Shape mismatch after alignment! TF: {tf_relevance_map.shape}, PT: {pt_relevance_map.shape}")
        
        # Try to make them comparable by taking common dimensions
        if tf_relevance_map.ndim == 2 and pt_relevance_map.ndim == 2:
            min_seq = min(tf_relevance_map.shape[0], pt_relevance_map.shape[0])
            min_channels = min(tf_relevance_map.shape[1], pt_relevance_map.shape[1])
            
            tf_relevance_map = tf_relevance_map[:min_seq, :min_channels]
            pt_relevance_map = pt_relevance_map[:min_seq, :min_channels]
            print(f"Trimmed to common shape: {tf_relevance_map.shape}")
        else:
            # Fallback to flattening
            tf_flat_fallback = tf_relevance_map.flatten()
            pt_flat_fallback = pt_relevance_map.flatten()
            min_len = min(len(tf_flat_fallback), len(pt_flat_fallback))
            tf_flat = tf_flat_fallback[:min_len]
            pt_flat = pt_flat_fallback[:min_len]
            print(f"Fallback to flattened comparison: {len(tf_flat)} elements")
    else:
        # Shapes match - aggregate properly for time series
        print(f"Shapes match: {tf_relevance_map.shape}")
        
        # For time series, average across channels to get per-timestep relevance
        if tf_relevance_map.ndim == 2:  # [Seq, Channels]
            tf_flat = np.mean(np.abs(tf_relevance_map), axis=1)  # Average across channels
            pt_flat = np.mean(np.abs(pt_relevance_map), axis=1)  # Average across channels
        else:
            tf_flat = tf_relevance_map.flatten()
            pt_flat = pt_relevance_map.flatten()
        
        print(f"Aggregated to time series: TF={len(tf_flat)}, PT={len(pt_flat)} timesteps")
    
    # Ensure equal lengths
    min_len = min(len(tf_flat), len(pt_flat))
    tf_flat = tf_flat[:min_len]
    pt_flat = pt_flat[:min_len]
    
    # Normalize both to [0,1] range for fair comparison
    if tf_flat.max() != tf_flat.min():
        tf_flat = (tf_flat - tf_flat.min()) / (tf_flat.max() - tf_flat.min())
    if pt_flat.max() != pt_flat.min():
        pt_flat = (pt_flat - pt_flat.min()) / (pt_flat.max() - pt_flat.min())
    
    # Compute comparison metrics
    correlation = 0.0
    mae = 0.0

    if len(tf_flat) == len(pt_flat) and len(tf_flat) > 0:
        # Standardize for correlation computation
        tf_std = tf_flat.std() + 1e-8
        pt_std = pt_flat.std() + 1e-8
        
        tf_norm = (tf_flat - tf_flat.mean()) / tf_std
        pt_norm = (pt_flat - pt_flat.mean()) / pt_std

        # Compute correlation
        if not (np.isnan(tf_norm).any() or np.isnan(pt_norm).any()):
            correlation = float(np.corrcoef(tf_norm, pt_norm)[0, 1])
        
        # Mean absolute error on normalized values
        mae = float(np.mean(np.abs(tf_norm - pt_norm)))
    
    print(f"Comparison metrics: correlation={correlation:.4f}, mae={mae:.4f}")
    return tf_flat, pt_flat, correlation, mae


def process_relevance_map_for_visualization(
        relevance_map: np.ndarray,
        posthresh: float = 0.2,
        cmap_adjust: float = 0.3,
        method: str = ''
) -> np.ndarray:
    """
    Process relevance map for visualization: normalize, threshold, and amplify.
    
    Args:
        relevance_map: Raw relevance map
        posthresh: Threshold for highlighting positive relevance values
        cmap_adjust: Amplification factor for highlighted relevance values
        method: The XAI method name (for method-specific processing)
        
    Returns:
        Processed relevance map ready for visualization
    """
    # Handle batch dimension if present
    if relevance_map.ndim > 3:
        relevance_map = relevance_map[0]  # Take the first item in the batch
    
    # Check if we need to transpose or reshape
    if relevance_map.ndim == 3 and relevance_map.shape[0] == 1 and relevance_map.shape[2] > relevance_map.shape[1]:
        # If shape is [1, channels, seq_len], reshape to [seq_len, channels]
        relevance_map = np.squeeze(relevance_map, axis=0).transpose(1, 0)
    elif relevance_map.ndim == 3 and relevance_map.shape[0] > 1:
        # If shape is [channels, seq_len, 1], reshape to [seq_len, channels]
        relevance_map = relevance_map.transpose(1, 0)
    
    # Ensure relevance_map is 2D after all transformations
    if relevance_map.ndim != 2:
        print(f"Warning: Expected 2D shape for visualization, got {relevance_map.shape}. Reshaping...")
        # Try to reshape based on contents
        if relevance_map.size > 0:
            if relevance_map.ndim == 1:
                # If 1D, assume it's a single channel and expand
                relevance_map = relevance_map.reshape(-1, 1)
            elif relevance_map.ndim > 2:
                # If higher dimensional, flatten to 2D guessing best dimensions
                orig_shape = relevance_map.shape
                if len(orig_shape) >= 3:
                    # For 3D+, assume first dim is batch or channel, second is sequence length
                    seq_len = orig_shape[1] if orig_shape[1] > orig_shape[0] else orig_shape[0]
                    channels = relevance_map.size // seq_len
                    relevance_map = relevance_map.reshape(seq_len, channels)
    
    # Method-specific processing
    if method.lower() == 'smoothgrad':
        # For SmoothGrad, use absolute values to focus on magnitude, not direction
        relevance_map = np.abs(relevance_map)
    else:
        # For other methods, use only positives
        relevance_map_pos = relevance_map.copy()
        relevance_map_pos[relevance_map_pos < 0] = 0
        relevance_map = relevance_map_pos
    
    # Normalize relevance map
    relevance_map_norm = normalize_ecg_relevancemap(relevance_map, local=False)
    
    # Apply thresholding and amplification
    relevance_map_norm[relevance_map_norm <= posthresh] = 0
    relevance_map_norm[relevance_map_norm > posthresh] = relevance_map_norm[relevance_map_norm > posthresh] + cmap_adjust
    
    return relevance_map_norm


def main():
    """Main function to run the comparison."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Results will be saved to {args.output_dir}")

    # Set TensorFlow to use CPU only if GPU is available
    try:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.set_visible_devices([], 'GPU')
    except AttributeError:
        print("No GPU devices found, using CPU by default")

    # Load models based on pathology argument
    print("\n--- Loading Models ---")
    tf_model, tf_model_no_softmax, tf_model_info = load_tensorflow_model(args.pathology)
    pt_model, pt_model_no_softmax, pt_model_info = load_pytorch_model(args.pathology)

    # Load ECG data
    print("\n--- Loading ECG Data ---")
    raw_ecg_data, original_ecg_data = load_ecg_data(args.record_id, tf_model_info)
    
    # Prepare ECG data for both frameworks
    print("\n--- Preparing Data ---")
    tf_input, pt_input = prepare_ecg_data(raw_ecg_data)

    # Make predictions
    print("\n--- Computing Predictions ---")
    tf_pred = tf_model.predict(tf_input, verbose=0)
    target_class_tf = np.argmax(tf_pred[0])
    print(f"TensorFlow predicted class: {target_class_tf}, confidence: {tf_pred[0][target_class_tf]:.4f}")

    with torch.no_grad():
        pt_pred = pt_model(pt_input)
    target_class_pt = torch.argmax(pt_pred, dim=1).item()
    print(f"PyTorch predicted class: {target_class_pt}, confidence: {pt_pred[0][target_class_pt].item():.4f}")

    # Get method parameters
    method_params = get_method_params(args.method, tf_model_info, tf_input)

    # Compute relevance maps
    tf_relevance_map, pt_relevance_map = compute_relevance_maps(
        args.method, tf_model_no_softmax, pt_model_no_softmax,
        tf_input, pt_input, target_class_tf, target_class_pt, method_params
    )

    # Normalize and compare
    tf_flat, pt_flat, correlation, mae = normalize_and_compare_relevance_maps(
        tf_relevance_map, pt_relevance_map, tf_model_info
    )

    # Print comparison metrics
    print(f"\n--- Comparison Metrics ---")
    print(f"Correlation between TensorFlow and PyTorch relevance maps: {correlation:.4f}")
    print(f"Mean absolute error between normalized relevance maps: {mae:.4f}")

    # Save the numerical data (save both model input and original visualization data)
    data_filename = f"{args.pathology.lower() if args.pathology else 'ecg'}_{args.method}_{args.record_id}_data.npz"
    data_filepath = os.path.join(args.output_dir, data_filename)
    np.savez(
        data_filepath,
        ecg_data=raw_ecg_data,
        original_ecg_data=original_ecg_data,
        tf_relevance=tf_relevance_map,
        pt_relevance=pt_relevance_map,
        correlation=correlation,
        mae=mae
    )
    print(f"Saved numerical data to: {data_filepath}")

    # Create 12-lead ECG comparison visualization
    print(f"\n--- Creating 12-Lead ECG Comparison ---")
    comparison_plot_path = plot_ecg_comparison_12leads(
        ecg_data=original_ecg_data,
        tf_relevance_map=tf_relevance_map, 
        pt_relevance_map=pt_relevance_map,
        method_name=args.method,
        pathology=args.pathology or 'ECG',
        record_id=args.record_id,
        output_dir=args.output_dir,
        sampling_rate=500
    )
    print(f"Saved 12-lead comparison to: {comparison_plot_path}")


if __name__ == "__main__":
    main()
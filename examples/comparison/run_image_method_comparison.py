import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import inspect
import importlib

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_predictions_tf
from signxai.utils.utils import load_image as signxai_load_tf_image
from signxai.utils.utils import remove_softmax as tf_remove_softmax
from signxai.tf_signxai.methods.wrappers import calculate_relevancemap as tf_calculate_relevancemap

# PyTorch imports
import torch
from PIL import Image
from signxai.torch_signxai import calculate_relevancemap as torch_calculate_relevancemap
from signxai.torch_signxai.utils import remove_softmax as torch_remove_softmax
from signxai.torch_signxai.utils import decode_predictions as decode_predictions_pytorch
import signxai.torch_signxai.methods.wrappers as torch_wrappers

# --- Import visualization utilities from signxai.common.visualization ---
from signxai.common.visualization import normalize_relevance_map, relevance_to_heatmap, overlay_heatmap
print("Successfully imported visualization utilities from signxai.common.visualization.")

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

IMAGE_NAME = 'example.jpg'
IMAGE_PATH = os.path.join(PROJECT_ROOT, 'examples/data/images', IMAGE_NAME)
TF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'examples/data/models/tensorflow/VGG16/model.h5')
PT_MODEL_DEFINITION_DIR = os.path.join(PROJECT_ROOT, 'examples/data/models/pytorch/VGG16')
PT_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'examples/data/models/pytorch/VGG16/vgg16_ported_weights.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')

TARGET_SIZE = (224, 224)
IMAGENET_CLASSES_PATH = None


def call_pytorch_method(method_name, model, input_tensor, target_class, **kwargs):
    """Call PyTorch method with improved error handling and method routing."""
    print(f"Calling PyTorch method: {method_name}")
    print(f"Input tensor shape: {input_tensor.shape}, target class: {target_class}")
    print(f"Method kwargs: {kwargs}")
    
    # Ensure model is in eval mode
    model.eval()
    
    # Ensure input tensor requires grad for gradient methods
    if not input_tensor.requires_grad:
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
    
    try:
        # Try wrapper function first (preferred for compatibility)
        if hasattr(torch_wrappers, method_name):
            wrapper_func = getattr(torch_wrappers, method_name)
            if callable(wrapper_func):
                print(f"Using wrapper function for {method_name}")
                
                # Adjust kwargs for wrapper function if needed
                wrapper_kwargs = kwargs.copy()
                
                # Some wrapper functions expect different parameter names
                if 'layer_name' in wrapper_kwargs:
                    # Convert layer name to actual layer object if needed
                    layer_name = wrapper_kwargs.pop('layer_name')
                    if hasattr(model, layer_name):
                        wrapper_kwargs['target_layer'] = getattr(model, layer_name)
                    else:
                        print(f"Warning: Layer {layer_name} not found in model")
                
                result = wrapper_func(model, input_tensor, **wrapper_kwargs)
                print(f"Wrapper function succeeded, result shape: {result.shape if hasattr(result, 'shape') else 'no shape'}")
                return result
    
    except Exception as wrapper_error:
        print(f"Wrapper function failed: {wrapper_error}")
        print("Falling back to Zennit implementation")
    
    try:
        # Fall back to the regular Zennit-based implementation
        print(f"Using Zennit implementation for {method_name}")
        
        # Prepare kwargs for Zennit implementation
        zennit_kwargs = kwargs.copy()
        
        # Map parameter names if needed
        if 'num_samples' in zennit_kwargs and 'noise_level' in zennit_kwargs:
            # SmoothGrad parameters
            pass  # Already in correct format
        elif 'steps' in zennit_kwargs:
            # Integrated Gradients parameters
            pass  # Already in correct format
        
        result = torch_calculate_relevancemap(
            model=model, 
            input_tensor=input_tensor,
            method=method_name, 
            target_class=target_class, 
            **zennit_kwargs
        )
        print(f"Zennit implementation succeeded, result shape: {result.shape if hasattr(result, 'shape') else 'no shape'}")
        return result
        
    except Exception as zennit_error:
        print(f"Zennit implementation failed: {zennit_error}")
        
        # Last resort: try method-specific implementations
        try:
            print(f"Trying method-specific implementation for {method_name}")
            
            if method_name == 'gradient' or method_name == 'vanilla_gradients':
                # Simple gradient implementation
                input_tensor.grad = None
                output = model(input_tensor)
                target_output = output[0, target_class]
                target_output.backward()
                return input_tensor.grad.clone()
            
            elif method_name == 'grad_cam':
                # Try direct GradCAM implementation
                from signxai.torch_signxai.methods.grad_cam import calculate_grad_cam_relevancemap
                target_layer = kwargs.get('target_layer')
                if target_layer is None and 'layer_name' in kwargs:
                    layer_name = kwargs['layer_name']
                    if hasattr(model, layer_name):
                        target_layer = getattr(model, layer_name)
                
                if target_layer is not None:
                    return calculate_grad_cam_relevancemap(
                        model, input_tensor, target_layer, target_class
                    )
            
            # If no specific implementation, raise the last error
            raise zennit_error
            
        except Exception as final_error:
            print(f"All PyTorch implementations failed for {method_name}: {final_error}")
            raise final_error


def discover_all_methods():
    """Dynamically discover all available methods from both frameworks with improved filtering."""
    tf_methods = []
    pt_methods = []
    
    # Import TensorFlow wrapper module
    try:
        tf_wrapper_module = importlib.import_module('signxai.tf_signxai.methods.wrappers')
        
        def is_valid_tf_method(name, obj):
            if not inspect.isfunction(obj) or name.startswith('_'):
                return False
            # Exclude utility functions
            if name.startswith('calculate_native') or name.startswith('calculate_grad_cam'):
                return False
            # Exclude wrapper utility functions
            if name.endswith('_wrapper') or name in ['calculate_relevancemap', 'calculate_relevancemaps']:
                return False
            return True
        
        tf_methods = [name for name, obj in inspect.getmembers(tf_wrapper_module) 
                     if is_valid_tf_method(name, obj)]
        print(f"Discovered {len(tf_methods)} TensorFlow methods")
        print(f"TF methods sample: {tf_methods[:10]}")
    except Exception as e:
        print(f"Error discovering TensorFlow methods: {e}")
    
    # Import PyTorch wrapper module  
    try:
        pt_wrapper_module = importlib.import_module('signxai.torch_signxai.methods.wrappers')
        
        def is_valid_pt_method(name, obj):
            if not inspect.isfunction(obj) or name.startswith('_'):
                return False
            
            # Exclude utility functions
            if name in ['_calculate_relevancemap', 'calculate_relevancemap', 'calculate_relevancemaps']:
                return False
            if name.endswith('_wrapper'):
                return False
            
            # Check function signature to handle methods with required parameters properly
            try:
                sig = inspect.signature(obj)
                required_params = [p for p in sig.parameters.values() 
                                 if p.default == inspect.Parameter.empty and 
                                 p.name not in ['model_no_softmax', 'x', 'model', 'input_tensor', 'args', 'kwargs']]
                
                # Allow methods with mu if they have wrapper versions without mu
                if any(p.name in ['mu'] for p in required_params):
                    # Check if there's a wrapper version without mu
                    wrapper_name = f"{name}_wrapper"
                    if hasattr(pt_wrapper_module, wrapper_name):
                        return False  # Skip this, use wrapper instead
                    else:
                        return False  # Skip methods that require mu without wrapper
                
                # Allow other required parameters that are commonly provided
                allowed_required = ['target_class', 'neuron_selection', 'baseline', 'target_layer']
                remaining_required = [p for p in required_params if p.name not in allowed_required]
                if remaining_required:
                    return False
                    
            except Exception as sig_error:
                print(f"Error checking signature for {name}: {sig_error}")
                return True  # Include if we can't check signature
            
            return True
        
        pt_methods = [name for name, obj in inspect.getmembers(pt_wrapper_module) 
                     if is_valid_pt_method(name, obj)]
        print(f"Discovered {len(pt_methods)} PyTorch methods")
        print(f"PT methods sample: {pt_methods[:10]}")
    except Exception as e:
        print(f"Error discovering PyTorch methods: {e}")
    
    # Find common methods
    common_methods = list(set(tf_methods) & set(pt_methods))
    common_methods.sort()
    
    print(f"Found {len(common_methods)} common methods between frameworks")
    print(f"Common methods sample: {common_methods[:10]}")
    
    return tf_methods, pt_methods, common_methods


def categorize_methods(methods):
    """Categorize methods by type for better organization and analysis."""
    categories = {
        'gradient_based': [],
        'gradient_variants': [],
        'smooth_integrated': [],
        'backprop_methods': [],
        'feature_based': [],
        'lrp_epsilon': [],
        'lrp_alpha_beta': [],
        'lrp_sequential': [],
        'lrp_rule_based': [],
        'lrp_sign': [],
        'lrp_zbox': [],
        'lrp_std_x': [],
        'deeplift_methods': [],
        'combination_methods': [],
        'model_specific': [],
        'utilities': []
    }
    
    for method in methods:
        # Core gradient methods
        if method in ['gradient', 'input_t_gradient']:
            categories['gradient_based'].append(method)
        elif method in ['smoothgrad', 'integrated_gradients', 'vargrad']:
            categories['smooth_integrated'].append(method)
        elif 'gradient' in method and ('x_input' in method or 'x_sign' in method):
            categories['gradient_variants'].append(method)
        
        # Backpropagation methods
        elif method in ['guided_backprop', 'deconvnet'] or 'guided_' in method or 'deconvnet' in method:
            categories['backprop_methods'].append(method)
        
        # Feature-based methods
        elif 'grad_cam' in method:
            categories['feature_based'].append(method)
        
        # DeepLift methods
        elif 'deeplift' in method:
            categories['deeplift_methods'].append(method)
        
        # LRP Epsilon methods (most common LRP variant)
        elif 'epsilon' in method and ('lrp' in method or 'lrpz' in method or 'lrpsign' in method):
            if 'std_x' in method:
                categories['lrp_std_x'].append(method)
            else:
                categories['lrp_epsilon'].append(method)
        
        # LRP Alpha-Beta methods
        elif ('alpha' in method and 'beta' in method) or 'sequential_composite' in method:
            categories['lrp_alpha_beta'].append(method)
        
        # LRP Sequential/Composite methods
        elif 'sequential' in method or 'composite' in method:
            categories['lrp_sequential'].append(method)
        
        # LRP Rule-based methods (gamma, flat, w_square, z_plus, etc.)
        elif any(rule in method for rule in ['lrp_gamma', 'lrp_flat', 'lrp_w_square', 'lrp_z_plus', 'lrp_z', 'flatlrp', 'w2lrp']):
            categories['lrp_rule_based'].append(method)
        
        # LRP Sign methods
        elif 'lrpsign' in method or ('sign' in method and 'lrp' in method):
            categories['lrp_sign'].append(method)
        
        # LRP ZBox/bounded methods
        elif 'zblrp' in method or 'bounded' in method:
            categories['lrp_zbox'].append(method)
        
        # Combination methods (x_input, x_sign combinations)
        elif any(combo in method for combo in ['_x_input', '_x_sign', '_x_input_x_sign']):
            categories['combination_methods'].append(method)
        
        # Model-specific methods
        elif any(model in method for model in ['VGG16', 'MNIST', 'timeseries', 'ILSVRC', 'MITPL365']):
            categories['model_specific'].append(method)
        
        # Utility methods
        elif method in ['random_uniform'] or method.startswith('calculate_'):
            categories['utilities'].append(method)
        
        # Fallback for any unclassified LRP methods
        elif method.startswith('lrp') or 'lrp' in method:
            categories['lrp_rule_based'].append(method)
        
        # Everything else
        else:
            categories['utilities'].append(method)
    
    return categories


def get_method_parameters(method_name):
    """Get appropriate parameters for each method type.
    
    This function provides comprehensive parameter mapping for all ~202 XAI methods
    to ensure equivalent behavior between TensorFlow and PyTorch implementations.
    """
    base_params = {'tf_kwargs': {}, 'pt_kwargs': {}}
    
    print(f"Getting parameters for method: {method_name}")
    
    # Core gradient-based methods
    if method_name in ['gradient', 'gradient_x_input']:
        # Basic gradient methods - no special parameters needed
        pass
        
    elif method_name in ['smoothgrad'] or 'smoothgrad' in method_name:
        # SmoothGrad parameters - ensure identical noise sampling
        if 'smoothgrad_x_input' in method_name:
            base_params['tf_kwargs'] = {'augment_by_n': 25, 'noise_scale': 0.1}
            base_params['pt_kwargs'] = {'num_samples': 25, 'noise_level': 0.1}
        else:
            base_params['tf_kwargs'] = {'augment_by_n': 25, 'noise_scale': 0.1}
            base_params['pt_kwargs'] = {'num_samples': 25, 'noise_level': 0.1}
    
    elif method_name in ['integrated_gradients'] or 'integrated_gradients' in method_name:
        # Integrated Gradients - ensure identical step count
        base_params['tf_kwargs'] = {'steps': 50, 'reference_inputs': None}  # None = zeros baseline
        base_params['pt_kwargs'] = {'steps': 50, 'baseline': None}  # None = zeros baseline
    
    elif method_name in ['vargrad'] or 'vargrad' in method_name:
        # VarGrad parameters
        base_params['tf_kwargs'] = {'num_samples': 25, 'noise_level': 0.2}
        base_params['pt_kwargs'] = {'num_samples': 25, 'noise_level': 0.2}
    
    # Guided backpropagation methods
    elif 'guided_backprop' in method_name or 'guided_grad_cam' in method_name:
        # Guided backprop - no special parameters
        pass
    
    # Deconvolution methods
    elif 'deconvnet' in method_name:
        # Deconvnet - no special parameters
        pass
    
    # GradCAM methods - layer mapping is critical
    elif 'grad_cam' in method_name:
        if 'VGG16ILSVRC' in method_name or 'VGG16' in method_name:
            base_params['tf_kwargs'] = {'layer_name': 'block5_conv3'}
            base_params['pt_kwargs'] = {'layer_name': 'features.28'}  # Equivalent layer in PyTorch VGG16
        elif 'VGG16MITPL365' in method_name:
            base_params['tf_kwargs'] = {'layer_name': 'block5_conv3'}
            base_params['pt_kwargs'] = {'layer_name': 'features.28'}
        elif 'MNISTCNN' in method_name:
            base_params['tf_kwargs'] = {'layer_name': 'conv2d_1'}
            base_params['pt_kwargs'] = {'layer_name': 'conv2'}
        elif 'timeseries' in method_name:
            # For timeseries models
            base_params['tf_kwargs'] = {'layer_name': 'conv1d_2'}
            base_params['pt_kwargs'] = {'layer_name': 'conv3'}
        else:
            # Default to VGG16 layers
            base_params['tf_kwargs'] = {'layer_name': 'block5_conv3'}
            base_params['pt_kwargs'] = {'layer_name': 'features.28'}
    
    # DeepLift methods
    elif 'deeplift' in method_name:
        base_params['tf_kwargs'] = {'baseline_type': 'zero'}
        base_params['pt_kwargs'] = {'baseline_type': 'zero'}
    
    # LRP Epsilon methods - extensive parameter mapping
    elif 'epsilon' in method_name:
        # Extract epsilon value from method name
        if '0_001' in method_name:
            eps_val = 0.001
        elif '0_01' in method_name:
            eps_val = 0.01
        elif '0_1' in method_name:
            eps_val = 0.1
        elif '0_2' in method_name:
            eps_val = 0.2
        elif '0_25' in method_name:
            eps_val = 0.25
        elif '0_5' in method_name:
            eps_val = 0.5
        elif method_name.endswith('_1') or 'epsilon_1' in method_name:
            eps_val = 1.0
        elif method_name.endswith('_5') or 'epsilon_5' in method_name:
            eps_val = 5.0
        elif method_name.endswith('_10') or 'epsilon_10' in method_name:
            eps_val = 10.0
        elif method_name.endswith('_20') or 'epsilon_20' in method_name:
            eps_val = 20.0
        elif method_name.endswith('_50') or 'epsilon_50' in method_name:
            eps_val = 50.0
        elif method_name.endswith('_75') or 'epsilon_75' in method_name:
            eps_val = 75.0
        elif method_name.endswith('_100') or 'epsilon_100' in method_name:
            eps_val = 100.0
        else:
            eps_val = 0.1  # Default epsilon
        
        base_params['tf_kwargs'] = {'epsilon': eps_val}
        base_params['pt_kwargs'] = {'epsilon': eps_val}
        
        # Add standard deviation factor for std_x variants
        if 'std_x' in method_name:
            if '0_1_std_x' in method_name:
                stdfactor = 0.1
            elif '0_25_std_x' in method_name:
                stdfactor = 0.25
            elif '0_5_std_x' in method_name:
                stdfactor = 0.5
            elif '1_std_x' in method_name:
                stdfactor = 1.0
            elif '2_std_x' in method_name:
                stdfactor = 2.0
            elif '3_std_x' in method_name:
                stdfactor = 3.0
            else:
                stdfactor = 0.25  # Default
            base_params['tf_kwargs']['stdfactor'] = stdfactor
            base_params['pt_kwargs']['stdfactor'] = stdfactor
    
    # LRP Alpha-Beta methods
    elif ('alpha' in method_name and 'beta' in method_name) or 'sequential_composite' in method_name:
        if 'alpha_1_beta_0' in method_name:
            base_params['tf_kwargs'] = {}  # TF has built-in alpha_1_beta_0
            base_params['pt_kwargs'] = {'alpha': 1.0, 'beta': 0.0}
        elif 'alpha_2_beta_1' in method_name:
            base_params['tf_kwargs'] = {}  # TF has built-in alpha_2_beta_1
            base_params['pt_kwargs'] = {'alpha': 2.0, 'beta': 1.0}
        elif 'sequential_composite_a' in method_name:
            base_params['tf_kwargs'] = {}
            base_params['pt_kwargs'] = {'composite_rule': 'composite_a'}
        elif 'sequential_composite_b' in method_name:
            base_params['tf_kwargs'] = {}
            base_params['pt_kwargs'] = {'composite_rule': 'composite_b'}
    
    # LRP Gamma methods
    elif 'lrp_gamma' in method_name:
        base_params['tf_kwargs'] = {}
        base_params['pt_kwargs'] = {'gamma': 0.5}
    
    # LRP specialized rule methods
    elif 'lrp_z' in method_name or 'lrp_z_plus' in method_name:
        base_params['tf_kwargs'] = {}
        base_params['pt_kwargs'] = {'rule': 'z_plus'}
    
    elif 'lrp_flat' in method_name or 'flatlrp' in method_name:
        base_params['tf_kwargs'] = {}
        base_params['pt_kwargs'] = {'rule': 'flat'}
    
    elif 'lrp_w_square' in method_name or 'w2lrp' in method_name:
        base_params['tf_kwargs'] = {}
        base_params['pt_kwargs'] = {'rule': 'w_square'}
    
    # LRP ZBox/bounded methods  
    elif 'zblrp' in method_name or 'bounded' in method_name:
        # ZBox requires input bounds - assume [0,1] for normalized images
        base_params['tf_kwargs'] = {'low': 0.0, 'high': 1.0}
        base_params['pt_kwargs'] = {'low': 0.0, 'high': 1.0}
    
    # LRP Sign methods with mu parameter
    elif 'lrpsign' in method_name or ('sign' in method_name and 'mu' in method_name):
        # Extract mu value
        if '_mu_0_5' in method_name:
            mu_val = 0.5
        elif '_mu_neg_0_5' in method_name:
            mu_val = -0.5
        elif '_mu_0' in method_name:
            mu_val = 0.0
        else:
            mu_val = 0.0  # Default
        
        base_params['tf_kwargs'] = {'mu': mu_val}
        base_params['pt_kwargs'] = {'mu': mu_val, 'sign_method': True}
    
    # Methods with mu parameter (sign variants)
    elif '_mu_' in method_name or method_name.endswith('_mu_0') or method_name.endswith('_mu_0_5') or method_name.endswith('_mu_neg_0_5'):
        if '_mu_0_5' in method_name:
            mu_val = 0.5
        elif '_mu_neg_0_5' in method_name:
            mu_val = -0.5
        elif '_mu_0' in method_name:
            mu_val = 0.0
        else:
            mu_val = 0.0
        
        base_params['tf_kwargs'] = {'mu': mu_val}
        base_params['pt_kwargs'] = {'mu': mu_val}
    
    # Method combinations with x_input
    elif '_x_input' in method_name:
        # Input multiplication variants - typically no additional parameters
        pass
    
    # Method combinations with x_sign  
    elif '_x_sign' in method_name:
        # Sign variants - typically no additional parameters
        pass
    
    # Random baseline method
    elif method_name == 'random_uniform':
        # Random uniform baseline
        pass
    
    print(f"Parameters set - TF: {base_params['tf_kwargs']}, PT: {base_params['pt_kwargs']}")
    return base_params


# --- Helper to add PyTorch model directory to path ---
def add_to_sys_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_to_sys_path(PT_MODEL_DEFINITION_DIR)
try:
    from VGG16 import VGG16_PyTorch
except ImportError as e:
    print(f"Error importing VGG16_PyTorch from {PT_MODEL_DEFINITION_DIR}: {e}")
    sys.exit(1)


def preprocess_pytorch_image(image_path, target_size):
    with Image.open(image_path) as img_opened:
        pil_img_pt = img_opened.convert('RGB')
    pil_img_pt_resized = pil_img_pt.resize(target_size)
    image_np = np.array(pil_img_pt_resized, dtype=np.float32)
    image_np_bgr = image_np[:, :, ::-1]
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    image_np_bgr_mean_subtracted = image_np_bgr - mean_bgr
    x_pt = torch.tensor(np.transpose(image_np_bgr_mean_subtracted, (2, 0, 1)), dtype=torch.float32).unsqueeze(0)
    return pil_img_pt_resized, x_pt


def aggregate_explanation(explanation_map, framework_name="Framework"):
    """Aggregate explanation maps with improved channel handling and dimension consistency.
    
    Args:
        explanation_map: Raw explanation map from XAI method
        framework_name: Name for logging (TensorFlow/PyTorch)
        
    Returns:
        Aggregated 2D explanation map of TARGET_SIZE
    """
    print(f"Aggregating {framework_name} explanation with shape: {explanation_map.shape if hasattr(explanation_map, 'shape') else 'no shape attr'}")
    
    # Convert tensor to numpy if needed
    if hasattr(explanation_map, 'numpy'):
        explanation_map = explanation_map.numpy()
    elif hasattr(explanation_map, 'detach'):
        explanation_map = explanation_map.detach().cpu().numpy()
    elif not isinstance(explanation_map, np.ndarray):
        try:
            explanation_map = np.array(explanation_map)
        except:
            print(f"Warning: {framework_name} explanation cannot be converted to numpy array. Using zeros.")
            return np.zeros(TARGET_SIZE)
    
    if not hasattr(explanation_map, 'ndim'):
        print(f"Warning: {framework_name} explanation is not a proper array. Using zeros for heatmap.")
        return np.zeros(TARGET_SIZE)
    
    print(f"Processing {framework_name} explanation shape: {explanation_map.shape}")
    
    # Handle different dimensionalities
    if explanation_map.ndim == 4:  # Batch dimension present
        if explanation_map.shape[0] == 1:
            explanation_map = explanation_map[0]  # Remove batch dimension
            print(f"Removed batch dimension: {explanation_map.shape}")
        else:
            print(f"Warning: {framework_name} has unexpected batch size {explanation_map.shape[0]}. Taking first item.")
            explanation_map = explanation_map[0]
    
    if explanation_map.ndim == 3:  # 3D: Could be (H,W,C) or (C,H,W)
        h, w = TARGET_SIZE
        
        # Determine format based on dimensions
        if explanation_map.shape[0] == h and explanation_map.shape[1] == w:  # (H,W,C) format
            print(f"Detected TF format (H,W,C): {explanation_map.shape}")
            # Sum across channels, but use weighted sum for better aggregation
            agg_map = np.mean(np.abs(explanation_map), axis=-1)  # Use absolute mean instead of sum
        elif explanation_map.shape[1] == h and explanation_map.shape[2] == w:  # (C,H,W) format
            print(f"Detected PT format (C,H,W): {explanation_map.shape}")
            # Sum across channels (first axis)
            agg_map = np.mean(np.abs(explanation_map), axis=0)  # Use absolute mean instead of sum
        else:
            # Try to determine which dimension is channels based on size
            # Channels typically much smaller than spatial dimensions
            dims = explanation_map.shape
            channel_dim = np.argmin(dims)  # Assume smallest dimension is channels
            
            print(f"Ambiguous format {dims}, assuming channel dimension: {channel_dim}")
            if channel_dim == 0:  # (C,H,W)
                agg_map = np.mean(np.abs(explanation_map), axis=0)
            elif channel_dim == 2:  # (H,W,C)
                agg_map = np.mean(np.abs(explanation_map), axis=-1)
            else:  # (H,C,W) - unusual but handle it
                agg_map = np.mean(np.abs(explanation_map), axis=1)
            
            print(f"Aggregated to shape: {agg_map.shape}")
    
    elif explanation_map.ndim == 2:  # Already 2D
        print(f"Already 2D: {explanation_map.shape}")
        agg_map = explanation_map
    
    elif explanation_map.ndim == 1:  # 1D - unusual, try to reshape
        print(f"1D explanation {explanation_map.shape}, attempting to reshape to {TARGET_SIZE}")
        if explanation_map.size == TARGET_SIZE[0] * TARGET_SIZE[1]:
            agg_map = explanation_map.reshape(TARGET_SIZE)
        else:
            print(f"Cannot reshape 1D array of size {explanation_map.size} to {TARGET_SIZE}. Using zeros.")
            return np.zeros(TARGET_SIZE)
    
    else:
        print(f"Warning: {framework_name} explanation has unsupported shape {explanation_map.shape}. Using zeros.")
        return np.zeros(TARGET_SIZE)
    
    # Ensure we have a 2D array at this point
    if agg_map.ndim != 2:
        print(f"Warning: Aggregation resulted in {agg_map.ndim}D array. Flattening and reshaping.")
        if agg_map.size == TARGET_SIZE[0] * TARGET_SIZE[1]:
            agg_map = agg_map.flatten().reshape(TARGET_SIZE)
        else:
            print(f"Cannot reshape to target size. Using zeros.")
            return np.zeros(TARGET_SIZE)
    
    # Resize if needed with high-quality interpolation
    if agg_map.shape != TARGET_SIZE:
        print(f"Resizing {framework_name} from {agg_map.shape} to {TARGET_SIZE}")
        try:
            # Normalize to [0,1] for PIL compatibility
            agg_min, agg_max = agg_map.min(), agg_map.max()
            if agg_max != agg_min:
                agg_normalized = (agg_map - agg_min) / (agg_max - agg_min)
            else:
                agg_normalized = agg_map
            
            pil_temp = Image.fromarray((agg_normalized * 255).astype(np.uint8))
            pil_resized = pil_temp.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            agg_map_resized = np.array(pil_resized).astype(np.float32) / 255.0
            
            # Denormalize back to original range
            if agg_max != agg_min:
                agg_map = agg_map_resized * (agg_max - agg_min) + agg_min
            else:
                agg_map = agg_map_resized
            
            print(f"Successfully resized to {agg_map.shape}, range: [{agg_map.min():.4f}, {agg_map.max():.4f}]")
        except Exception as resize_err:
            print(f"Error resizing heatmap: {resize_err}. Using zeros.")
            return np.zeros(TARGET_SIZE)
    
    print(f"Final {framework_name} aggregated shape: {agg_map.shape}, range: [{agg_map.min():.4f}, {agg_map.max():.4f}]")
    return agg_map


def run_comparison_for_method(method_name, tf_model_original, tf_model_no_softmax, x_tf,
                              pt_model_original, pt_model_for_xai, x_pt,
                              pil_img_for_display, save_plots=True):
    print(f"\n--- Running for Method: {method_name.upper()} ---")

    # TensorFlow Part
    print("--- TensorFlow ---")
    preds_tf = tf_model_original.predict(x_tf, verbose=0)
    decoded_preds_tf_keras = decode_predictions_tf(preds_tf, top=1)[0][0]
    print(
        f"TF Predicted: {decoded_preds_tf_keras[1]} (ID: {decoded_preds_tf_keras[0]}, Score: {decoded_preds_tf_keras[2]:.4f})")
    target_class_tf = np.argmax(preds_tf[0])
    
    # Get method parameters
    method_params = get_method_parameters(method_name)
    tf_kwargs = method_params['tf_kwargs']
    
    print(f"Attempting to call tf_calculate_relevancemap with method '{method_name}' and kwargs: {tf_kwargs}")
    print(f"TF input shape: {x_tf.shape}, TF model type: {type(tf_model_no_softmax)}, target class: {target_class_tf}")
    try:
        # Ensure input is properly formatted for TF
        tf_input_copy = x_tf.copy()
        if isinstance(tf_input_copy, torch.Tensor):
            tf_input_copy = tf_input_copy.detach().cpu().numpy()
        
        explanation_tf = tf_calculate_relevancemap(method_name, tf_input_copy, tf_model_no_softmax,
                                                   neuron_selection=int(target_class_tf), **tf_kwargs)
        print(f"TF raw explanation shape: {explanation_tf.shape if hasattr(explanation_tf, 'shape') else 'N/A'}")
        
        # Convert to numpy if needed
        if hasattr(explanation_tf, 'numpy'):
            explanation_tf = explanation_tf.numpy()
        elif isinstance(explanation_tf, torch.Tensor):
            explanation_tf = explanation_tf.detach().cpu().numpy()
            
        explanation_tf_agg = aggregate_explanation(explanation_tf, "TensorFlow")
        print(
            f"TF Explanation shape (aggregated): {explanation_tf_agg.shape if hasattr(explanation_tf_agg, 'shape') else 'N/A'}")
        tf_success = True
    except Exception as e:
        print(f"Error during TensorFlow '{method_name}' explanation: {e}")
        import traceback
        traceback.print_exc()
        explanation_tf_agg = np.zeros(TARGET_SIZE)
        tf_success = False

    # PyTorch Part
    print("\n--- PyTorch ---")
    with torch.no_grad():
        output_pt = pt_model_original(x_pt)
    decoded_preds_pt_list = decode_predictions_pytorch(output_pt, top=1, class_list_path=IMAGENET_CLASSES_PATH)
    decoded_preds_pt_item = decoded_preds_pt_list[0][0] if decoded_preds_pt_list and decoded_preds_pt_list[0] else (
    "N/A", "Unknown", 0.0)
    print(
        f"PyTorch Predicted: {decoded_preds_pt_item[1]} (ID: {decoded_preds_pt_item[0]}, Score: {decoded_preds_pt_item[2]:.4f})")
    target_class_pt = torch.argmax(output_pt, dim=1).item()
    
    pt_kwargs = method_params['pt_kwargs']
    print(f"Attempting to call PyTorch method '{method_name}' with kwargs: {pt_kwargs}")
    print(f"PT input shape: {x_pt.shape}, PT model type: {type(pt_model_for_xai)}, target class: {target_class_pt}")
    try:
        # Ensure input is properly formatted for PT
        pt_input_copy = x_pt.clone()
        if isinstance(pt_input_copy, np.ndarray):
            pt_input_copy = torch.from_numpy(pt_input_copy).float()
            
        explanation_pt_raw = call_pytorch_method(method_name, pt_model_for_xai, pt_input_copy, target_class_pt, **pt_kwargs)
        print(f"PT raw explanation shape: {explanation_pt_raw.shape if hasattr(explanation_pt_raw, 'shape') else 'N/A'}")
        
        # Convert to numpy if needed
        if isinstance(explanation_pt_raw, torch.Tensor):
            explanation_pt_raw = explanation_pt_raw.detach().cpu().numpy()
            
        explanation_pt_agg = aggregate_explanation(explanation_pt_raw, "PyTorch")
        print(
            f"PyTorch Explanation shape (aggregated): {explanation_pt_agg.shape if hasattr(explanation_pt_agg, 'shape') else 'N/A'}")
        pt_success = True
    except Exception as e:
        print(f"Error during PyTorch '{method_name}' explanation: {e}")
        import traceback
        traceback.print_exc()
        explanation_pt_agg = np.zeros(TARGET_SIZE)
        pt_success = False

    # Calculate metrics
    result = {'tf_success': tf_success, 'pt_success': pt_success}
    
    if tf_success and pt_success:
        tf_norm_expl = normalize_relevance_map(explanation_tf_agg)
        pt_norm_expl = normalize_relevance_map(explanation_pt_agg)
        
        if tf_norm_expl.shape == pt_norm_expl.shape:
            diff_map = tf_norm_expl - pt_norm_expl
            result['mae'] = float(np.mean(np.abs(diff_map)))
            result['mse'] = float(np.mean(np.square(diff_map)))
            result['max_diff'] = float(np.max(np.abs(diff_map)))
            
            # Calculate correlation
            tf_flat = tf_norm_expl.flatten()
            pt_flat = pt_norm_expl.flatten()
            result['correlation'] = float(np.corrcoef(tf_flat, pt_flat)[0, 1] if not np.isnan(tf_flat).any() and not np.isnan(pt_flat).any() else float('nan'))
            result['success'] = True
        else:
            result['error'] = f"Shape mismatch: TF {tf_norm_expl.shape}, PT {pt_norm_expl.shape}"
            result['success'] = False
    else:
        result['success'] = False
        if not tf_success and not pt_success:
            result['error'] = "Both frameworks failed"
        elif not tf_success:
            result['error'] = "TensorFlow failed"
        else:
            result['error'] = "PyTorch failed"

    # Save visualization if requested and both succeeded
    if save_plots and tf_success and pt_success:
        print("\n--- Visualizing ---")
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(pil_img_for_display);
        axs[0].set_title("Original Image");
        axs[0].axis('off')

        valid_plots = True
        if not (hasattr(explanation_tf_agg, 'shape') and hasattr(explanation_pt_agg, 'shape') and \
                explanation_tf_agg.ndim == 2 and explanation_pt_agg.ndim == 2 and \
                explanation_tf_agg.shape == TARGET_SIZE and explanation_pt_agg.shape == TARGET_SIZE):
            print(
                f"Error: Explanations for method '{method_name}' are not valid 2D arrays of target size for plotting. Skipping visualization details.")
            axs[1].text(0.5, 0.5, "TF Data Error", ha='center', va='center');
            axs[1].axis('off')
            axs[2].text(0.5, 0.5, "PT Data Error", ha='center', va='center');
            axs[2].axis('off')
            axs[3].text(0.5, 0.5, "Diff Error", ha='center', va='center');
            axs[3].axis('off')
            valid_plots = False

        if valid_plots:
            tf_norm_expl = normalize_relevance_map(explanation_tf_agg)
            print(f"TF explanation shape after normalization: {tf_norm_expl.shape}, range: [{tf_norm_expl.min():.4f}, {tf_norm_expl.max():.4f}]")
            
            # Use relevance_to_heatmap for consistent RGB output
            tf_heatmap_rgb = relevance_to_heatmap(tf_norm_expl, cmap="seismic", symmetric=True)
            axs[1].imshow(tf_heatmap_rgb)
            axs[1].set_title(f"TF {method_name}")
            axs[1].axis('off')

            pt_norm_expl = normalize_relevance_map(explanation_pt_agg)
            print(f"PT explanation shape after normalization: {pt_norm_expl.shape}, range: [{pt_norm_expl.min():.4f}, {pt_norm_expl.max():.4f}]")
            
            # Use relevance_to_heatmap for consistent RGB output
            pt_heatmap_rgb = relevance_to_heatmap(pt_norm_expl, cmap="seismic", symmetric=True)
            axs[2].imshow(pt_heatmap_rgb)
            axs[2].set_title(f"PyTorch {method_name}")
            axs[2].axis('off')

            if tf_norm_expl.shape != pt_norm_expl.shape:  # Should be caught by earlier check, but good to have
                print(f"Warning: Shape mismatch for difference map. TF: {tf_norm_expl.shape}, PT: {pt_norm_expl.shape}")
                axs[3].text(0.5, 0.5, "Shape Mismatch", ha='center', va='center', fontsize=9)
                axs[3].set_title("Difference (Error)");
                axs[3].axis('off')
            else:
                diff_map = tf_norm_expl - pt_norm_expl
                abs_diff_map = np.abs(diff_map)
                mae = np.mean(abs_diff_map)
                max_abs_diff_val = np.max(abs_diff_map) if np.any(abs_diff_map) else 1.0
                im = axs[3].imshow(abs_diff_map, cmap='hot', vmin=0, vmax=max_abs_diff_val if max_abs_diff_val > 0 else 1.0)
                axs[3].set_title(f"Abs Difference (MAE: {mae:.3e})");
                axs[3].axis('off')
                try:
                    fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
                except Exception as e:
                    print(f"Could not add colorbar for difference map: {e}")

        plt.suptitle(f"Comparison: {method_name.capitalize()} on VGG16", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        output_filename = os.path.join(OUTPUT_DIR, f"{method_name}_comparison_tf_pt_vgg16.jpg")
        plt.savefig(output_filename);
        print(f"Comparison image for '{method_name}' saved as {output_filename}")
        plt.close(fig)

    return result


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        try:
            from signxai.utils.utils import download_image  # Keep import local
            print("Attempting to download example.jpg...")
            os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
            download_image(IMAGE_PATH)
            if not os.path.exists(IMAGE_PATH): print(
                f"Download failed. Please place {IMAGE_NAME} at {IMAGE_PATH}."); return
        except ImportError:
            print(f"download_image utility not found. Please place {IMAGE_NAME} at {IMAGE_PATH}."); return
        except Exception as e:
            print(f"Error downloading image: {e}. Please place {IMAGE_NAME} at {IMAGE_PATH}."); return

    if not os.path.exists(TF_MODEL_PATH): print(f"TF Model not found at {TF_MODEL_PATH}"); return
    tf_model_original = tf_load_model(TF_MODEL_PATH)
    tf_model_no_softmax = tf_remove_softmax(tf_load_model(TF_MODEL_PATH))  # Load fresh for no_softmax
    _, x_tf = signxai_load_tf_image(IMAGE_PATH, target_size=TARGET_SIZE, expand_dims=True,
                                    use_original_preprocessing=True)

    if not os.path.exists(PT_MODEL_WEIGHTS_PATH): print(
        f"PyTorch Model weights not found at {PT_MODEL_WEIGHTS_PATH}"); return
    pt_model_original = VGG16_PyTorch(num_classes=1000);
    pt_model_original.load_state_dict(torch.load(PT_MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')));
    pt_model_original.eval()
    pt_model_for_xai = VGG16_PyTorch(num_classes=1000);
    pt_model_for_xai.load_state_dict(torch.load(PT_MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')));
    pt_model_for_xai.eval()
    torch_remove_softmax(pt_model_for_xai)
    pil_img_for_display, x_pt = preprocess_pytorch_image(IMAGE_PATH, TARGET_SIZE)
    
    # Discover all available methods
    print("Discovering all available methods...")
    tf_methods, pt_methods, common_methods = discover_all_methods()
    
    # Filter out utility methods and internal functions
    excluded_methods = {
        'calculate_relevancemap', '_calculate_relevancemap', 'calculate_native_gradient', 
        'calculate_native_integrated_gradients', 'calculate_native_smoothgrad',
        'calculate_grad_cam_relevancemap', 'calculate_grad_cam_relevancemap_timeseries',
        'calculate_relevancemaps', 'calculate_grad_cam_relevancemap_VGG16ILSVRC',
        'calculate_grad_cam_relevancemap_VGG16MITPL365', 'calculate_grad_cam_relevancemap_MNISTCNN',
        # Additional utility functions to exclude
        'calculate_relevancemap', 'calculate_relevancemaps',
        'deconvnet_x_sign_mu_wrapper', 'gradient_x_sign_mu_wrapper', 
        'guided_backprop_x_sign_mu_wrapper', 'lrp_epsilon_wrapper',
        'deeplift_method'
    }
    common_methods = [m for m in common_methods if m not in excluded_methods and 
                     not m.startswith('calculate_') and not m.startswith('_')]
    
    print(f"Testing {len(common_methods)} common methods")
    
    # Categorize methods
    categories = categorize_methods(common_methods)
    
    # Initialize data collection for the summary report
    comparison_results = {
        'model_logits': {
            'tf': None,
            'pt': None
        },
        'predictions': {
            'tf': None,
            'pt': None
        },
        'methods': {},
        'categories': categories
    }
    
    # Get model predictions 
    tf_preds = tf_model_original.predict(x_tf, verbose=0)
    pt_preds = pt_model_original(x_pt).detach().cpu().numpy()
    
    # Store predictions in results
    comparison_results['model_logits']['tf'] = tf_preds[0]
    comparison_results['model_logits']['pt'] = pt_preds[0]
    
    # Store top class predictions
    decoded_preds_tf = decode_predictions_tf(tf_preds, top=1)[0][0]
    comparison_results['predictions']['tf'] = {
        'class_id': decoded_preds_tf[0],
        'class_name': decoded_preds_tf[1],
        'score': float(decoded_preds_tf[2])
    }
    
    decoded_preds_pt_list = decode_predictions_pytorch(torch.tensor(pt_preds), top=1, class_list_path=IMAGENET_CLASSES_PATH)
    decoded_preds_pt_item = decoded_preds_pt_list[0][0] if decoded_preds_pt_list and decoded_preds_pt_list[0] else ("N/A", "Unknown", 0.0)
    comparison_results['predictions']['pt'] = {
        'class_id': decoded_preds_pt_item[0],
        'class_name': decoded_preds_pt_item[1],
        'score': float(decoded_preds_pt_item[2])
    }
    
    # Run all methods and collect data
    total_methods = len(common_methods)
    for i, method_name in enumerate(common_methods, 1):
        print(f"\n{'-'*80}")
        print(f"Progress: {i}/{total_methods} - Running method: {method_name}")
        print(f"{'-'*80}")
        
        try:
            result = run_comparison_for_method(method_name, tf_model_original, tf_model_no_softmax, x_tf, 
                                             pt_model_original, pt_model_for_xai, x_pt, pil_img_for_display, 
                                             save_plots=True)
            comparison_results['methods'][method_name] = result
        except Exception as e:
            print(f"Error running comparison for {method_name}: {e}")
            comparison_results['methods'][method_name] = {
                'error': str(e),
                'success': False,
                'tf_success': False,
                'pt_success': False
            }
    
    # Write detailed report
    report_path = os.path.join(OUTPUT_DIR, 'values_image_method_comparison.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SIGNXAI: Comprehensive TensorFlow vs PyTorch Comparison Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"METHODS TESTED: {len(common_methods)} common methods\n")
        f.write(f"TensorFlow methods available: {len(tf_methods)}\n")
        f.write(f"PyTorch methods available: {len(pt_methods)}\n")
        f.write(f"Common methods: {len(common_methods)}\n\n")
        
        f.write("MODEL PREDICTIONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"TensorFlow: Class {comparison_results['predictions']['tf']['class_name']} ")
        f.write(f"(ID: {comparison_results['predictions']['tf']['class_id']}, ")
        f.write(f"Score: {comparison_results['predictions']['tf']['score']:.4f})\n")
        
        f.write(f"PyTorch: Class {comparison_results['predictions']['pt']['class_name']} ")
        f.write(f"(ID: {comparison_results['predictions']['pt']['class_id']}, ")
        f.write(f"Score: {comparison_results['predictions']['pt']['score']:.4f})\n\n")
        
        f.write("FRAMEWORK SUCCESS RATES\n")
        f.write("-" * 80 + "\n")
        tf_successes = sum(1 for r in comparison_results['methods'].values() if r.get('tf_success', False))
        pt_successes = sum(1 for r in comparison_results['methods'].values() if r.get('pt_success', False))
        both_successes = sum(1 for r in comparison_results['methods'].values() if r.get('success', False))
        
        f.write(f"TensorFlow successful methods: {tf_successes}/{len(common_methods)} ({tf_successes/len(common_methods)*100:.1f}%)\n")
        f.write(f"PyTorch successful methods: {pt_successes}/{len(common_methods)} ({pt_successes/len(common_methods)*100:.1f}%)\n")
        f.write(f"Both frameworks successful: {both_successes}/{len(common_methods)} ({both_successes/len(common_methods)*100:.1f}%)\n\n")
        
        f.write("METHOD COMPARISONS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<35} {'MAE':<12} {'MSE':<12} {'Max Diff':<12} {'Correlation':<12} {'Status':<15}\n")
        f.write("-" * 80 + "\n")
        
        for method_name in common_methods:
            result = comparison_results['methods'].get(method_name, {'success': False, 'error': 'Not evaluated'})
            
            if result.get('success', False):
                f.write(f"{method_name:<35} {result['mae']:<12.6f} {result['mse']:<12.6f} ")
                f.write(f"{result['max_diff']:<12.6f} {result['correlation']:<12.6f} {'Success':<15}\n")
            else:
                error = result.get('error', 'Unknown error')
                status = f"Failed: {error[:10]}"
                f.write(f"{method_name:<35} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {status:<15}\n")
        
        f.write("\n\nSUMMARY BY METHOD CATEGORY\n")
        f.write("-" * 80 + "\n")
        
        # Group methods by category
        for category, methods in categories.items():
            if not methods:
                continue
                
            f.write(f"\n{category.replace('_', ' ').title()} Methods ({len(methods)} methods):\n")
            successful_methods = [m for m in methods if m in comparison_results['methods'] 
                                  and comparison_results['methods'][m].get('success', False)]
            
            if successful_methods:
                avg_mae = np.mean([comparison_results['methods'][m]['mae'] for m in successful_methods])
                corr_values = [comparison_results['methods'][m]['correlation'] for m in successful_methods 
                               if not np.isnan(comparison_results['methods'][m]['correlation'])]
                avg_corr = np.mean(corr_values) if corr_values else float('nan')
                
                f.write(f"  Average MAE: {avg_mae:.6f}\n")
                f.write(f"  Average Correlation: {avg_corr:.6f}\n")
                f.write(f"  Success Rate: {len(successful_methods)}/{len(methods)} methods ({len(successful_methods)/len(methods)*100:.1f}%)\n")
                
                # List best and worst in category
                if len(successful_methods) > 1:
                    best_method = min(successful_methods, key=lambda m: comparison_results['methods'][m]['mae'])
                    worst_method = max(successful_methods, key=lambda m: comparison_results['methods'][m]['mae'])
                    f.write(f"  Best: {best_method} (MAE: {comparison_results['methods'][best_method]['mae']:.6f})\n")
                    f.write(f"  Worst: {worst_method} (MAE: {comparison_results['methods'][worst_method]['mae']:.6f})\n")
            else:
                f.write(f"  No successful methods in this category\n")
        
        f.write("\n\nOVERALL CONCLUSION\n")
        f.write("-" * 80 + "\n")
        
        # Calculate overall statistics for successful methods
        successful_methods = [m for m in common_methods if m in comparison_results['methods'] 
                              and comparison_results['methods'][m].get('success', False)]
        
        if successful_methods:
            avg_mae = np.mean([comparison_results['methods'][m]['mae'] for m in successful_methods])
            corr_values = [comparison_results['methods'][m]['correlation'] for m in successful_methods 
                           if not np.isnan(comparison_results['methods'][m]['correlation'])]
            avg_corr = np.mean(corr_values) if corr_values else float('nan')
            
            best_method = min(successful_methods, key=lambda m: comparison_results['methods'][m]['mae'])
            worst_method = max(successful_methods, key=lambda m: comparison_results['methods'][m]['mae'])
            
            f.write(f"Overall Success Rate: {len(successful_methods)}/{len(common_methods)} methods ({len(successful_methods)/len(common_methods)*100:.1f}%)\n")
            f.write(f"Average MAE across all successful methods: {avg_mae:.6f}\n")
            f.write(f"Average Correlation across all successful methods: {avg_corr:.6f}\n\n")
            
            f.write(f"Most similar implementation (lowest MAE): {best_method} ")
            f.write(f"(MAE: {comparison_results['methods'][best_method]['mae']:.6f})\n")
            
            f.write(f"Least similar implementation (highest MAE): {worst_method} ")
            f.write(f"(MAE: {comparison_results['methods'][worst_method]['mae']:.6f})\n\n")
            
            # Framework-specific analysis
            tf_only_failures = [m for m in common_methods if not comparison_results['methods'][m].get('tf_success', False) 
                               and comparison_results['methods'][m].get('pt_success', False)]
            pt_only_failures = [m for m in common_methods if comparison_results['methods'][m].get('tf_success', False) 
                               and not comparison_results['methods'][m].get('pt_success', False)]
            
            if tf_only_failures:
                f.write(f"TensorFlow-only failures ({len(tf_only_failures)}): {', '.join(tf_only_failures[:10])}")
                if len(tf_only_failures) > 10:
                    f.write(f" ... and {len(tf_only_failures) - 10} more")
                f.write("\n")
                
            if pt_only_failures:
                f.write(f"PyTorch-only failures ({len(pt_only_failures)}): {', '.join(pt_only_failures[:10])}")
                if len(pt_only_failures) > 10:
                    f.write(f" ... and {len(pt_only_failures) - 10} more")
                f.write("\n")
        else:
            f.write("No methods were successfully compared\n")
    
    print(f"\nComprehensive comparison report saved to {report_path}")
    
    # Create a summary plot of similarity metrics
    successful_methods = [m for m in common_methods if m in comparison_results['methods'] 
                          and comparison_results['methods'][m].get('success', False)]
    
    if successful_methods:
        plt.figure(figsize=(20, 12))
        
        # MAE plot
        mae_values = [comparison_results['methods'][m]['mae'] for m in successful_methods]
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(successful_methods)), mae_values)
        plt.xticks(range(len(successful_methods)), successful_methods, rotation=45, ha='right')
        plt.title(f'Mean Absolute Error (MAE) between TensorFlow and PyTorch - {len(successful_methods)} Methods')
        plt.ylabel('MAE (lower is better)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.yscale('log')
        
        # Color bars by category
        colors = {'gradient_based': 'blue', 'gradient_variants': 'lightblue', 'backprop_methods': 'green', 
                 'feature_based': 'red', 'lrp_core': 'purple', 'lrp_epsilon': 'magenta', 
                 'lrp_alpha_beta': 'orange', 'lrp_variants': 'brown', 'lrp_sign': 'pink',
                 'lrp_zbox': 'gray', 'lrp_flat': 'olive', 'lrp_w2': 'cyan', 
                 'model_specific': 'yellow', 'utilities': 'black'}
        
        for i, method in enumerate(successful_methods):
            for category, category_methods in categories.items():
                if method in category_methods:
                    bars[i].set_color(colors.get(category, 'lightgray'))
                    break
        
        # Correlation plot
        corr_values = [comparison_results['methods'][m]['correlation'] for m in successful_methods
                      if not np.isnan(comparison_results['methods'][m]['correlation'])]
        corr_methods = [m for m in successful_methods 
                       if not np.isnan(comparison_results['methods'][m]['correlation'])]
        
        plt.subplot(2, 1, 2)
        bars = plt.bar(range(len(corr_methods)), corr_values)
        plt.xticks(range(len(corr_methods)), corr_methods, rotation=45, ha='right')
        plt.title(f'Correlation between TensorFlow and PyTorch - {len(corr_methods)} Methods')
        plt.ylabel('Correlation (higher is better)')
        plt.ylim(-1, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Color bars by category
        for i, method in enumerate(corr_methods):
            for category, category_methods in categories.items():
                if method in category_methods:
                    bars[i].set_color(colors.get(category, 'lightgray'))
                    break
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=category.replace('_', ' ').title()) 
                          for category, color in colors.items() 
                          if any(any(m in methods for m in categories.get(category, [])) for methods in [successful_methods, corr_methods])]
        plt.figlegend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Similarity metrics visualization saved to {os.path.join(OUTPUT_DIR, 'method_comparison_metrics.png')}")
        
    print(f"\nFinal Summary:")
    print(f"- Tested {len(common_methods)} common methods")
    print(f"- TensorFlow success rate: {tf_successes}/{len(common_methods)} ({tf_successes/len(common_methods)*100:.1f}%)")
    print(f"- PyTorch success rate: {pt_successes}/{len(common_methods)} ({pt_successes/len(common_methods)*100:.1f}%)")
    print(f"- Both frameworks success rate: {both_successes}/{len(common_methods)} ({both_successes/len(common_methods)*100:.1f}%)")
    
    return comparison_results


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        if physical_devices:
            for device in physical_devices: tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(f"TF GPU Memory Growth Error: {e} (GPU already initialized differently?).")
    except Exception as e:
        print(f"An unexpected error occurred during TF GPU setup: {e}")

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR); print(f"Created output directory: {OUTPUT_DIR}")
    main()
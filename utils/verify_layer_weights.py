# utils/verify_layer_weights.py
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse

# --- Dynamically add the project root to sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of dynamic import path setup ---

MODEL_CONFIGS = {
    'vgg16': {
        'pt_model_module_name': 'VGG16',  # Module filename (VGG16.py)
        'pt_model_class_name': 'VGG16_PyTorch',  # Class name inside the module
        'pt_weights_file': 'vgg16_ported_weights.pt',
        'default_num_classes': 1000,
        'default_input_channels': 3,
    },
    'ecg': {
        'pt_model_module_name': 'ecg_model',
        'pt_model_class_name': 'ECG_PyTorch',
        'pt_weights_file': 'ecg_ported_weights.pt',
        'default_num_classes': 3,
        'default_input_channels': 1,
    },
    'avb_ecg': {
        'pt_model_module_name': 'pathology_ecg_model',
        'pt_model_class_name': 'Pathology_ECG_PyTorch',
        'pt_weights_file': 'AVB/avb_ported_weights.pt',
        'default_num_classes': 2,
        'default_input_channels': 12,
    },
    'isch_ecg': {
        'pt_model_module_name': 'pathology_ecg_model',
        'pt_model_class_name': 'Pathology_ECG_PyTorch',
        'pt_weights_file': 'ISCH/isch_ported_weights.pt',
        'default_num_classes': 2,
        'default_input_channels': 12,
    },
    'lbbb_ecg': {
        'pt_model_module_name': 'pathology_ecg_model',
        'pt_model_class_name': 'Pathology_ECG_PyTorch',
        'pt_weights_file': 'LBBB/lbbb_ported_weights.pt',
        'default_num_classes': 2,
        'default_input_channels': 12,
    },
    'rbbb_ecg': {
        'pt_model_module_name': 'pathology_ecg_model',
        'pt_model_class_name': 'Pathology_ECG_PyTorch',
        'pt_weights_file': 'RBBB/rbbb_ported_weights.pt',
        'default_num_classes': 2,
        'default_input_channels': 12,
    }
}


def add_pytorch_model_def_path(model_type):
    """Adds the correct PyTorch model definition directory to sys.path."""
    if model_type.upper() == 'VGG16':
        model_def_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'VGG16')
    elif model_type.upper().endswith('ECG'):
        model_def_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'ECG')
    else:
        raise ValueError(f"Model definition path not configured for model_type: {model_type}")

    if model_def_dir not in sys.path:
        sys.path.insert(0, model_def_dir)
    return model_def_dir


def get_pytorch_layer(model, access_string: str):
    parts = access_string.split('.')
    obj = model
    for part in parts:
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (TypeError, IndexError, ValueError) as e:  # Added ValueError
                raise AttributeError(
                    f"Could not access index '{part}' in sequence for '{access_string}'. Object: {type(obj)}. Error: {e}")
        else:
            try:
                obj = getattr(obj, part)
            except AttributeError as e:
                raise AttributeError(
                    f"Could not access attribute '{part}' for '{access_string}'. Object: {type(obj)}. Error: {e}")
    return obj


def verify_weights(args):
    print(
        f"--- Verifying Weights for Model: {args.model_type}, TF Layer: {args.tf_layer_name}, PT Layer Path: {args.pt_layer_access}, Layer Type: {args.layer_type} ---")

    if args.model_type not in MODEL_CONFIGS:
        print(f"Error: Configuration for model_type '{args.model_type}' not found.")
        return

    config = MODEL_CONFIGS[args.model_type]

    try:
        add_pytorch_model_def_path(args.model_type)
        pt_module_imported = __import__(config['pt_model_module_name'])
        PyTorchModelClass = getattr(pt_module_imported, config['pt_model_class_name'])
    except ImportError as e:
        print(
            f"Error importing PyTorch model class {config['pt_model_class_name']} from module {config['pt_model_module_name']}: {e}")
        print(f"Current sys.path: {sys.path}")
        return
    except AttributeError as e:
        print(f"Error: Class {config['pt_model_class_name']} not found in module {config['pt_model_module_name']}. {e}")
        return

    if args.model_type == 'vgg16':
        pt_weights_base_dir = os.path.join(project_root, 'examples/data/models/pytorch/VGG16')
    else:  # ECG models
        pt_weights_base_dir = os.path.join(project_root, 'examples/data/models/pytorch/ECG')

    pytorch_weights_path = os.path.join(pt_weights_base_dir, config['pt_weights_file'])

    tf_kernel_filename = f"{args.tf_layer_name}_tf_kernel.npy"
    tf_bias_filename = f"{args.tf_layer_name}_tf_bias.npy"
    tf_kernel_path = os.path.join(project_root, args.tf_weights_dir, tf_kernel_filename)
    tf_bias_path = os.path.join(project_root, args.tf_weights_dir, tf_bias_filename)

    try:
        num_classes = args.num_classes if args.num_classes is not None else config['default_num_classes']
        init_args = {'num_classes': num_classes}
        if 'default_input_channels' in config:
            input_channels = args.input_channels if args.input_channels is not None else config[
                'default_input_channels']
            init_args['input_channels'] = input_channels

        if args.model_type == 'vgg16' and 'input_channels' in init_args:
            try:
                pt_model = PyTorchModelClass(**init_args)
            except TypeError:
                print("VGG16_PyTorch constructor might not accept 'input_channels', retrying without it.")
                del init_args['input_channels']
                pt_model = PyTorchModelClass(**init_args)
        else:
            pt_model = PyTorchModelClass(**init_args)

        pt_model.load_state_dict(torch.load(pytorch_weights_path, map_location=torch.device('cpu')))
        pt_model.eval()
        print(f"Successfully loaded PyTorch model from: {pytorch_weights_path} with args {init_args}")
    except Exception as e:
        print(f"Error loading PyTorch model or weights: {e}")
        import traceback;
        traceback.print_exc();
        return

    try:
        pt_layer = get_pytorch_layer(pt_model, args.pt_layer_access)
        if not hasattr(pt_layer, 'weight') or pt_layer.weight is None:
            print(f"Error: PyTorch layer at '{args.pt_layer_access}' does not have a 'weight' attribute or it's None.")
            return
        pt_kernel_np = pt_layer.weight.data.cpu().numpy()
        print(f"PyTorch layer '{args.pt_layer_access}' kernel shape: {pt_kernel_np.shape}")
        pt_bias_np = None
        if hasattr(pt_layer, 'bias') and pt_layer.bias is not None:
            pt_bias_np = pt_layer.bias.data.cpu().numpy()
            print(f"PyTorch layer '{args.pt_layer_access}' bias shape: {pt_bias_np.shape}")
        else:
            print(f"PyTorch layer '{args.pt_layer_access}' has no bias or bias is None.")
    except AttributeError as e:
        print(f"Error accessing PyTorch layer '{args.pt_layer_access}': {e}")
        return
    except Exception as e:
        print(f"Error extracting weights from PyTorch model: {e}")
        return

    try:
        if not os.path.exists(tf_kernel_path):
            print(f"Error: TensorFlow kernel file not found: {tf_kernel_path}")
            return
        tf_kernel_np = np.load(tf_kernel_path)
        print(
            f"Successfully loaded TF kernel for '{args.tf_layer_name}' from: {tf_kernel_path} (shape: {tf_kernel_np.shape})")
        tf_bias_np = None
        if os.path.exists(tf_bias_path):
            tf_bias_np = np.load(tf_bias_path)
            print(
                f"Successfully loaded TF bias for '{args.tf_layer_name}' from: {tf_bias_path} (shape: {tf_bias_np.shape})")
        else:
            print(f"TF bias file not found: {tf_bias_path}. Assuming layer has no bias or bias was not saved.")
    except Exception as e:
        print(f"Error loading TensorFlow .npy weight files: {e}")
        return

    tf_kernel_transposed_np = None
    # Normalize layer_type from argparse before comparison
    normalized_layer_type = args.layer_type.upper()

    if normalized_layer_type == 'CONV2D':  # Fixed: Compare with uppercase
        if tf_kernel_np.ndim == 4:
            tf_kernel_transposed_np = tf_kernel_np.transpose(3, 2, 0, 1)
    elif normalized_layer_type == 'CONV1D':  # Fixed: Compare with uppercase
        if tf_kernel_np.ndim == 3:
            tf_kernel_transposed_np = tf_kernel_np.transpose(2, 1, 0)
    elif normalized_layer_type == 'DENSE':  # Fixed: Compare with uppercase
        if tf_kernel_np.ndim == 2:
            tf_kernel_transposed_np = tf_kernel_np.T
    else:
        # This should not be reached if argparse choices are enforced and handled above
        print(
            f"Error: Internal - Mismatch or unknown normalized_layer_type '{normalized_layer_type}' for transposition.")
        return

    if tf_kernel_transposed_np is None:
        print(
            f"Error: Could not transpose TF kernel for layer_type '{args.layer_type}' with TF shape {tf_kernel_np.shape}")
        return
    print(f"Transposed TF kernel shape for {args.layer_type}: {tf_kernel_transposed_np.shape}")

    comparison_atol = 1e-7
    if pt_kernel_np.shape != tf_kernel_transposed_np.shape:
        print(
            f"\nFAILURE: Kernel shapes for {args.tf_layer_name} -> {args.pt_layer_access} do NOT match after transposition!")
        print(f"  PyTorch kernel shape: {pt_kernel_np.shape}")
        print(f"  Transposed TF kernel shape: {tf_kernel_transposed_np.shape}")
    elif np.allclose(pt_kernel_np, tf_kernel_transposed_np, atol=comparison_atol):
        print(
            f"\nSUCCESS: Kernel weights for {args.tf_layer_name} -> {args.pt_layer_access} are very close (atol={comparison_atol})!")
    else:
        kernel_diff = np.abs(pt_kernel_np - tf_kernel_transposed_np)
        print(
            f"\nFAILURE: Kernel weights for {args.tf_layer_name} -> {args.pt_layer_access} are NOT close enough (atol={comparison_atol}).")
        print(f"  Max absolute difference in kernels: {np.max(kernel_diff)}")
        print(f"  Mean absolute difference in kernels: {np.mean(kernel_diff)}")

    if pt_bias_np is not None and tf_bias_np is not None:
        if pt_bias_np.shape != tf_bias_np.shape:
            print(f"\nFAILURE: Bias shapes for {args.tf_layer_name} -> {args.pt_layer_access} do NOT match!")
            print(f"  PyTorch bias shape: {pt_bias_np.shape}")
            print(f"  TF bias shape: {tf_bias_np.shape}")
        elif np.allclose(pt_bias_np, tf_bias_np, atol=comparison_atol):
            print(
                f"\nSUCCESS: Bias weights for {args.tf_layer_name} -> {args.pt_layer_access} are very close (atol={comparison_atol})!")
        else:
            bias_diff = np.abs(pt_bias_np - tf_bias_np)
            print(
                f"\nFAILURE: Bias weights for {args.tf_layer_name} -> {args.pt_layer_access} are NOT close enough (atol={comparison_atol}).")
            print(f"  Max absolute difference in biases: {np.max(bias_diff)}")
            print(f"  Mean absolute difference in biases: {np.mean(bias_diff)}")
    elif pt_bias_np is None and tf_bias_np is not None:
        print(
            f"\nINFO: PyTorch layer {args.pt_layer_access} has no bias, but TF layer {args.tf_layer_name} does. TF bias not compared.")
    elif pt_bias_np is not None and tf_bias_np is None:
        print(
            f"\nINFO: TF layer {args.tf_layer_name} has no (saved) bias, but PyTorch layer {args.pt_layer_access} does. PyTorch bias not compared with TF.")
    else:
        print(
            f"\nINFO: Neither PyTorch layer {args.pt_layer_access} nor TF layer {args.tf_layer_name} have (saved) biases to compare.")

    print("\n--- Verification Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify weights between a TensorFlow layer and a PyTorch layer.")
    parser.add_argument('--model_type', type=str.lower, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Type of the model architecture (e.g., 'vgg16', 'avb_ecg').")
    parser.add_argument('--tf_layer_name', type=str, required=True,
                        help="Name of the TensorFlow layer (used to find .npy weight files).")
    parser.add_argument('--pt_layer_access', type=str, required=True,
                        help="Dot-separated path to access the PyTorch layer (e.g., 'features.0', 'conv3').")
    parser.add_argument('--layer_type', type=str.upper, required=True,
                        choices=['CONV2D', 'CONV1D', 'DENSE'],  # Choices are uppercase
                        help="Type of the layer (Conv2D, Conv1D, Dense) for correct weight transposition.")
    parser.add_argument('--tf_weights_dir', type=str, default='utils/output',
                        help="Directory where TF .npy weight files are stored (relative to project root).")
    parser.add_argument('--num_classes', type=int, default=None,
                        help="Optional: Override number of output classes for PyTorch model instantiation.")
    parser.add_argument('--input_channels', type=int, default=None,
                        help="Optional: Override number of input channels for PyTorch model instantiation.")

    args = parser.parse_args()
    verify_weights(args)
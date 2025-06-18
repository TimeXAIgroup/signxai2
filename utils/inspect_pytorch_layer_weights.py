import torch
import numpy as np
import os
import sys
import argparse

# --- Define paths relative to the project root ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))


# Helper to add PyTorch model definition paths
def add_pytorch_model_def_path():
    ecg_base_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'ECG')
    if ecg_base_dir not in sys.path:
        sys.path.insert(0, ecg_base_dir)
    # This print can be moved to main if preferred
    # print(f"Ensured {ecg_base_dir} is in sys.path for PyTorch ECG model definitions.")


def inspect_pytorch_layer_weights(model_type, pytorch_weights_path, layer_attribute_path,
                                  output_prefix, output_dir, num_samples=10,
                                  input_channels_override=None, num_classes_override=None):
    """
    Inspects and saves weights of a specific layer from a ported PyTorch model.
    """
    print(f"\n--- Inspecting PyTorch Layer Weights ---")
    print(f"Model Type: {model_type}")
    print(f"PyTorch Weights Path: {pytorch_weights_path}")
    print(f"Layer Attribute Path: {layer_attribute_path}")
    print(f"Output Prefix: {output_prefix}")
    print(f"Output Directory: {output_dir}")

    add_pytorch_model_def_path()
    PT_Model_Class = None
    pt_init_args = {}

    # --- Instantiate PyTorch Model ---
    # Set defaults and allow overrides
    if model_type == 'ecg':
        from ecg_model import ECG_PyTorch
        PT_Model_Class = ECG_PyTorch
        pt_init_args['input_channels'] = input_channels_override if input_channels_override is not None else 1
        pt_init_args['num_classes'] = num_classes_override if num_classes_override is not None else 3
    elif model_type in ['avb_ecg', 'isch_ecg', 'lbbb_ecg', 'rbbb_ecg']:
        try:
            from pathology_ecg_model import Pathology_ECG_PyTorch  # Assumes this is your class name
        except ImportError as e:
            print(f"ERROR: Could not import Pathology_ECG_PyTorch from pathology_ecg_model.py: {e}")
            print("Ensure the file exists in examples/data/models/pytorch/ECG/ and class name is correct.")
            return
        PT_Model_Class = Pathology_ECG_PyTorch
        pt_init_args['input_channels'] = input_channels_override if input_channels_override is not None else 12
        pt_init_args['num_classes'] = num_classes_override if num_classes_override is not None else 2
    # Add other model types here if needed (e.g., vgg16)
    # elif model_type == 'vgg16':
    #     from VGG16 import VGG16_PyTorch # Assuming VGG16.py is in a findable VGG16 folder
    #     PT_Model_Class = VGG16_PyTorch
    #     pt_init_args['num_classes'] = num_classes_override if num_classes_override is not None else 1000
    #     # VGG16 usually has fixed input_channels=3, handle if your class takes it
    else:
        print(f"ERROR: Unsupported model_type '{model_type}' for PyTorch model instantiation.")
        return

    if not os.path.exists(pytorch_weights_path):
        print(f"ERROR: Ported PyTorch weights not found at {pytorch_weights_path}")
        return

    try:
        pt_model = PT_Model_Class(**pt_init_args)
        pt_model.load_state_dict(torch.load(pytorch_weights_path, map_location=torch.device('cpu')))
        pt_model.eval()
        print(f"PyTorch model '{PT_Model_Class.__name__}' loaded with ported weights and args {pt_init_args}.")
    except Exception as e:
        print(f"Error loading PyTorch model or weights: {e}")
        import traceback;
        traceback.print_exc()
        return

    # --- Access and Inspect Layer ---
    try:
        # For simple attribute paths like 'conv1' or 'fc_out'
        # For nested: 'features.0', this would need functools.reduce or iterative getattr
        target_layer = pt_model
        for attr_name in layer_attribute_path.split('.'):  # Simple dot notation for access
            target_layer = getattr(target_layer, attr_name)

        print(f"\n--- PyTorch Layer '{layer_attribute_path}' Weights ---")

        if hasattr(target_layer, 'weight') and target_layer.weight is not None:
            weights = target_layer.weight.data.cpu().numpy()
            print(f"  Kernel/Weight shape: {weights.shape}")
            print(f"  Kernel/Weight mean: {np.mean(weights):.6f}, std: {np.std(weights):.6f}")
            print(f"  Kernel/Weight min: {np.min(weights):.6f}, max: {np.max(weights):.6f}")
            print(f"  Kernel/Weight (first {num_samples} flattened values): {weights.flatten()[:num_samples]}")

            os.makedirs(output_dir, exist_ok=True)
            kernel_filename_pt = os.path.join(output_dir, f"{output_prefix}_kernel.npy")
            np.save(kernel_filename_pt, weights)
            print(f"  PyTorch Kernel/Weight saved to: {kernel_filename_pt}")
        else:
            print(f"  Layer {layer_attribute_path} has no 'weight' attribute or it is None.")

        if hasattr(target_layer, 'bias') and target_layer.bias is not None:
            bias = target_layer.bias.data.cpu().numpy()
            print(f"  Bias shape: {bias.shape}")
            print(f"  Bias mean: {np.mean(bias):.6f}, std: {np.std(bias):.6f}")
            print(f"  Bias min: {np.min(bias):.6f}, max: {np.max(bias):.6f}")
            print(f"  Bias (first {num_samples} values): {bias.flatten()[:num_samples]}")

            os.makedirs(output_dir, exist_ok=True)
            bias_filename_pt = os.path.join(output_dir, f"{output_prefix}_bias.npy")
            np.save(bias_filename_pt, bias)
            print(f"  PyTorch Bias saved to: {bias_filename_pt}")
        else:
            print(f"  Layer {layer_attribute_path} has no 'bias' attribute or it is None.")

    except AttributeError:
        print(f"ERROR: Could not access layer or weights at '{layer_attribute_path}' in the PyTorch model.")
    except Exception as e:
        print(f"An error occurred during weight inspection: {e}")
        import traceback;
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect weights of a specific layer in a ported PyTorch model.")
    parser.add_argument('--model_type', type=str.lower, required=True,
                        choices=['ecg', 'avb_ecg', 'isch_ecg', 'lbbb_ecg', 'rbbb_ecg', 'vgg16'],  # Add more as needed
                        help="Type of the PyTorch model architecture.")
    parser.add_argument('--pytorch_weights_path', type=str, required=True,
                        help="Path to the .pt file containing the model's state_dict.")
    parser.add_argument('--layer_attribute_path', type=str, required=True,
                        help="Path to the layer attribute (e.g., 'conv1', 'fc_out', or 'features.0' for nested).")
    parser.add_argument('--output_prefix', type=str, required=True,
                        help="Prefix for the output .npy files (e.g., 'lbbb_conv1d_3_pt').")
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, "utils/output"),
                        help="Directory to save the .npy files.")
    parser.add_argument('--num_samples', type=int, default=10,
                        help="Number of sample weight values to print.")
    parser.add_argument('--input_channels_override', type=int,
                        help="Optional: Override number of input channels for PyTorch model.")
    parser.add_argument('--num_classes_override', type=int,
                        help="Optional: Override number of output classes for PyTorch model.")
    args = parser.parse_args()

    # Construct absolute paths if relative paths are given
    abs_pytorch_weights_path = os.path.join(project_root, args.pytorch_weights_path) if not os.path.isabs(
        args.pytorch_weights_path) else args.pytorch_weights_path
    abs_output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(
        args.output_dir) else args.output_dir

    inspect_pytorch_layer_weights(
        model_type=args.model_type,
        pytorch_weights_path=abs_pytorch_weights_path,
        layer_attribute_path=args.layer_attribute_path,
        output_prefix=args.output_prefix,
        output_dir=abs_output_dir,
        num_samples=args.num_samples,
        input_channels_override=args.input_channels_override,
        num_classes_override=args.num_classes_override
    )
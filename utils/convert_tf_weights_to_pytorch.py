import tensorflow as tf
from tensorflow.keras.models import model_from_json
import torch
import numpy as np
import sys
import os
import argparse

# --- Global project root (defined once) ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_DIR, '..'))


def add_pytorch_model_path(model_type_for_path):
    if model_type_for_path.upper() == 'VGG16':
        model_def_dir = os.path.join(PROJECT_ROOT, 'examples', 'data', 'models', 'pytorch', 'VGG16')
    elif model_type_for_path.upper().endswith('ECG'):
        model_def_dir = os.path.join(PROJECT_ROOT, 'examples', 'data', 'models', 'pytorch', 'ECG')
    else:
        model_def_dir = os.path.join(PROJECT_ROOT, 'examples', 'data', 'models', 'pytorch', model_type_for_path.upper())
        print(f"Warning: Using default path resolution for model type {model_type_for_path}")

    if model_def_dir not in sys.path:
        sys.path.insert(0, model_def_dir)
    print(f"Ensured {model_def_dir} is in sys.path for importing PyTorch model definitions.")
    return model_def_dir


# print_weight_debug_info function (remains the same as you provided)
def print_weight_debug_info(tf_layer_name, tf_weights_np, tf_bias_np,
                            pt_module_name, pt_module,
                            pt_converted_weights_tensor, pt_converted_bias_tensor,
                            print_values=False, num_values_to_print=5):
    print(f"  TF Layer: {tf_layer_name}")
    print(f"    Keras weights shape: {tf_weights_np.shape if tf_weights_np is not None else 'N/A'}")
    if tf_bias_np is not None:
        print(f"    Keras bias shape: {tf_bias_np.shape}")
    else:
        print(f"    Keras bias: None")

    pt_module_str = str(pt_module_name)
    print(f"  PyTorch Module: {pt_module_str}")

    if hasattr(pt_module, 'weight') and pt_module.weight is not None:
        print(f"    PyTorch expected weights shape: {pt_module.weight.shape}")
    else:
        print(f"    PyTorch module '{pt_module_str}' has no 'weight' parameter or it's None.")

    if hasattr(pt_module, 'bias') and pt_module.bias is not None:
        print(f"    PyTorch expected bias shape: {pt_module.bias.shape}")
    else:
        print(f"    PyTorch module '{pt_module_str}' has no 'bias' parameter or it's None.")

    if pt_converted_weights_tensor is not None:
        print(
            f"    Converted PyTorch weights tensor shape (after permute/transpose): {pt_converted_weights_tensor.shape}")
    else:
        print(f"    No weights tensor converted for PyTorch (check TF layer).")

    if pt_converted_bias_tensor is not None:
        print(f"    Converted PyTorch bias tensor shape: {pt_converted_bias_tensor.shape}")
    else:
        print(
            f"    No bias tensor converted for PyTorch (TF layer might not have had a bias, or PT module doesn't expect one).")

    if hasattr(pt_module, 'weight') and pt_module.weight is not None and \
            pt_converted_weights_tensor is not None and \
            pt_module.weight.shape != pt_converted_weights_tensor.shape:
        print(
            f"    !!!!!! FATAL SHAPE MISMATCH (Weights) for {tf_layer_name} -> {pt_module_str}! PyTorch Expected: {pt_module.weight.shape}, Converted: {pt_converted_weights_tensor.shape}")

    pt_has_bias_attr = hasattr(pt_module, 'bias')
    pt_bias_is_not_none = pt_has_bias_attr and pt_module.bias is not None

    if pt_bias_is_not_none and pt_converted_bias_tensor is not None:
        if pt_module.bias.shape != pt_converted_bias_tensor.shape:
            print(
                f"    !!!!!! FATAL SHAPE MISMATCH (Bias) for {tf_layer_name} -> {pt_module_str}! PyTorch Expected: {pt_module.bias.shape}, Converted: {pt_converted_bias_tensor.shape}")
    elif not pt_bias_is_not_none and pt_converted_bias_tensor is not None:
        print(
            f"    !!!!!! WARNING: PyTorch module {pt_module_str} has no bias parameter, but bias tensor was converted from TF layer {tf_layer_name}. Bias will NOT be assigned.")
    elif pt_bias_is_not_none and pt_converted_bias_tensor is None and tf_bias_np is not None:
        print(
            f"    !!!!!! WARNING: PyTorch module {pt_module_str} expects bias, TF layer {tf_layer_name} had bias, but no bias tensor was converted for PyTorch. PT bias will remain uninitialized from TF.")

    if print_values:
        if tf_weights_np is not None and pt_converted_weights_tensor is not None:
            print(
                f"    TF Weights (numpy, sample, first {num_values_to_print} flat values): {tf_weights_np.flatten()[:num_values_to_print]}")
            print(
                f"    PT Converted Weights Tensor (sample, before assignment, first {num_values_to_print} flat values): {pt_converted_weights_tensor.detach().cpu().numpy().flatten()[:num_values_to_print]}")
        if tf_bias_np is not None and pt_converted_bias_tensor is not None:
            print(
                f"    TF Bias (numpy, sample, first {num_values_to_print} flat values): {tf_bias_np.flatten()[:num_values_to_print]}")
            print(
                f"    PT Converted Bias Tensor (sample, before assignment, first {num_values_to_print} flat values): {pt_converted_bias_tensor.detach().cpu().numpy().flatten()[:num_values_to_print]}")


def get_model_specific_config(model_type, pytorch_model_instance):
    """Returns layer_map and layers_to_print_values_for based on model_type."""
    if model_type == 'vgg16':
        layer_map = {  # VGG16 specific map
            'block1_conv1': pytorch_model_instance.features[0], 'block1_conv2': pytorch_model_instance.features[2],
            'block2_conv1': pytorch_model_instance.features[5], 'block2_conv2': pytorch_model_instance.features[7],
            'block3_conv1': pytorch_model_instance.features[10], 'block3_conv2': pytorch_model_instance.features[12],
            'block3_conv3': pytorch_model_instance.features[14],
            'block4_conv1': pytorch_model_instance.features[17], 'block4_conv2': pytorch_model_instance.features[19],
            'block4_conv3': pytorch_model_instance.features[21],
            'block5_conv1': pytorch_model_instance.features[24], 'block5_conv2': pytorch_model_instance.features[26],
            'block5_conv3': pytorch_model_instance.features[28],
            'fc1': pytorch_model_instance.classifier[0], 'fc2': pytorch_model_instance.classifier[2],
            'predictions': pytorch_model_instance.classifier[4]
        }
        layers_to_print_values_for = ['block1_conv1', 'fc1', 'predictions']
        return layer_map, layers_to_print_values_for
    elif model_type == 'ecg':  # For the original 3-class ECG model with Flatten
        layer_map = {
            'conv1d': pytorch_model_instance.conv1,
            'conv1d_1': pytorch_model_instance.conv2,
            'last_conv': pytorch_model_instance.conv3,
            'dense': pytorch_model_instance.fc1,
            'dense_1': pytorch_model_instance.fc2
        }
        layers_to_print_values_for = ['conv1d', 'dense_1']
        return layer_map, layers_to_print_values_for
    elif model_type in ['avb_ecg', 'isch_ecg', 'lbbb_ecg', 'rbbb_ecg']:  # For shared pathology architecture
        # Assumes TF layer names are consistent across AVB, ISCH, LBBB, RBBB
        # and map to attributes in Pathology_ECG_PyTorch
        layer_map = {
            'conv1d': pytorch_model_instance.conv1,
            'conv1d_1': pytorch_model_instance.conv2,
            'conv1d_2': pytorch_model_instance.conv3,
            'conv1d_3': pytorch_model_instance.conv4,
            'conv1d_4': pytorch_model_instance.conv5,
            'dense': pytorch_model_instance.fc1,
            'dense_1': pytorch_model_instance.fc2,
            'dense_2': pytorch_model_instance.fc3,
            'dense_3': pytorch_model_instance.fc_out,
        }
        layers_to_print_values_for = ['conv1d', 'dense_3']
        return layer_map, layers_to_print_values_for
    else:
        raise ValueError(f"Unsupported model_type for config: {model_type}")


def load_and_convert_tf_model_weights(tf_model, pytorch_model_instance, model_type):
    # This function (core weight transfer logic) remains the same as your provided version
    print(f"Starting weight transfer for {model_type} model.")
    pytorch_model_instance.eval()

    layer_map, layers_to_print_values_for = get_model_specific_config(model_type, pytorch_model_instance)

    for tf_layer_name, pt_module in layer_map.items():
        try:
            tf_layer = tf_model.get_layer(name=tf_layer_name)
        except ValueError:
            print(f"SKIPPING: TensorFlow layer '{tf_layer_name}' not found in loaded Keras model.")
            continue

        tf_layer_weights_list = tf_layer.get_weights()
        if not tf_layer_weights_list:
            print(f"SKIPPING: TensorFlow layer '{tf_layer_name}' has no weights.")
            continue

        pt_module_name_for_debug = f"{type(pt_module).__name__} (TF: {tf_layer_name})"
        print(f"\nProcessing Layer: TF '{tf_layer_name}' -> PyTorch Module {pt_module_name_for_debug}")

        tf_weights_np = tf_layer_weights_list[0]
        tf_bias_np = tf_layer_weights_list[1] if len(tf_layer_weights_list) > 1 else None

        pt_converted_weights_tensor = None
        pt_converted_bias_tensor = None
        should_print_values = tf_layer_name in layers_to_print_values_for

        if isinstance(pt_module, torch.nn.Conv2d):
            pt_converted_weights_tensor = torch.from_numpy(tf_weights_np).permute(3, 2, 0, 1).contiguous()
        elif isinstance(pt_module, torch.nn.Conv1d):
            pt_converted_weights_tensor = torch.from_numpy(tf_weights_np.transpose(2, 1, 0)).contiguous()
        elif isinstance(pt_module, torch.nn.Linear):
            pt_converted_weights_tensor = torch.from_numpy(tf_weights_np.T).contiguous()
        else:
            print(
                f"SKIPPING: PyTorch module type {type(pt_module)} not handled for TF layer '{tf_layer_name}'.")
            continue

        if tf_bias_np is not None:
            pt_converted_bias_tensor = torch.from_numpy(tf_bias_np).contiguous()

        print_weight_debug_info(tf_layer_name, tf_weights_np, tf_bias_np,
                                pt_module_name_for_debug, pt_module,
                                pt_converted_weights_tensor, pt_converted_bias_tensor,
                                print_values=should_print_values)

        if hasattr(pt_module, 'weight') and pt_module.weight is not None and pt_converted_weights_tensor is not None:
            pt_module.weight.data.copy_(pt_converted_weights_tensor)
        elif hasattr(pt_module, 'weight') and pt_module.weight is not None and pt_converted_weights_tensor is None:
            print(
                f"    WARNING: PyTorch module '{pt_module_name_for_debug}' has weight, but no weights converted from TF '{tf_layer_name}'.")

        if hasattr(pt_module, 'bias') and pt_module.bias is not None and \
                tf_bias_np is not None and pt_converted_bias_tensor is not None:
            pt_module.bias.data.copy_(pt_converted_bias_tensor)
        elif hasattr(pt_module, 'bias') and pt_module.bias is not None and tf_bias_np is None:
            print(
                f"    INFO: TF layer '{tf_layer_name}' had no bias. PT module '{pt_module_name_for_debug}' bias remains initialized by PyTorch.")

        if should_print_values:
            if hasattr(pt_module, 'weight') and pt_module.weight is not None:
                print(
                    f"    PT Module Weights (sample, AFTER assignment): {pt_module.weight.data.detach().cpu().numpy().flatten()[:5]}")
            if hasattr(pt_module, 'bias') and pt_module.bias is not None and tf_bias_np is not None:
                print(
                    f"    PT Module Bias (sample, AFTER assignment): {pt_module.bias.data.detach().cpu().numpy().flatten()[:5]}")
        print(f"  Successfully assigned weights/bias for TF:'{tf_layer_name}' to PT:'{pt_module_name_for_debug}'")

    print("\nWeight transfer completed.")
    return pytorch_model_instance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert TensorFlow Keras model weights to PyTorch.")
    parser.add_argument('--model_type', type=str.lower, required=True,
                        choices=['vgg16', 'ecg', 'avb_ecg', 'isch_ecg', 'lbbb_ecg', 'rbbb_ecg'],  # Expanded choices
                        help="Type of the model architecture to convert.")
    parser.add_argument('--tf_model_h5_path', type=str,
                        help="Path to TensorFlow Keras model file (.h5, arch + weights). For 'ecg', 'vgg16'.")
    parser.add_argument('--tf_json_path', type=str,
                        help="Path to TensorFlow Keras model architecture file (.json). For 'avb_ecg', etc.")
    parser.add_argument('--tf_weights_h5_path', type=str,
                        help="Path to TensorFlow Keras model weights file (.h5, use with --tf_json_path).")
    parser.add_argument('--pytorch_save_path', type=str, required=True,
                        help="Full path to save the PyTorch ported model's state_dict (.pt file).")
    parser.add_argument('--num_classes_override', type=int,
                        help="Optional: Override number of output classes for PyTorch model.")
    parser.add_argument('--input_channels_override', type=int,
                        help="Optional: Override number of input channels for PyTorch model.")
    args = parser.parse_args()

    tf.config.set_visible_devices([], 'GPU')

    # --- Load TensorFlow Model ---
    tf_model = None
    # Construct absolute paths for TF model files
    abs_tf_json_path = os.path.join(PROJECT_ROOT, args.tf_json_path) if args.tf_json_path and not os.path.isabs(
        args.tf_json_path) else args.tf_json_path
    abs_tf_weights_h5_path = os.path.join(PROJECT_ROOT,
                                          args.tf_weights_h5_path) if args.tf_weights_h5_path and not os.path.isabs(
        args.tf_weights_h5_path) else args.tf_weights_h5_path
    abs_tf_model_h5_path = os.path.join(PROJECT_ROOT,
                                        args.tf_model_h5_path) if args.tf_model_h5_path and not os.path.isabs(
        args.tf_model_h5_path) else args.tf_model_h5_path

    if abs_tf_json_path and abs_tf_weights_h5_path:
        print(f"Loading TensorFlow model from JSON: {abs_tf_json_path} and H5 weights: {abs_tf_weights_h5_path}")
        try:
            with open(abs_tf_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            tf_model = model_from_json(loaded_model_json)
            tf_model.load_weights(abs_tf_weights_h5_path)
            print("TensorFlow model loaded successfully from JSON and H5 weights.")
        except Exception as e:
            print(f"Error loading TF model from JSON/H5: {e}");
            exit(1)
    elif abs_tf_model_h5_path:
        print(f"Loading TensorFlow model from single H5 file: {abs_tf_model_h5_path}")
        try:
            tf_model = tf.keras.models.load_model(abs_tf_model_h5_path)
            print("TensorFlow model loaded successfully from single H5 file.")
        except Exception as e:
            print(f"Error loading TF model from single H5 file: {e}");
            exit(1)
    else:
        print(
            "Error: You must provide TF model path(s): --tf_model_h5_path OR (--tf_json_path AND --tf_weights_h5_path).")
        exit(1)

    # --- Instantiate PyTorch Model ---
    pytorch_model_instance = None
    # Infer parameters from TF model
    try:
        inferred_num_classes = tf_model.output_shape[-1] if isinstance(tf_model.output_shape, tuple) else \
        tf_model.output_shape[0][-1]
        inferred_input_channels = tf_model.input_shape[-1] if isinstance(tf_model.input_shape, tuple) else \
        tf_model.input_shape[0][-1]
    except Exception as e_shape:
        print(f"Warning: Could not reliably infer shapes from TF model: {e_shape}. Relying on overrides or defaults.")
        inferred_num_classes = None
        inferred_input_channels = None

    num_pt_classes = args.num_classes_override if args.num_classes_override is not None else inferred_num_classes
    num_pt_input_channels = args.input_channels_override if args.input_channels_override is not None else inferred_input_channels

    if num_pt_classes is None:
        print("Error: Number of classes for PyTorch model could not be determined. Use --num_classes_override.");
        exit(1)

    # Ensure the base ECG model directory is in sys.path for imports
    add_pytorch_model_path(args.model_type)  # Handles VGG16 path too if 'VGG16' is passed

    if args.model_type == 'ecg':
        from ecg_model import ECG_PyTorch

        if num_pt_input_channels is None: num_pt_input_channels = 1  # Default for original ECG
        pytorch_model_instance = ECG_PyTorch(num_classes=int(num_pt_classes), input_channels=int(num_pt_input_channels))

    elif args.model_type in ['avb_ecg', 'isch_ecg', 'lbbb_ecg', 'rbbb_ecg']:
        try:
            # Assumes pathology_ecg_model.py contains Pathology_ECG_PyTorch class
            # and is directly in examples/data/models/pytorch/ECG/
            from pathology_ecg_model import Pathology_ECG_PyTorch
        except ImportError as e:
            print(f"ImportError for Pathology_ECG_PyTorch: {e}")
            print(
                "Ensure 'pathology_ecg_model.py' exists in '.../pytorch/ECG/' and contains 'Pathology_ECG_PyTorch' class.")
            exit(1)

        if num_pt_input_channels is None: num_pt_input_channels = 12  # Default for these pathology models
        if int(num_pt_classes) != 2:  # These models are binary classification
            print(
                f"Warning: TF model output units ({num_pt_classes}) for {args.model_type} differ from expected 2. Using {num_pt_classes}.")
        pytorch_model_instance = Pathology_ECG_PyTorch(num_classes=int(num_pt_classes),
                                                       input_channels=int(num_pt_input_channels))

    elif args.model_type == 'vgg16':
        from VGG16 import VGG16_PyTorch

        if num_pt_input_channels is None or num_pt_input_channels != 3: num_pt_input_channels = 3
        try:  # VGG16 might not always take input_channels as an argument
            pytorch_model_instance = VGG16_PyTorch(num_classes=int(num_pt_classes),
                                                   input_channels=int(num_pt_input_channels))
        except TypeError:
            pytorch_model_instance = VGG16_PyTorch(num_classes=int(num_pt_classes))
    else:
        # This case should ideally be caught by argparse choices, but as a safeguard:
        print(f"Model type '{args.model_type}' not configured for PyTorch model instantiation.");
        exit(1)

    if pytorch_model_instance is None:
        print(f"Failed to instantiate PyTorch model for type {args.model_type}.");
        exit(1)

    print(
        f"Instantiated PyTorch model '{args.model_type}' with input_channels={num_pt_input_channels}, num_classes={num_pt_classes}")

    # --- Perform Conversion and Save ---
    abs_pytorch_save_path = os.path.join(PROJECT_ROOT, args.pytorch_save_path) if not os.path.isabs(
        args.pytorch_save_path) else args.pytorch_save_path
    os.makedirs(os.path.dirname(abs_pytorch_save_path), exist_ok=True)
    try:
        converted_pytorch_model = load_and_convert_tf_model_weights(tf_model, pytorch_model_instance, args.model_type)
        if converted_pytorch_model:
            torch.save(converted_pytorch_model.state_dict(), abs_pytorch_save_path)
            print(f"\nSuccessfully converted and saved PyTorch model weights to: {abs_pytorch_save_path}")

            # Basic Sanity Check
            print("\n--- Basic Sanity Check ---")
            # Re-instantiate for the check
            test_pt_model_instance = None
            if args.model_type == 'ecg':
                test_pt_model_instance = ECG_PyTorch(num_classes=int(num_pt_classes),
                                                     input_channels=int(num_pt_input_channels))
            elif args.model_type in ['avb_ecg', 'isch_ecg', 'lbbb_ecg', 'rbbb_ecg']:
                test_pt_model_instance = Pathology_ECG_PyTorch(num_classes=int(num_pt_classes),
                                                               input_channels=int(num_pt_input_channels))
            elif args.model_type == 'vgg16':
                try:
                    test_pt_model_instance = VGG16_PyTorch(num_classes=int(num_pt_classes),
                                                           input_channels=int(num_pt_input_channels))
                except TypeError:
                    test_pt_model_instance = VGG16_PyTorch(num_classes=int(num_pt_classes))

            if test_pt_model_instance:
                test_pt_model_instance.load_state_dict(torch.load(abs_pytorch_save_path))
                test_pt_model_instance.eval()
                print(f"PyTorch model '{args.model_type}' with ported weights loaded successfully for basic check.")
            else:
                print(f"Could not re-instantiate PyTorch model for type {args.model_type} for sanity check.")
        else:
            print(f"Weight conversion failed for model {args.model_type}.")
    except Exception as e:
        print(f"An error occurred during conversion or saving: {e}")
        import traceback;

        traceback.print_exc()
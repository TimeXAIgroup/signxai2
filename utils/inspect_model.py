import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
import os
import sys
import argparse

# --- Define paths relative to the project root ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))


def main():
    parser = argparse.ArgumentParser(description="Inspect a TensorFlow Keras model and save its architecture.")
    parser.add_argument('--model_h5_path', type=str,
                        help="Path to the Keras model (.h5 file, contains arch and weights).")
    parser.add_argument('--model_json_path', type=str, help="Path to the Keras model architecture (.json file).")
    parser.add_argument('--model_weights_path', type=str,
                        help="Path to Keras model weights (.h5 file, use with --model_json_path).")
    parser.add_argument('--output_file_path', type=str, required=True,
                        help="Path to save the architecture details text file.")
    parser.add_argument('--inspect_weights_for_layer', type=str, default=None,  # New argument
                        help="Optional: Name of the layer whose weights you want to inspect.")
    parser.add_argument('--num_weight_samples', type=int, default=10,  # New argument
                        help="Number of sample weights to print if inspecting weights (default: 10).")

    args = parser.parse_args()

    model = None
    model_source_info = ""

    def get_abs_path(path_arg):
        if path_arg and not os.path.isabs(path_arg):
            return os.path.join(project_root, path_arg)
        return path_arg

    abs_model_h5_path = get_abs_path(args.model_h5_path)
    abs_model_json_path = get_abs_path(args.model_json_path)
    abs_model_weights_path = get_abs_path(args.model_weights_path)
    abs_output_file_path = get_abs_path(args.output_file_path)

    os.makedirs(os.path.dirname(abs_output_file_path), exist_ok=True)

    try:
        tf_gpus = tf.config.experimental.list_physical_devices('GPU')
        if tf_gpus:
            for gpu in tf_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            print("No GPU found by TensorFlow. Running on CPU.")

        if abs_model_json_path and abs_model_weights_path:
            model_source_info = f"JSON: {abs_model_json_path}, Weights: {abs_model_weights_path}"
            print(f"Loading model from JSON ({abs_model_json_path}) and weights ({abs_model_weights_path})...")
            with open(abs_model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
            model.load_weights(abs_model_weights_path)
            print("Model successfully loaded from JSON and weights.\n")
        elif abs_model_h5_path:
            model_source_info = f"H5 file: {abs_model_h5_path}"
            print(f"Loading model from H5 file: {abs_model_h5_path}...")
            model = tf.keras.models.load_model(abs_model_h5_path)
            print(f"Model successfully loaded from {abs_model_h5_path}.\n")
        else:
            print("Error: Must provide either --model_h5_path OR (--model_json_path AND --model_weights_path).")
            exit(1)
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)

    original_stdout = sys.stdout
    print(f"Saving architecture and weight details to: {abs_output_file_path}")
    with open(abs_output_file_path, 'w') as f:
        sys.stdout = f

        print(f"--- Model loaded from: {model_source_info} ---")
        print("--- Model Summary (model.summary()) ---")
        model.summary()
        print("\n" + "=" * 50 + "\n")

        print("--- Detailed Layer Information (model.layers) ---")
        for i, layer in enumerate(model.layers):
            print(f"Layer {i}: {layer.name} (Type: {layer.__class__.__name__})")
            config = layer.get_config()
            print(f"  Config: {config}")
            try:
                print(f"  Input shape: {layer.input_shape}")
                print(f"  Output shape: {layer.output_shape}")
            except AttributeError:
                print("  Input/Output shape not directly available for this layer type.")

            # Print specific details like before
            if isinstance(layer, tf.keras.layers.Conv1D):
                print(f"  Conv1D - Filters: {layer.filters}")
                print(f"  Conv1D - Kernel Size: {layer.kernel_size[0]}")
                print(f"  Conv1D - Strides: {layer.strides[0]}")
                print(f"  Conv1D - Padding: {layer.padding}")
                print(f"  Conv1D - Activation: {tf.keras.activations.serialize(layer.activation)}")
                print(f"  Conv1D - Data Format: {layer.data_format}")
                if layer.use_bias:
                    print(f"  Conv1D - Bias is used.")
                else:
                    print(f"  Conv1D - No Bias.")
            # ... (add other specific layer details as you had them)

            # *** NEW: Inspect weights for the specified layer ***
            if args.inspect_weights_for_layer and layer.name == args.inspect_weights_for_layer:
                layer_weights = layer.get_weights()
                if layer_weights:
                    print(f"  --- Weights for layer: {layer.name} ---")
                    if len(layer_weights) > 0:  # Kernel weights
                        kernel = layer_weights[0]
                        print(f"    Kernel shape: {kernel.shape}")
                        print(f"    Kernel mean: {np.mean(kernel):.6f}, std: {np.std(kernel):.6f}")
                        print(f"    Kernel min: {np.min(kernel):.6f}, max: {np.max(kernel):.6f}")
                        print(
                            f"    Kernel (first {args.num_weight_samples} flattened values): {kernel.flatten()[:args.num_weight_samples]}")
                        # Optionally save to .npy file
                        kernel_filename = os.path.join(os.path.dirname(abs_output_file_path),
                                                       f"{layer.name}_tf_kernel.npy")
                        np.save(kernel_filename, kernel)
                        print(f"    Kernel saved to: {kernel_filename}")

                    if len(layer_weights) > 1:  # Bias weights
                        bias = layer_weights[1]
                        print(f"    Bias shape: {bias.shape}")
                        print(f"    Bias mean: {np.mean(bias):.6f}, std: {np.std(bias):.6f}")
                        print(f"    Bias min: {np.min(bias):.6f}, max: {np.max(bias):.6f}")
                        print(
                            f"    Bias (first {args.num_weight_samples} values): {bias.flatten()[:args.num_weight_samples]}")
                        bias_filename = os.path.join(os.path.dirname(abs_output_file_path), f"{layer.name}_tf_bias.npy")
                        np.save(bias_filename, bias)
                        print(f"    Bias saved to: {bias_filename}")
                    print(f"  --- End Weights for layer: {layer.name} ---")
                else:
                    print(f"  Layer {layer.name} has no weights to inspect.")
            print("-" * 30)

        print("\n" + "=" * 50 + "\n")
        print("--- Overall Model Input/Output ---")
        # ... (same as before) ...
        try:
            print(f"Model Inputs: {model.inputs}")
            print(f"Model Outputs: {model.outputs}")
        except Exception as e_io:
            print(f"Could not directly retrieve inputs/outputs: {e_io}")

    sys.stdout = original_stdout
    print(f"Model architecture and specified weight details saved to: {abs_output_file_path}")


if __name__ == '__main__':
    main()
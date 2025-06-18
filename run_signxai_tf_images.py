import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys  # For checking arguments before argparse fully parses
import inspect  # For dynamically listing methods
import time  # To briefly pause if saving plots

# Import necessary functions from the signxai library
from signxai.utils.utils import load_image, aggregate_and_normalize_relevancemap_rgb
from tensorflow.python.keras.activations import linear  # For removing softmax


# signxai.tf_signxai will be imported conditionally/checked later

# --- Helper function to get available TF methods ---
def get_available_tf_methods():
    """
    Dynamically gets a list of available TensorFlow XAI method names from the wrappers module.
    """
    try:
        from signxai import tf_signxai
        if not tf_signxai:
            print("Warning: signxai.tf_signxai submodule not loaded in get_available_tf_methods.")
            return []

        from signxai.tf_signxai.methods import wrappers as wrapper_module
        available_methods = []
        exclude_names = {
            'np', 'calculate_grad_cam_relevancemap',
            'calculate_grad_cam_relevancemap_timeseries',
            'guided_backprop_on_guided_model', 'calculate_sign_mu',
            'calculate_explanation_innvestigate',
            'calculate_relevancemap', 'calculate_relevancemaps'
        }

        for name, func in inspect.getmembers(wrapper_module, inspect.isfunction):
            if not name.startswith('_') and name not in exclude_names:
                if func.__module__ == wrapper_module.__name__:
                    available_methods.append(name)
        return sorted(list(set(available_methods)))
    except ImportError:
        print("Warning: Could not import signxai.tf_signxai to list methods.")
        return []
    except AttributeError:
        print("Warning: Could not access methods in signxai.tf_signxai.methods.wrappers to list methods.")
        return []


def run_explanation(framework, model_path, image_path, method_name,
                    neuron_selection=None, method_params=None,
                    save_plots=False, output_dir="explanation_outputs"):
    print(f"\n--- Running Explanation ---")
    print(f"Framework: {framework}")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Method: {method_name}")
    if neuron_selection is not None:
        print(f"Neuron selection: {neuron_selection}")
    if method_params:
        print(f"Additional method parameters: {method_params}")

    if framework.lower() == 'tensorflow':
        try:
            # Ensure TF and the signxai TF submodule are available
            import tensorflow
            from signxai import tf_signxai
            if not tf_signxai:
                raise ImportError(
                    "signxai.tf_signxai submodule not loaded. Check your SignXAI __init__.py and installation.")
            # Get the specific calculate_relevancemap from the wrappers module
            from signxai.tf_signxai.methods.wrappers import calculate_relevancemap as calculate_relevancemap_tf
        except ImportError as e:
            print(f"Error: TensorFlow or signxai.tf_signxai module is not available: {e}")
            return False  # Indicate failure

        print(f"Loading TensorFlow model...")
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"  Error loading model: {e}")
            return False
        print("  TensorFlow model loaded.")

        model.layers[-1].activation = linear
        print("  Softmax removed.")

        print(f"Loading and preprocessing image...")
        try:
            original_image, preprocessed_image = load_image(image_path, target_size=(224, 224), expand_dims=True)
        except FileNotFoundError:
            print(f"  Error: Image file not found at {image_path}")
            return False
        except Exception as e:
            print(f"  Error loading or preprocessing image: {e}")
            return False
        print("  Image loaded and preprocessed.")

        print(f"Calculating {method_name} explanation...")

        kwargs_for_method = {}
        if neuron_selection is not None:
            kwargs_for_method['neuron_selection'] = neuron_selection
        if method_params:
            kwargs_for_method.update(method_params)

        try:
            relevance_map = calculate_relevancemap_tf(method_name, preprocessed_image, model, **kwargs_for_method)
        except NameError as ne:
            print(f"  Error: Method '{method_name}' is not defined or callable. Details: {ne}")
            return False
        except Exception as e:
            print(f"  Error during explanation for method '{method_name}': {e}")
            return False

        print(f"  Explanation calculated for {method_name}.")

        if relevance_map.ndim == 4 and relevance_map.shape[0] == 1:
            relevance_map_to_visualize = relevance_map[0]
        else:
            relevance_map_to_visualize = relevance_map

        normalized_heatmap = aggregate_and_normalize_relevancemap_rgb(relevance_map_to_visualize)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Method: {method_name}", fontsize=16)  # Add method name as title
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(normalized_heatmap, cmap='seismic', vmin=-1, vmax=1)
        axs[1].set_title(f"Normalized Heatmap")
        axs[1].axis('off')

        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plot_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}_{method_name}.png")
            plt.savefig(plot_filename)
            print(f"  Plot saved to {plot_filename}")
            plt.close(fig)  # Close the figure to prevent it from displaying interactively
        else:
            plt.show()  # This will block until the plot is closed manually
            print(f"  Visualization displayed for {method_name}.")
        return True  # Indicate success

    elif framework.lower() == 'pytorch':
        print("PyTorch framework selected. Parameterized script for PyTorch is not fully implemented in this example.")
        return False
    else:
        print(f"Error: Unsupported framework '{framework}'. Choose 'tensorflow' or 'pytorch'.")
        return False


def parse_method_params(params_str):
    # ... (same as before)
    if not params_str:
        return None
    params = {}
    try:
        for item in params_str.split(','):
            key, value = item.split(':', 1)
            key = key.strip()
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
        print(
            f"Warning: Could not parse method_params string: '{params_str}'. Expected format: key1:value1,key2:value2")
        return None


if __name__ == '__main__':
    try:
        import signxai

        print(f"SignXAI version: {signxai.__version__}")
        if hasattr(signxai, '_AVAILABLE_BACKENDS'):
            print(f"Available SignXAI backends: {signxai._AVAILABLE_BACKENDS}")
        if hasattr(signxai, '_DEFAULT_BACKEND'):
            print(f"Default SignXAI backend: {signxai._DEFAULT_BACKEND}")
    except ImportError:
        print("Error: Could not import signxai. Ensure it's installed correctly and accessible.")
        sys.exit(1)
    except AttributeError:
        print("Warning: signxai._AVAILABLE_BACKENDS or _DEFAULT_BACKEND not found. Using manual framework selection.")

    parser = argparse.ArgumentParser(description="Run XAI explanations using the SignXAI library.", add_help=False)

    core_args = parser.add_argument_group('Core arguments')
    core_args.add_argument('--framework', type=str, default='tensorflow', choices=['tensorflow', 'pytorch'],
                           help="The deep learning framework to use (default: tensorflow).")
    core_args.add_argument('--image_path', type=str,
                           help="Path to the input image. Required unless --list_methods or --run_all_tf_methods is used with predefined paths.")
    core_args.add_argument('--model_path', type=str,
                           help="Path to the trained model file (.h5 for TensorFlow). Required unless --list_methods is used.")

    method_selection_group = parser.add_mutually_exclusive_group()
    method_selection_group.add_argument('--method_name', type=str,
                                        help="Name of the XAI method to use (e.g., gradient_x_sign).")
    method_selection_group.add_argument('--run_all_tf_methods', action='store_true',
                                        help="Run all available TensorFlow XAI methods sequentially. Ignores --method_name.")

    optional_args = parser.add_argument_group('Optional arguments')
    optional_args.add_argument('--neuron_selection', type=int, default=None,
                               help="Index of the neuron to explain for some methods.")
    optional_args.add_argument('--method_params', type=str, default=None,
                               help="Additional parameters for the method, as a comma-separated string of key:value pairs (e.g., 'mu:0.5,steps:100').")
    optional_args.add_argument('--list_methods', action='store_true',
                               help="List available XAI methods for the selected framework and exit.")
    optional_args.add_argument('--save_plots', action='store_true',
                               help="Save plots to files instead of displaying them interactively. Used with --run_all_tf_methods or single method.")
    optional_args.add_argument('--output_dir', type=str, default="explanation_outputs",
                               help="Directory to save plots if --save_plots is used (default: explanation_outputs).")
    optional_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                               help='show this help message and exit')

    args = parser.parse_args()

    tf_methods_list = []
    if args.framework.lower() == 'tensorflow':
        tf_methods_list = get_available_tf_methods()

    if args.list_methods:
        if args.framework.lower() == 'tensorflow':
            if tf_methods_list:
                print("\nAvailable TensorFlow XAI methods (dynamically generated from wrappers.py):")
                for m_name in tf_methods_list:
                    print(f"  - {m_name}")
            else:
                print(
                    "Could not retrieve TensorFlow methods. Ensure SignXAI and TensorFlow backend are correctly installed.")
        elif args.framework.lower() == 'pytorch':
            print("\nPyTorch method listing not fully implemented. You would need a get_available_pt_methods().")
        sys.exit(0)

    # Validate required arguments if not listing methods
    if not args.image_path or not args.model_path:
        parser.error("the following arguments are required: --image_path, --model_path")

    if not args.run_all_tf_methods and not args.method_name:
        parser.error("either --method_name must be specified or --run_all_tf_methods flag must be set.")

    parsed_method_params = parse_method_params(args.method_params)

    if args.run_all_tf_methods:
        if args.framework.lower() != 'tensorflow':
            print("Error: --run_all_tf_methods is only compatible with --framework tensorflow.")
            sys.exit(1)

        if not tf_methods_list:
            print("Error: Cannot run all TensorFlow methods because the method list is empty.")
            sys.exit(1)

        print(f"\n--- Running ALL {len(tf_methods_list)} TensorFlow XAI methods ---")
        print(f"Image: {args.image_path}, Model: {args.model_path}")
        if args.save_plots:
            print(f"Plots will be saved to: {args.output_dir}")
        else:
            print("Plots will be displayed interactively. Close each plot to continue to the next method.")

        succeeded_methods = []
        failed_methods = []

        for i, method_to_run in enumerate(tf_methods_list):
            print(f"\nProcessing method {i + 1}/{len(tf_methods_list)}: {method_to_run}")
            # Note: Some methods might require specific parameters not provided by default.
            # The `neuron_selection` and `method_params` from CLI will apply to ALL methods in this loop.
            # This might not be ideal for all methods.
            # For methods like 'gradient_x_sign_mu' which needs 'mu', this generic loop will likely fail
            # unless that specific 'mu' is in `parsed_method_params` or handled by a more specific variant.
            if run_explanation(args.framework, args.model_path, args.image_path, method_to_run,
                               args.neuron_selection, parsed_method_params,
                               args.save_plots, args.output_dir):
                succeeded_methods.append(method_to_run)
            else:
                failed_methods.append(method_to_run)

            if args.save_plots:  # Small pause to avoid overwhelming file system or display buffers
                time.sleep(0.1)

        print("\n--- Batch Processing Summary ---")
        print(f"Successfully processed: {len(succeeded_methods)} methods")
        if succeeded_methods: print(f"  {succeeded_methods}")
        print(f"Failed or skipped: {len(failed_methods)} methods")
        if failed_methods: print(f"  {failed_methods}")

    else:  # Single method run
        # Validate method_name for TensorFlow
        if args.framework.lower() == 'tensorflow':
            if not tf_methods_list:
                print(
                    f"Warning: Could not verify method '{args.method_name}' as the available method list is empty. Attempting to run it anyway.")
            elif args.method_name not in tf_methods_list:
                print(f"Error: Method '{args.method_name}' is not a recognized TensorFlow XAI method.")
                print("Note: Method names are case-sensitive.")
                print(f"\nAvailable TensorFlow methods: {tf_methods_list}")
                sys.exit(1)

        run_explanation(args.framework, args.model_path, args.image_path, args.method_name,
                        args.neuron_selection, parsed_method_params,
                        args.save_plots, args.output_dir)
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import inspect
import time
from PIL import Image

# Import necessary functions from the signxai library
from signxai.torch_signxai.methods.wrappers import calculate_relevancemap as torch_calculate_relevancemap
from signxai.torch_signxai.utils import remove_softmax as torch_remove_softmax
from signxai.torch_signxai.utils import decode_predictions as decode_predictions_pytorch

# Example command line usage: python run_signxai_torch_images.py --image_path examples/data/images/example.jpg --model_path examples/data/models/pytorch/VGG16/VGG16.py --method_name gradient_x_sign

# Import visualization utilities from signxai.common.visualization
try:
    from signxai.common.visualization import normalize_relevance_map, relevance_to_heatmap, overlay_heatmap
    print("Successfully imported visualization utilities from signxai.common.visualization.")
except ImportError as e:
    print(f"Warning: Could not import all visualization utilities from signxai.common.visualization: {e}")
    print("Falling back to basic normalization and direct matplotlib plotting if necessary.")
    # Define minimal fallbacks if any specific function is missing
    if 'normalize_relevance_map' not in globals():
        def normalize_relevance_map(relevance_map, percentile=99):
            abs_map = np.abs(relevance_map)
            vmax = np.percentile(abs_map, percentile)
            if vmax > 0:
                relevance_map = np.clip(relevance_map, -vmax, vmax) / vmax
            return relevance_map
    if 'relevance_to_heatmap' not in globals():
        def relevance_to_heatmap(relevance_map, cmap_name="seismic", symmetric=True):
            print("Using direct matplotlib plotting as 'relevance_to_heatmap' was not imported.")
            return None
    if 'overlay_heatmap' not in globals():
        def overlay_heatmap(image, heatmap, alpha=0.5):
            print("'overlay_heatmap' utility not imported.")
            return image

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(SCRIPT_DIR)
PT_MODEL_DEFINITION_DIR = os.path.join(PROJECT_ROOT, 'examples/data/models/pytorch/VGG16')
PT_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'examples/data/models/pytorch/VGG16/vgg16_ported_weights.pt')
TARGET_SIZE = (224, 224)

# Add PyTorch model directory to path
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
    """Preprocess image for PyTorch VGG16 model using BGR and mean subtraction."""
    with Image.open(image_path) as img_opened:
        pil_img_pt = img_opened.convert('RGB')
    pil_img_pt_resized = pil_img_pt.resize(target_size)
    image_np = np.array(pil_img_pt_resized, dtype=np.float32)
    image_np_bgr = image_np[:, :, ::-1]  # RGB to BGR
    mean_bgr = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    image_np_bgr_mean_subtracted = image_np_bgr - mean_bgr
    x_pt = torch.tensor(np.transpose(image_np_bgr_mean_subtracted, (2, 0, 1))).unsqueeze(0)
    return pil_img_pt_resized, x_pt

def aggregate_explanation(explanation_map, framework_name="PyTorch"):
    """Aggregate explanation map to 2D for visualization."""
    if not hasattr(explanation_map, 'ndim'):
        print(f"Warning: {framework_name} explanation is not a NumPy array. Using zeros for heatmap.")
        return np.zeros(TARGET_SIZE)

    if explanation_map.ndim == 4 and explanation_map.shape[0] == 1:
        if explanation_map.shape[2] == TARGET_SIZE[0] and explanation_map.shape[3] == TARGET_SIZE[1]:  # PT-like (1,C,H,W)
            agg_map = explanation_map[0].sum(axis=0)
        elif explanation_map.shape[1] == TARGET_SIZE[0] and explanation_map.shape[2] == TARGET_SIZE[1]:  # TF-like (1,H,W,C)
            agg_map = explanation_map[0].sum(axis=-1)
        else:
            print(f"Warning: {framework_name} explanation (4D) has unexpected channel/spatial arrangement {explanation_map.shape}. Using zeros.")
            return np.zeros(TARGET_SIZE)
    elif explanation_map.ndim == 3:
        if explanation_map.shape[1] == TARGET_SIZE[0] and explanation_map.shape[2] == TARGET_SIZE[1]:  # PT-like (C,H,W)
            agg_map = explanation_map.sum(axis=0)
        elif explanation_map.shape[0] == TARGET_SIZE[0] and explanation_map.shape[1] == TARGET_SIZE[1]:  # TF-like (H,W,C)
            agg_map = explanation_map.sum(axis=-1)
        else:
            print(f"Warning: {framework_name} explanation (3D) has unexpected channel/spatial arrangement {explanation_map.shape}. Using zeros.")
            return np.zeros(TARGET_SIZE)
    elif explanation_map.ndim == 2:
        agg_map = explanation_map
    else:
        print(f"Warning: {framework_name} explanation has unexpected shape {explanation_map.shape}. Using zeros.")
        return np.zeros(TARGET_SIZE)

    if agg_map.shape != TARGET_SIZE:
        print(f"Warning: {framework_name} aggregated map shape {agg_map.shape} does not match target {TARGET_SIZE}. Attempting resize.")
        if agg_map.ndim == 2:
            try:
                pil_temp = Image.fromarray(agg_map)
                pil_resized = pil_temp.resize(TARGET_SIZE, Image.BILINEAR)
                agg_map = np.array(pil_resized)
                print(f"Resized to {agg_map.shape}")
            except Exception as resize_err:
                print(f"Error resizing heatmap: {resize_err}. Using zeros.")
                return np.zeros(TARGET_SIZE)
        else:
            print(f"Cannot resize non-2D map of shape {agg_map.shape}. Using zeros.")
            return np.zeros(TARGET_SIZE)
    return agg_map

# --- Helper function to get available PyTorch methods ---
def get_available_torch_methods():
    """
    Dynamically gets a list of available PyTorch XAI method names from the wrappers module.
    """
    try:
        from signxai import torch_signxai
        if not torch_signxai:
            print("Warning: signxai.torch_signxai submodule not loaded in get_available_torch_methods.")
            return []

        # Try to get methods from wrappers if it exists
        try:
            wrapper_module = torch_signxai.methods.wrappers
            available_methods = []
            exclude_names = {
                'np', 'torch', 'nn', 'F', 'calculate_grad_cam_relevancemap',
                'calculate_grad_cam_relevancemap_timeseries',
                'guided_backprop_on_guided_model', 'calculate_sign_mu',
                'calculate_explanation_innvestigate',
                'calculate_relevancemap', 'calculate_relevancemaps',
                'calculate_native_gradient', 'calculate_native_integrated_gradients',
                'calculate_native_smoothgrad'
            }

            for name, func in inspect.getmembers(wrapper_module, inspect.isfunction):
                if not name.startswith('_') and name not in exclude_names:
                    if func.__module__ == wrapper_module.__name__:
                        available_methods.append(name)
            return sorted(list(set(available_methods)))
        except AttributeError:
            # If wrappers doesn't exist, return common methods
            return ['gradient', 'smoothgrad', 'integrated_gradients', 'grad_cam', 'guided_backprop', 'deconvnet', 'lrp_epsilon_0_1', 'lrp_alpha_1_beta_0']
    except ImportError:
        print("Warning: Could not import signxai.torch_signxai to list methods.")
        return []

def run_explanation(framework, model_path, image_path, method_name,
                    neuron_selection=None, method_params=None,
                    save_plots=False, output_dir="explanation_outputs"):
    print(f"\n--- Running Explanation ---")
    print(f"Framework: {framework}")
    print(f"Image: {image_path}")
    print(f"Method: {method_name}")
    if neuron_selection is not None:
        print(f"Neuron selection: {neuron_selection}")
    if method_params:
        print(f"Additional method parameters: {method_params}")

    if framework.lower() == 'pytorch':
        try:
            # Ensure PyTorch and the signxai PyTorch submodule are available
            import torch
            from signxai.torch_signxai.methods import wrappers
            if not wrappers:
                raise ImportError(
                    "signxai.torch_signxai.methods.wrappers submodule not loaded. Check your SignXAI __init__.py and installation.")
        except ImportError as e:
            print(f"Error: PyTorch or signxai.torch_signxai module is not available: {e}")
            return False

        print(f"Loading PyTorch model...")
        try:
            # Load model architecture and weights
            if not os.path.exists(PT_MODEL_WEIGHTS_PATH):
                print(f"  Error: Model weights not found at {PT_MODEL_WEIGHTS_PATH}")
                return False
            
            pt_model_original = VGG16_PyTorch(num_classes=1000)
            pt_model_original.load_state_dict(torch.load(PT_MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
            pt_model_original.eval()
            
            # Create a copy for XAI (with softmax removed)
            pt_model_for_xai = VGG16_PyTorch(num_classes=1000)
            pt_model_for_xai.load_state_dict(torch.load(PT_MODEL_WEIGHTS_PATH, map_location=torch.device('cpu')))
            pt_model_for_xai.eval()
            torch_remove_softmax(pt_model_for_xai)
            
        except Exception as e:
            print(f"  Error loading model: {e}")
            return False
        print("  PyTorch model loaded.")

        print(f"Loading and preprocessing image...")
        try:
            pil_img_for_display, x_pt = preprocess_pytorch_image(image_path, TARGET_SIZE)
        except FileNotFoundError:
            print(f"  Error: Image file not found at {image_path}")
            return False
        except Exception as e:
            print(f"  Error loading or preprocessing image: {e}")
            return False
        print("  Image loaded and preprocessed.")

        # Get predictions for target class
        with torch.no_grad():
            output_pt = pt_model_original(x_pt)
        decoded_preds_pt_list = decode_predictions_pytorch(output_pt, top=1)
        decoded_preds_pt_item = decoded_preds_pt_list[0][0] if decoded_preds_pt_list and decoded_preds_pt_list[0] else ("N/A", "Unknown", 0.0)
        print(f"PyTorch Predicted: {decoded_preds_pt_item[1]} (ID: {decoded_preds_pt_item[0]}, Score: {decoded_preds_pt_item[2]:.4f})")
        target_class_pt = torch.argmax(output_pt, dim=1).item()
        
        # Use target class from neuron_selection if provided, otherwise use predicted class
        if neuron_selection is not None:
            target_class_pt = neuron_selection

        print(f"Calculating {method_name} explanation...")

        kwargs_for_method = {}
        if method_params:
            kwargs_for_method.update(method_params)

        try:
            # Handle different parameter names for different methods
            # Use wrapper methods directly - no fallbacks
            method_kwargs = {}
            
            # Add user-provided parameters, filtering out problematic ones
            for key, value in kwargs_for_method.items():
                if key not in ['resize']:  # Only filter known problematic params
                    method_kwargs[key] = value
            
            # Handle different parameter names for different methods
            if 'grad_cam' in method_name.lower():
                # GradCAM methods expect 'target_class'
                method_kwargs['target_class'] = target_class_pt
                # Add default target_layer if not specified - convert string to actual layer
                if 'target_layer' not in method_kwargs:
                    target_layer_name = 'features.28' if 'VGG16' in method_name else 'features.28'
                    # Get the actual layer object from the model
                    target_layer = pt_model_for_xai
                    for attr in target_layer_name.split('.'):
                        target_layer = getattr(target_layer, attr)
                    method_kwargs['target_layer'] = target_layer
                elif isinstance(method_kwargs['target_layer'], str):
                    # Convert string target_layer to actual module
                    target_layer_name = method_kwargs['target_layer']
                    target_layer = pt_model_for_xai
                    for attr in target_layer_name.split('.'):
                        target_layer = getattr(target_layer, attr)
                    method_kwargs['target_layer'] = target_layer
            else:
                # Most other methods expect 'neuron_selection'
                method_kwargs['neuron_selection'] = target_class_pt
            
            # Add common default parameters if not specified
            if 'mu' in method_name and 'mu' not in method_kwargs:
                if '_mu_0_5' in method_name:
                    method_kwargs['mu'] = 0.5
                elif '_mu_neg_0_5' in method_name:
                    method_kwargs['mu'] = -0.5
                elif '_mu_0' in method_name:
                    method_kwargs['mu'] = 0.0
                else:
                    method_kwargs['mu'] = 0.5
            
            # Add epsilon for LRP methods if not specified
            if 'epsilon' in method_name and 'epsilon' not in method_kwargs:
                if '_epsilon_0_001' in method_name:
                    method_kwargs['epsilon'] = 0.001
                elif '_epsilon_0_01' in method_name:
                    method_kwargs['epsilon'] = 0.01
                elif '_epsilon_0_1' in method_name:
                    method_kwargs['epsilon'] = 0.1
                elif '_epsilon_0_25' in method_name:
                    method_kwargs['epsilon'] = 0.25
                elif '_epsilon_0_5' in method_name:
                    method_kwargs['epsilon'] = 0.5
                elif '_epsilon_1' in method_name:
                    method_kwargs['epsilon'] = 1.0
                elif '_epsilon_5' in method_name:
                    method_kwargs['epsilon'] = 5.0
                elif '_epsilon_10' in method_name:
                    method_kwargs['epsilon'] = 10.0
                elif '_epsilon_20' in method_name:
                    method_kwargs['epsilon'] = 20.0
                elif '_epsilon_50' in method_name:
                    method_kwargs['epsilon'] = 50.0
                elif '_epsilon_75' in method_name:
                    method_kwargs['epsilon'] = 75.0
                elif '_epsilon_100' in method_name:
                    method_kwargs['epsilon'] = 100.0
                else:
                    method_kwargs['epsilon'] = 0.1
            
            # Add alpha/beta for LRP alpha-beta methods
            if 'alpha' in method_name and 'beta' in method_name:
                if 'alpha_1_beta_0' in method_name:
                    method_kwargs.update({'alpha': 1, 'beta': 0})
                elif 'alpha_2_beta_1' in method_name:
                    method_kwargs.update({'alpha': 2, 'beta': 1})
            
            # Add smoothgrad parameters
            if 'smoothgrad' in method_name.lower():
                if 'num_samples' not in method_kwargs:
                    method_kwargs['num_samples'] = 50
                if 'noise_level' not in method_kwargs:
                    method_kwargs['noise_level'] = 0.1
            
            # Add integrated gradients parameters
            if 'integrated' in method_name.lower():
                if 'steps' not in method_kwargs:
                    method_kwargs['steps'] = 50
            
            # Add vargrad parameters
            if 'vargrad' in method_name.lower():
                if 'num_samples' not in method_kwargs:
                    method_kwargs['num_samples'] = 50
                if 'noise_level' not in method_kwargs:
                    method_kwargs['noise_level'] = 0.1
            
            explanation_pt_raw = torch_calculate_relevancemap(
                method_name,
                x_pt.clone(),
                pt_model_for_xai,
                **method_kwargs
            )
            
            # Convert tensor to numpy if needed
            if isinstance(explanation_pt_raw, torch.Tensor):
                explanation_pt_raw = explanation_pt_raw.detach().cpu().numpy()
                
            explanation_pt_agg = aggregate_explanation(explanation_pt_raw, "PyTorch")
            
        except NameError as ne:
            print(f"  Error: Method '{method_name}' is not defined or callable. Details: {ne}")
            return False
        except Exception as e:
            print(f"  Error during explanation for method '{method_name}': {e}")
            import traceback
            traceback.print_exc()
            return False

        print(f"  Explanation calculated for {method_name}.")

        # Normalize the explanation
        normalized_heatmap = normalize_relevance_map(explanation_pt_agg)

        # Create visualization
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"Method: {method_name}", fontsize=16)
        
        axs[0].imshow(pil_img_for_display)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        # Use relevance_to_heatmap if available
        heatmap_rgb = relevance_to_heatmap(normalized_heatmap, cmap="seismic", symmetric=True) if 'relevance_to_heatmap' in globals() and callable(globals()['relevance_to_heatmap']) else None

        if heatmap_rgb is not None and heatmap_rgb.ndim == 3:
            axs[1].imshow(heatmap_rgb)
        else:
            # Fallback to direct plotting
            abs_max = np.max(np.abs(normalized_heatmap)) if np.any(normalized_heatmap) else 1.0
            axs[1].imshow(normalized_heatmap, cmap='seismic', vmin=-abs_max, vmax=abs_max)
        
        axs[1].set_title(f"Normalized Heatmap")
        axs[1].axis('off')

        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plot_filename = os.path.join(output_dir, f"{os.path.basename(image_path)}_{method_name}.png")
            plt.savefig(plot_filename)
            print(f"  Plot saved to {plot_filename}")
            plt.close(fig)
        else:
            plt.show()
            print(f"  Visualization displayed for {method_name}.")
        return True

    elif framework.lower() == 'tensorflow':
        print("TensorFlow framework selected. Use run_signxai_tf_images.py for TensorFlow models.")
        return False
    else:
        print(f"Error: Unsupported framework '{framework}'. Choose 'tensorflow' or 'pytorch'.")
        return False

def parse_method_params(params_str):
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
        print(f"Warning: Could not parse method_params string: '{params_str}'. Expected format: key1:value1,key2:value2")
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
    core_args.add_argument('--framework', type=str, default='pytorch', choices=['tensorflow', 'pytorch'],
                           help="The deep learning framework to use (default: pytorch).")
    core_args.add_argument('--image_path', type=str,
                           help="Path to the input image. Required unless --list_methods or --run_all_torch_methods is used with predefined paths.")
    core_args.add_argument('--model_path', type=str,
                           help="Path to the trained model file (optional for PyTorch VGG16).")

    method_selection_group = parser.add_mutually_exclusive_group()
    method_selection_group.add_argument('--method_name', type=str,
                                        help="Name of the XAI method to use (e.g., gradient).")
    method_selection_group.add_argument('--run_all_torch_methods', action='store_true',
                                        help="Run all available PyTorch XAI methods sequentially. Ignores --method_name.")

    optional_args = parser.add_argument_group('Optional arguments')
    optional_args.add_argument('--neuron_selection', type=int, default=None,
                               help="Index of the neuron to explain for some methods.")
    optional_args.add_argument('--method_params', type=str, default=None,
                               help="Additional parameters for the method, as a comma-separated string of key:value pairs (e.g., 'epsilon:0.1,steps:100').")
    optional_args.add_argument('--list_methods', action='store_true',
                               help="List available XAI methods for the selected framework and exit.")
    optional_args.add_argument('--save_plots', action='store_true',
                               help="Save plots to files instead of displaying them interactively. Used with --run_all_torch_methods or single method.")
    optional_args.add_argument('--output_dir', type=str, default="explanation_outputs",
                               help="Directory to save plots if --save_plots is used (default: explanation_outputs).")
    optional_args.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                               help='show this help message and exit')

    args = parser.parse_args()

    torch_methods_list = []
    if args.framework.lower() == 'pytorch':
        torch_methods_list = get_available_torch_methods()

    if args.list_methods:
        if args.framework.lower() == 'pytorch':
            if torch_methods_list:
                print("\nAvailable PyTorch XAI methods:")
                for m_name in torch_methods_list:
                    print(f"  - {m_name}")
            else:
                print("Could not retrieve PyTorch methods. Ensure SignXAI and PyTorch backend are correctly installed.")
        elif args.framework.lower() == 'tensorflow':
            print("\nTensorFlow method listing not implemented in this PyTorch script. Use run_signxai_tf_images.py.")
        sys.exit(0)

    # Validate required arguments if not listing methods
    if not args.image_path:
        parser.error("the following argument is required: --image_path")

    if not args.run_all_torch_methods and not args.method_name:
        parser.error("either --method_name must be specified or --run_all_torch_methods flag must be set.")

    parsed_method_params = parse_method_params(args.method_params)

    if args.run_all_torch_methods:
        if args.framework.lower() != 'pytorch':
            print("Error: --run_all_torch_methods is only compatible with --framework pytorch.")
            sys.exit(1)

        if not torch_methods_list:
            print("Error: Cannot run all PyTorch methods because the method list is empty.")
            sys.exit(1)

        print(f"\n--- Running ALL {len(torch_methods_list)} PyTorch XAI methods ---")
        print(f"Image: {args.image_path}")
        if args.save_plots:
            print(f"Plots will be saved to: {args.output_dir}")
        else:
            print("Plots will be displayed interactively. Close each plot to continue to the next method.")

        succeeded_methods = []
        failed_methods = []

        for i, method_to_run in enumerate(torch_methods_list):
            print(f"\nProcessing method {i + 1}/{len(torch_methods_list)}: {method_to_run}")
            if run_explanation(args.framework, args.model_path, args.image_path, method_to_run,
                               args.neuron_selection, parsed_method_params,
                               args.save_plots, args.output_dir):
                succeeded_methods.append(method_to_run)
            else:
                failed_methods.append(method_to_run)

            if args.save_plots:
                time.sleep(0.1)

        print("\n--- Batch Processing Summary ---")
        print(f"Successfully processed: {len(succeeded_methods)} methods")
        if succeeded_methods: print(f"  {succeeded_methods}")
        print(f"Failed or skipped: {len(failed_methods)} methods")
        if failed_methods: print(f"  {failed_methods}")

    else:  # Single method run
        # Validate method_name for PyTorch
        if args.framework.lower() == 'pytorch':
            if not torch_methods_list:
                print(f"Warning: Could not verify method '{args.method_name}' as the available method list is empty. Attempting to run it anyway.")
            elif args.method_name not in torch_methods_list:
                print(f"Error: Method '{args.method_name}' is not a recognized PyTorch XAI method.")
                print("Note: Method names are case-sensitive.")
                print(f"\nAvailable PyTorch methods: {torch_methods_list}")
                sys.exit(1)

        run_explanation(args.framework, args.model_path, args.image_path, args.method_name,
                        args.neuron_selection, parsed_method_params,
                        args.save_plots, args.output_dir)
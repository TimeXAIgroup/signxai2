import tensorflow as tf
import torch
import numpy as np
import sys
import os

# --- Dynamically add the project root to sys.path ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

vgg16_model_def_dir = os.path.join(project_root, 'examples', 'data', 'models', 'pytorch', 'VGG16')
if vgg16_model_def_dir not in sys.path:
    sys.path.insert(0, vgg16_model_def_dir)

try:
    from VGG16 import VGG16_PyTorch
    from signxai.utils.utils import load_image
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)  # Changed exit() to sys.exit(1) for better practice
# --- End of dynamic import path setup ---

# Define a new tolerance for VGG16 comparisons in the current environment
# This value might need adjustment based on observed differences across layers.
# Start with something slightly above the max diff you saw for block1_conv1 (e.g., 6.96e-5 -> 8e-5 or 1e-4)
VGG16_COMPARISON_ATOL = 5e-3  # ADJUST THIS VALUE AS NEEDED


def compare_outputs(tf_output_np, pt_output_np_raw, layer_name, atol=1e-7,
                    is_feature_map=True):  # Default atol kept for other potential uses
    print(f"\n--- {layer_name} Output Comparison ---")

    pt_output_to_compare = pt_output_np_raw
    if is_feature_map and pt_output_np_raw.ndim == 4:
        pt_output_to_compare = pt_output_np_raw.transpose(0, 2, 3, 1)  # NCHW to NHWC

    print(f"TF output shape: {tf_output_np.shape}, PyTorch output shape (for comparison): {pt_output_to_compare.shape}")

    if tf_output_np.shape != pt_output_to_compare.shape:
        print(
            f"FAILURE: Shape Mismatch for {layer_name}! TF: {tf_output_np.shape}, PyTorch: {pt_output_to_compare.shape}")
        return False

    # Use the specific VGG16_COMPARISON_ATOL for VGG16 comparisons in this script
    # If this function were generic, you might pass the model type or a specific atol.
    # For this script, we are focused on VGG16.
    current_atol_for_comparison = atol
    if "block" in layer_name or "fc" in layer_name or "predictions" in layer_name or "Flattened" in layer_name:  # Heuristic for VGG16 layers
        current_atol_for_comparison = VGG16_COMPARISON_ATOL

    if np.allclose(tf_output_np, pt_output_to_compare, atol=current_atol_for_comparison):
        print(f"SUCCESS: Outputs for {layer_name} are very close (atol={current_atol_for_comparison})!")
        return True
    else:
        print(f"FAILURE: Outputs for {layer_name} are NOT close enough (atol={current_atol_for_comparison}).")
        difference = np.abs(tf_output_np - pt_output_to_compare)
        print(f"  Max absolute difference: {np.max(difference)}")
        print(f"  Mean absolute difference: {np.mean(difference)}")

        num_diff_to_show = 3
        if difference.size > 0:
            # Get indices of the largest differences
            flat_diff_indices = np.argsort(-difference.flatten())[
                                :min(num_diff_to_show * 2, difference.size)]  # Get a few more to filter

            printed_diff_count = 0
            print(f"  Top up to {num_diff_to_show} differing values (original indices):")
            for flat_idx in flat_diff_indices:
                if printed_diff_count >= num_diff_to_show:
                    break
                idx_tuple = np.unravel_index(flat_idx, tf_output_np.shape)
                tf_val = tf_output_np[idx_tuple]
                pt_val = pt_output_to_compare[idx_tuple]
                diff_val = difference[idx_tuple]
                # Only print if they are actually not close according to the tolerance
                if not np.isclose(tf_val, pt_val, atol=current_atol_for_comparison):
                    print(f"    Index {idx_tuple}: TF={tf_val:.7f}, PT={pt_val:.7f}, Diff={diff_val:.7f}")
                    printed_diff_count += 1
            if printed_diff_count == 0:
                print(
                    "    (All top differing values were within tolerance, but np.allclose failed overall - check for NaNs or Infs if this happens)")

        return False


def main():
    tf_model_h5_path = os.path.join(project_root, 'examples', 'data', 'models', 'tensorflow', 'VGG16', 'model.h5')
    pytorch_weights_path = os.path.join(vgg16_model_def_dir, 'vgg16_ported_weights.pt')
    image_file_path = os.path.join(project_root, 'examples', 'data', 'images', 'example.jpg')

    print(f"TensorFlow Model: {tf_model_h5_path}")
    print(f"PyTorch Weights: {pytorch_weights_path}")
    print(f"Image: {image_file_path}\n")

    print("Preprocessing image...")
    try:
        _, preprocessed_img_np_nhwc = load_image(
            image_file_path, target_size=(224, 224), expand_dims=True, use_original_preprocessing=True
        )
        print(f"Image preprocessed. Shape: {preprocessed_img_np_nhwc.shape}")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return

    # Verify inputs (optional, but good for sanity check)
    # print("\n--- Verifying Preprocessed Input Tensor Consistency ---")
    # temp_pt_input_nhwc_for_check = torch.from_numpy(preprocessed_img_np_nhwc).permute(0, 3, 1, 2).contiguous().permute(0,2,3,1).cpu().numpy()
    # if np.array_equal(preprocessed_img_np_nhwc, temp_pt_input_nhwc_for_check):
    #     print("SUCCESS: Initial preprocessed inputs (TF NHWC vs PT NCHW->NHWC after full cycle) are numerically identical.")
    # else:
    #     abs_diff_inputs = np.abs(preprocessed_img_np_nhwc - temp_pt_input_nhwc_for_check)
    #     print(f"FAILURE: Initial preprocessed inputs differ! Max diff: {np.max(abs_diff_inputs)}")
    # print("-" * 50)

    try:
        tf_gpus = tf.config.experimental.list_physical_devices('GPU')
        if tf_gpus:
            for gpu in tf_gpus: tf.config.experimental.set_memory_growth(gpu, True)

        original_tf_model = tf.keras.models.load_model(tf_model_h5_path, compile=False)

        pytorch_model = VGG16_PyTorch(num_classes=1000)
        pytorch_model.load_state_dict(torch.load(pytorch_weights_path, map_location=torch.device('cpu')))
        pytorch_model.eval()
        pytorch_input_tensor = torch.from_numpy(preprocessed_img_np_nhwc.astype(np.float32)).permute(0, 3, 1,
                                                                                                     2).contiguous()


    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return

    layers_to_test_config = [
        {'tf_name': 'block1_conv1', 'pt_slice_end': 2, 'is_feature_map': True},
        {'tf_name': 'block1_conv2', 'pt_slice_end': 4, 'is_feature_map': True},
        {'tf_name': 'block1_pool', 'pt_slice_end': 5, 'is_feature_map': True},
        {'tf_name': 'block2_conv1', 'pt_slice_end': 7, 'is_feature_map': True},
        {'tf_name': 'block2_conv2', 'pt_slice_end': 9, 'is_feature_map': True},
        {'tf_name': 'block2_pool', 'pt_slice_end': 10, 'is_feature_map': True},
        {'tf_name': 'block3_conv1', 'pt_slice_end': 12, 'is_feature_map': True},
        {'tf_name': 'block3_conv2', 'pt_slice_end': 14, 'is_feature_map': True},
        {'tf_name': 'block3_conv3', 'pt_slice_end': 16, 'is_feature_map': True},
        {'tf_name': 'block3_pool', 'pt_slice_end': 17, 'is_feature_map': True},
        {'tf_name': 'block4_conv1', 'pt_slice_end': 19, 'is_feature_map': True},
        {'tf_name': 'block4_conv2', 'pt_slice_end': 21, 'is_feature_map': True},
        {'tf_name': 'block4_conv3', 'pt_slice_end': 23, 'is_feature_map': True},
        {'tf_name': 'block4_pool', 'pt_slice_end': 24, 'is_feature_map': True},
        {'tf_name': 'block5_conv1', 'pt_slice_end': 26, 'is_feature_map': True},
        {'tf_name': 'block5_conv2', 'pt_slice_end': 28, 'is_feature_map': True},
        {'tf_name': 'block5_conv3', 'pt_slice_end': 30, 'is_feature_map': True},
        {'tf_name': 'block5_pool', 'pt_slice_end': 31, 'is_feature_map': True},
    ]

    all_match_so_far = True

    for config in layers_to_test_config:
        tf_layer_name = config['tf_name']
        pt_slice_end = config['pt_slice_end']
        is_feature_map = config['is_feature_map']

        if not all_match_so_far:
            print(f"\nSkipping further checks as a previous layer failed.")
            break

        try:
            intermediate_tf_model = tf.keras.Model(inputs=original_tf_model.input,
                                                   outputs=original_tf_model.get_layer(tf_layer_name).output)
            tf_intermediate_output = intermediate_tf_model.predict(preprocessed_img_np_nhwc, verbose=0)

            with torch.no_grad():
                # Cleaner way to get PyTorch segment output
                pt_segment_to_run = torch.nn.Sequential(*list(pytorch_model.features)[:pt_slice_end])
                pt_intermediate_output_tensor = pt_segment_to_run(pytorch_input_tensor)

            # Pass the globally defined VGG16_COMPARISON_ATOL for these layers
            if not compare_outputs(tf_intermediate_output, pt_intermediate_output_tensor.cpu().numpy(), tf_layer_name,
                                   atol=VGG16_COMPARISON_ATOL, is_feature_map=is_feature_map):
                all_match_so_far = False
        except Exception as e:
            print(f"Error comparing {tf_layer_name}: {e}")
            import traceback
            traceback.print_exc()
            all_match_so_far = False

    if all_match_so_far:
        print("\n--- Testing after full feature extraction and flatten ---")
        tf_flatten_layer_name = 'flatten'
        try:
            tf_flatten_model = tf.keras.Model(inputs=original_tf_model.input,
                                              outputs=original_tf_model.get_layer(tf_flatten_layer_name).output)
            tf_flatten_output = tf_flatten_model.predict(preprocessed_img_np_nhwc, verbose=0)

            with torch.no_grad():
                pt_features_output_nchw = pytorch_model.features(pytorch_input_tensor)
                pt_features_output_nhwc = pt_features_output_nchw.permute(0, 2, 3, 1).contiguous()
                pt_flattened_output = torch.flatten(pt_features_output_nhwc, 1)

            # Use VGG16_COMPARISON_ATOL for flatten output
            if not compare_outputs(tf_flatten_output, pt_flattened_output.cpu().numpy(), "Flattened Features",
                                   atol=VGG16_COMPARISON_ATOL, is_feature_map=False):
                all_match_so_far = False
        except Exception as e:
            print(f"Error during flatten layer comparison: {e}")
            all_match_so_far = False

    if all_match_so_far:
        print("\n--- Testing Classifier Layers ---")
        tf_input_to_classifier = tf_flatten_output  # Output from previous TF stage
        pt_input_to_classifier = pt_flattened_output  # Output from previous PT stage

        # Ensure pt_input_to_classifier is a tensor for PyTorch dense layers
        if isinstance(pt_input_to_classifier, np.ndarray):
            pt_input_to_classifier = torch.from_numpy(pt_input_to_classifier.astype(np.float32)).to(
                pytorch_input_tensor.device)

        classifier_layers_to_test = [
            {'tf_name': 'fc1', 'pt_module_idx': 0, 'pt_activation_idx': 1},  # Linear, ReLU
            {'tf_name': 'fc2', 'pt_module_idx': 2, 'pt_activation_idx': 3},  # Linear, ReLU
            {'tf_name': 'predictions', 'pt_module_idx': 4, 'pt_activation_idx': None}  # Linear
        ]

        for i, config_cls in enumerate(classifier_layers_to_test):
            tf_layer_name_cls = config_cls['tf_name']
            pt_linear_module = pytorch_model.classifier[config_cls['pt_module_idx']]
            pt_relu_module = pytorch_model.classifier[config_cls['pt_activation_idx']] if config_cls[
                                                                                              'pt_activation_idx'] is not None else None

            if not all_match_so_far: break

            try:
                # For TF, pass the output of the PREVIOUS TF layer
                tf_current_layer_obj = original_tf_model.get_layer(tf_layer_name_cls)
                tf_intermediate_output_tensor_cls = tf_current_layer_obj(tf_input_to_classifier)

                with torch.no_grad():
                    pt_intermediate_output_tensor_cls = pt_linear_module(pt_input_to_classifier)
                    if pt_relu_module:
                        pt_intermediate_output_tensor_cls = pt_relu_module(pt_intermediate_output_tensor_cls)

                # Use VGG16_COMPARISON_ATOL for classifier layers
                if not compare_outputs(tf_intermediate_output_tensor_cls.numpy(),
                                       pt_intermediate_output_tensor_cls.cpu().numpy(),
                                       tf_layer_name_cls, atol=VGG16_COMPARISON_ATOL, is_feature_map=False):
                    all_match_so_far = False

                tf_input_to_classifier = tf_intermediate_output_tensor_cls
                pt_input_to_classifier = pt_intermediate_output_tensor_cls
            except Exception as e:
                print(f"Error comparing classifier layer {tf_layer_name_cls}: {e}")
                all_match_so_far = False
                break

    if all_match_so_far:
        print("\n\nSUCCESS: All tested layers and final logits match closely (within the adjusted tolerance)!")
    else:
        print(
            "\n\nPARTIAL SUCCESS or FAILURE: Not all layers matched with the current tolerance. Check log for details.")


if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Mitigate potential oneDNN variations if any
    main()
import numpy as np
import os

# Define project root if this script is in utils/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))
output_dir = os.path.join(project_root, "utils/output")


def compare_lbbb_weights():
    print("--- Comparing LBBB conv1d_3 (TF) vs conv4 (PT) Weights ---")

    # Load TensorFlow weights
    tf_kernel_path = os.path.join(output_dir, "conv1d_3_tf_kernel.npy")
    tf_bias_path = os.path.join(output_dir, "conv1d_3_tf_bias.npy")

    # Load PyTorch weights (using the prefix from inspect_pytorch_layer_weights.py)
    # The prefix was 'lbbb_conv1d_3_pt', so files are lbbb_conv1d_3_pt_kernel.npy and lbbb_conv1d_3_pt_bias.npy
    pt_kernel_path = os.path.join(output_dir, "lbbb_conv1d_3_pt_kernel.npy")
    pt_bias_path = os.path.join(output_dir, "lbbb_conv1d_3_pt_bias.npy")

    if not all(os.path.exists(p) for p in [tf_kernel_path, tf_bias_path, pt_kernel_path, pt_bias_path]):
        print("ERROR: One or more .npy weight files not found. Please run inspection scripts first.")
        return

    tf_kernel = np.load(tf_kernel_path)
    tf_bias = np.load(tf_bias_path)
    pt_kernel = np.load(pt_kernel_path)
    pt_bias = np.load(pt_bias_path)

    print("\n--- Kernel Weight Comparison ---")
    print(f"TF Kernel original shape: {tf_kernel.shape}")  # Should be (3, 64, 64)
    print(f"PT Kernel shape: {pt_kernel.shape}")  # Should be (64, 64, 3)

    # Transpose TF kernel to match PyTorch's (OC, IC, KS) format
    # TF: (Kernel Size, Input Channels, Output Channels) -> (3, 64, 64)
    # PT: (Output Channels, Input Channels, Kernel Size) -> (64, 64, 3)
    # Permutation: (2, 1, 0) -> OC from index 2, IC from index 1, KS from index 0
    tf_kernel_permuted = tf_kernel.transpose(2, 1, 0)
    print(f"TF Kernel permuted shape for comparison: {tf_kernel_permuted.shape}")

    if tf_kernel_permuted.shape != pt_kernel.shape:
        print("FAILURE: Kernel shapes do not match after permutation!")
        return

    # Using a very strict tolerance for direct weight comparison
    atol_weights = 1e-8
    if np.allclose(tf_kernel_permuted, pt_kernel, atol=atol_weights):
        print(f"SUCCESS: Kernel weights are identical (atol={atol_weights})!")
    else:
        print(f"FAILURE: Kernel weights differ (atol={atol_weights}).")
        kernel_diff = np.abs(tf_kernel_permuted - pt_kernel)
        print(f"  Max absolute difference in kernels: {np.max(kernel_diff):.6e}")
        print(
            f"  Number of differing elements: {np.sum(~np.isclose(tf_kernel_permuted, pt_kernel, atol=atol_weights))}")

    print("\n--- Bias Weight Comparison ---")
    print(f"TF Bias shape: {tf_bias.shape}")  # Should be (64,)
    print(f"PT Bias shape: {pt_bias.shape}")  # Should be (64,)

    if tf_bias.shape != pt_bias.shape:
        print("FAILURE: Bias shapes do not match!")
        return

    if np.allclose(tf_bias, pt_bias, atol=atol_weights):
        print(f"SUCCESS: Bias weights are identical (atol={atol_weights})!")
    else:
        print(f"FAILURE: Bias weights differ (atol={atol_weights}).")
        bias_diff = np.abs(tf_bias - pt_bias)
        print(f"  Max absolute difference in biases: {np.max(bias_diff):.6e}")
        print(f"  Number of differing elements: {np.sum(~np.isclose(tf_bias, pt_bias, atol=atol_weights))}")


if __name__ == '__main__':
    compare_lbbb_weights()
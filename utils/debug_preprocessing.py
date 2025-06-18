import numpy as np
from PIL import Image
import tensorflow as tf  # For the standard Keras VGG16 preprocessing
import sys
import os

# --- Dynamically add the project root to sys.path ---
# This script is in utils/
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, '..'))  # Up one level to signxai-0.1.0/

if project_root not in sys.path:
    sys.path.insert(0, project_root)  # Add project root

try:
    # Import your utility function from your signxai package
    from signxai.utils.utils import load_image as signxai_load_image
except ImportError as e:
    print(f"Error importing signxai.utils.utils.load_image: {e}")
    print("Please ensure your 'signxai' package is installed (e.g., 'pip install -e .' from project root).")
    exit()


# --- End of dynamic import path setup ---

def main():
    image_file_path = os.path.join(project_root, 'examples', 'data', 'images', 'example.jpg')
    print(f"Using image: {image_file_path}\n")

    # --- Method 1: Your signxai.utils.utils.load_image ---
    print("--- Preprocessing with signxai.utils.utils.load_image (use_original_preprocessing=True) ---")
    try:
        # We want NHWC, BGR, mean-subtracted.
        # Your function with use_original_preprocessing=True and expand_dims=True should provide this.
        _, custom_preprocessed_output_nhwc = signxai_load_image(
            image_file_path,
            target_size=(224, 224),
            expand_dims=True,
            use_original_preprocessing=True
        )
        print(f"Custom Preprocessed Output Shape: {custom_preprocessed_output_nhwc.shape}")
        print(f"Custom Preprocessed Output dtype: {custom_preprocessed_output_nhwc.dtype}")
        # print("Custom pixel (0,0,0) BGR values:", custom_preprocessed_output_nhwc[0,0,0,:])
        # print("Custom pixel (100,100,0) BGR values:", custom_preprocessed_output_nhwc[0,100,100,:])

    except Exception as e:
        print(f"Error using signxai.utils.utils.load_image: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Method 2: Standard tf.keras.applications.vgg16.preprocess_input ---
    print("\n--- Preprocessing with tf.keras.applications.vgg16.preprocess_input ---")
    try:
        from tensorflow.keras.applications.vgg16 import preprocess_input as keras_vgg16_preprocess_input
        from tensorflow.keras.preprocessing import image as keras_image_loader

        img_pil = Image.open(image_file_path).resize((224, 224))
        img_np_rgb_hwc = np.array(img_pil, dtype=np.float32)  # HWC, RGB

        # Keras preprocess_input expects a batch of RGB images (NHWC)
        img_np_rgb_nhwc_batch = np.expand_dims(img_np_rgb_hwc.copy(), axis=0)

        # This function will convert RGB to BGR and subtract the specific VGG16 means.
        keras_preprocessed_output_nhwc = keras_vgg16_preprocess_input(img_np_rgb_nhwc_batch.copy())

        print(f"Keras Preprocessed Output Shape: {keras_preprocessed_output_nhwc.shape}")
        print(f"Keras Preprocessed Output dtype: {keras_preprocessed_output_nhwc.dtype}")
        # print("Keras pixel (0,0,0) BGR values:", keras_preprocessed_output_nhwc[0,0,0,:])
        # print("Keras pixel (100,100,0) BGR values:", keras_preprocessed_output_nhwc[0,100,100,:])

    except Exception as e:
        print(f"Error using tf.keras.applications.vgg16.preprocess_input: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Comparison of Preprocessing Outputs ---
    print("\n--- Preprocessing Comparison ---")
    if custom_preprocessed_output_nhwc.shape != keras_preprocessed_output_nhwc.shape:
        print(
            f"Shape Mismatch! Custom: {custom_preprocessed_output_nhwc.shape}, Keras: {keras_preprocessed_output_nhwc.shape}")
    else:
        if np.allclose(custom_preprocessed_output_nhwc, keras_preprocessed_output_nhwc, atol=1e-5):
            print("SUCCESS: Preprocessing outputs are identical or very close!")
        else:
            print("FAILURE: Preprocessing outputs DIFFER.")
            difference = np.abs(custom_preprocessed_output_nhwc - keras_preprocessed_output_nhwc)
            print(f"  Max absolute difference in preprocessing: {np.max(difference)}")
            print(f"  Mean absolute difference in preprocessing: {np.mean(difference)}")

            # Check a few specific pixels if they differ
            differing_pixels = np.where(np.abs(custom_preprocessed_output_nhwc - keras_preprocessed_output_nhwc) > 1e-5)
            if differing_pixels[0].size > 0:
                print("  Example of a differing pixel (first one found):")
                idx = (differing_pixels[0][0], differing_pixels[1][0], differing_pixels[2][0], differing_pixels[3][0])
                print(f"    Pixel index: ({idx[1]}, {idx[2]}, {idx[3]}) (H,W,C for BGR)")  # Assuming batch is 1
                print(f"    Custom Value: {custom_preprocessed_output_nhwc[idx]}")
                print(f"    Keras Value:  {keras_preprocessed_output_nhwc[idx]}")
                print(f"    Difference:   {custom_preprocessed_output_nhwc[idx] - keras_preprocessed_output_nhwc[idx]}")


if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    # Ensure TensorFlow doesn't grab all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"RuntimeError during memory growth setting: {e}")
    main()
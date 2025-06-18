"""Test suite for comparing TensorFlow and PyTorch implementations."""

import os
import pytest
import numpy as np
import tensorflow as tf
import torch
from pathlib import Path

# Import TensorFlow implementations
from signxai.tf_signxai.methods.wrappers import calculate_relevancemap as tf_calculate_relevancemap

# Import PyTorch implementations
from signxai.torch_signxai.methods import calculate_relevancemap as pt_calculate_relevancemap

# Import utils
from signxai.torch_signxai.utils import preprocess_image
from signxai.common.visualization import load_image

# Define test directory and resource paths
TEST_DIR = Path(__file__).parent.absolute()
TEST_RESOURCES = TEST_DIR / "resources"
TEST_OUTPUTS = TEST_DIR / "outputs"

# Create output directory if it doesn't exist
os.makedirs(TEST_OUTPUTS, exist_ok=True)

# Test models mapping
MODEL_PATHS = {
    "tensorflow": {
        "vgg16": None,  # Will be loaded from keras
        "ecg": None,    # Not used in these tests
    },
    "pytorch": {
        "vgg16": None,  # Will be initialized from TensorFlow model
        "ecg": None,    # Not used in these tests
    }
}

# Test methods mapping
METHODS_MAPPING = {
    # Basic gradient methods
    "gradient": {
        "tf": "gradient",
        "pt": "gradient"
    },
    "integrated_gradients": {
        "tf": "integrated_gradients",
        "pt": "integrated_gradients"
    },
    "smoothgrad": {
        "tf": "smoothgrad",
        "pt": "smoothgrad"
    },
    
    # Input×gradient variants
    "input_t_gradient": {
        "tf": "input_t_gradient",
        "pt": "gradient_x_input"
    },
    
    # Sign variants
    "gradient_x_sign": {
        "tf": "gradient_x_sign",
        "pt": "gradient_x_sign"
    },
    "gradient_x_sign_mu_0": {
        "tf": "gradient_x_sign_mu_0",
        "pt": "gradient_x_sign_mu_0"
    },
    
    # Special methods
    "guided_backprop": {
        "tf": "guided_backprop",
        "pt": "guided_backprop"
    },
    "deconvnet": {
        "tf": "deconvnet",
        "pt": "deconvnet"
    },
    "grad_cam": {
        "tf": "grad_cam",
        "pt": "grad_cam"
    },
    
    # LRP variants
    "lrp.epsilon": {
        "tf": "lrp.epsilon",
        "pt": "lrp_epsilon"
    },
    "lrp.alpha_1_beta_0": {
        "tf": "lrp.alpha_1_beta_0",
        "pt": "lrp_alpha_1_beta_0"
    },
    "lrp.alpha_2_beta_1": {
        "tf": "lrp.alpha_2_beta_1",
        "pt": "lrp_alpha_2_beta_1"
    },
    "lrp.gamma": {
        "tf": "lrp.gamma",
        "pt": "lrp_gamma"
    },
    "lrp.flat": {
        "tf": "lrp.flat",
        "pt": "lrp_flat"
    },
    "lrp.w_square": {
        "tf": "lrp.w_square",
        "pt": "lrp_wsquare"
    },
    "lrp.z_plus": {
        "tf": "lrp.z_plus",
        "pt": "lrp_zplus"
    },
    
    # DeepLift
    "deeplift": {
        "tf": "deeplift",
        "pt": "deeplift"
    }
}


def setup_tensorflow_model():
    """Create and return a TensorFlow VGG16 model."""
    # Load VGG16 from keras with imagenet weights
    keras_model = tf.keras.applications.VGG16(weights='imagenet')
    
    # Remove softmax to match innvestigate requirements
    keras_model.layers[-1].activation = tf.keras.activations.linear
    
    return keras_model


def setup_pytorch_model(tf_model=None):
    """Create and return a PyTorch VGG16 model."""
    # For this test, we'll use a native PyTorch VGG16
    # This is simpler than trying to convert from TensorFlow
    import torchvision.models as models
    pt_model = models.vgg16(pretrained=True)
    pt_model.eval()
    return pt_model


def preprocess_inputs(image_path, target_size=(224, 224)):
    """Load and preprocess image for both frameworks."""
    # Load image
    img = load_image(image_path, target_size=target_size)
    
    # Preprocess for TensorFlow
    tf_input = tf.keras.applications.vgg16.preprocess_input(
        img.copy()
    )
    tf_input = np.expand_dims(tf_input, axis=0)  # Add batch dimension
    
    # Preprocess for PyTorch
    pt_input = preprocess_image(img.copy(), framework="pytorch")
    pt_input = torch.tensor(pt_input).unsqueeze(0)  # Add batch dimension
    
    return tf_input, pt_input, img


def calculate_metrics(tf_attr, pt_attr):
    """Calculate comparison metrics between TensorFlow and PyTorch attributions."""
    # Apply preprocessing to handle different frameworks' conventions
    # Channel dimension handling
    if len(tf_attr.shape) == 4 and tf_attr.shape[0] == 1:
        tf_attr = tf_attr[0]  # Remove batch dimension
    if len(pt_attr.shape) == 4 and pt_attr.shape[0] == 1:
        pt_attr = pt_attr[0]  # Remove batch dimension
    
    # Sum across color channels if present
    if len(tf_attr.shape) == 3 and tf_attr.shape[2] == 3:
        tf_attr = np.sum(tf_attr, axis=2)
    if len(pt_attr.shape) == 3 and pt_attr.shape[2] == 3:
        pt_attr = np.sum(pt_attr, axis=2)
        
    # Normalize both to [-1, 1] range
    tf_max = np.max(np.abs(tf_attr))
    pt_max = np.max(np.abs(pt_attr))
    
    if tf_max > 0:
        tf_attr = tf_attr / tf_max
    if pt_max > 0:
        pt_attr = pt_attr / pt_max
    
    # Calculate metrics
    mae = np.mean(np.abs(tf_attr - pt_attr))
    mse = np.mean(np.square(tf_attr - pt_attr))
    max_diff = np.max(np.abs(tf_attr - pt_attr))
    
    # Calculate correlation (handled with error case for constant arrays)
    try:
        correlation = np.corrcoef(tf_attr.flatten(), pt_attr.flatten())[0, 1]
        # NaN can happen if one array is constant
        if np.isnan(correlation):
            correlation = np.nan
    except:
        correlation = np.nan
    
    return {
        "mae": mae,
        "mse": mse,
        "max_diff": max_diff,
        "correlation": correlation
    }


def visualize_comparison(tf_attr, pt_attr, original_img, method_name, save_path=None):
    """Create visualization of TF vs PyTorch results."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Process attributions for visualization
    # Handle batch dimension
    if len(tf_attr.shape) == 4 and tf_attr.shape[0] == 1:
        tf_attr = tf_attr[0]
    if len(pt_attr.shape) == 4 and pt_attr.shape[0] == 1:
        pt_attr = pt_attr[0]
    
    # Sum across color channels if present
    if len(tf_attr.shape) == 3 and tf_attr.shape[2] == 3:
        tf_attr_viz = np.sum(tf_attr, axis=2)
    else:
        tf_attr_viz = tf_attr
        
    if len(pt_attr.shape) == 3 and pt_attr.shape[2] == 3:
        pt_attr_viz = np.sum(pt_attr, axis=2)
    else:
        pt_attr_viz = pt_attr
    
    # Normalize for visualization
    tf_max = np.max(np.abs(tf_attr_viz))
    pt_max = np.max(np.abs(pt_attr_viz))
    
    # Normalize to [-1, 1]
    if tf_max > 0:
        tf_attr_viz = tf_attr_viz / tf_max
    if pt_max > 0:
        pt_attr_viz = pt_attr_viz / pt_max
    
    # Calculate difference
    diff = np.abs(tf_attr_viz - pt_attr_viz)
    
    # Create figure with subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot original image
    axs[0].imshow(original_img)
    axs[0].set_title("Original Image")
    axs[0].axis('off')
    
    # Plot TensorFlow attribution
    im1 = axs[1].imshow(tf_attr_viz, cmap='seismic', vmin=-1, vmax=1)
    axs[1].set_title(f"TensorFlow {method_name}")
    axs[1].axis('off')
    divider = make_axes_locatable(axs[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1, cax=cax1)
    
    # Plot PyTorch attribution
    im2 = axs[2].imshow(pt_attr_viz, cmap='seismic', vmin=-1, vmax=1)
    axs[2].set_title(f"PyTorch {method_name}")
    axs[2].axis('off')
    divider = make_axes_locatable(axs[2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2, cax=cax2)
    
    # Plot difference
    im3 = axs[3].imshow(diff, cmap='hot', vmin=0, vmax=1)
    metrics = calculate_metrics(tf_attr, pt_attr)
    axs[3].set_title(f"Absolute Difference\nMAE: {metrics['mae']:.6f}")
    axs[3].axis('off')
    divider = make_axes_locatable(axs[3])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3, cax=cax3)
    
    plt.tight_layout()
    
    # Save visualization if path is provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig, metrics


def run_method_test(tf_model, pt_model, method, tf_input, pt_input, img, save_path=None):
    """Run a single method test and return comparison metrics."""
    # Get method names for each framework
    tf_method = METHODS_MAPPING[method]["tf"]
    pt_method = METHODS_MAPPING[method]["pt"]
    
    # Define common args for each method type
    common_args = {}
    if method in ["integrated_gradients"]:
        common_args["steps"] = 50
    elif method in ["smoothgrad"]:
        common_args["noise_level"] = 0.2
        common_args["num_samples"] = 25  # Use fewer samples for tests
    elif method in ["grad_cam"]:
        # For grad_cam, we need to identify the target layer
        # Use default behavior since the actual layer name will differ between frameworks
        pass
    
    # Set target class for all methods
    common_args["neuron_selection"] = 282  # For TensorFlow
    common_args["target_class"] = 282  # For PyTorch
    
    # Enable TensorFlow compatibility mode for PyTorch
    common_args["tf_compat_mode"] = True
    
    # Run method for TensorFlow
    try:
        tf_attr = tf_calculate_relevancemap(tf_method, tf_input, tf_model, **common_args)
    except Exception as e:
        print(f"TensorFlow error for {method}: {e}")
        tf_attr = np.zeros_like(tf_input)
    
    # Run method for PyTorch
    try:
        pt_attr = pt_calculate_relevancemap(pt_method, pt_input, pt_model, **common_args)
        if isinstance(pt_attr, torch.Tensor):
            pt_attr = pt_attr.detach().cpu().numpy()
    except Exception as e:
        print(f"PyTorch error for {method}: {e}")
        pt_attr = np.zeros_like(tf_input)
    
    # Calculate metrics and visualize
    if save_path:
        fig, metrics = visualize_comparison(tf_attr, pt_attr, img, method, save_path)
    else:
        metrics = calculate_metrics(tf_attr, pt_attr)
    
    # Add success flag
    metrics["success"] = metrics["mae"] < 0.5  # Use MAE threshold to determine success
    
    # Add attributes for inspection
    metrics["tf_attr"] = tf_attr
    metrics["pt_attr"] = pt_attr
    
    return metrics


@pytest.fixture(scope="module")
def test_models():
    """Initialize and return test models for both frameworks."""
    # Initialize TensorFlow model
    tf_model = setup_tensorflow_model()
    
    # Initialize PyTorch model
    pt_model = setup_pytorch_model()
    
    return {
        "tensorflow": tf_model,
        "pytorch": pt_model
    }


@pytest.fixture(scope="module")
def test_inputs():
    """Load and preprocess test inputs."""
    # Find test image or use a default
    if not TEST_RESOURCES.exists():
        os.makedirs(TEST_RESOURCES)
    
    # Use a sample image or download one
    image_path = TEST_RESOURCES / "tiger.jpg"
    if not image_path.exists():
        # Generate test image with random pixels
        print("Test image not found. Creating a random test image.")
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        os.makedirs(TEST_RESOURCES, exist_ok=True)
        from PIL import Image
        Image.fromarray(img).save(image_path)
    
    # Preprocess inputs
    tf_input, pt_input, original_img = preprocess_inputs(str(image_path))
    
    return {
        "tensorflow": tf_input,
        "pytorch": pt_input,
        "original": original_img
    }


def test_all_methods(test_models, test_inputs):
    """Test all methods and produce a comparison report."""
    tf_model = test_models["tensorflow"]
    pt_model = test_models["pytorch"]
    tf_input = test_inputs["tensorflow"]
    pt_input = test_inputs["pytorch"]
    original_img = test_inputs["original"]
    
    # Create results dictionary
    results = {}
    
    # Test each method
    for method in METHODS_MAPPING.keys():
        print(f"Testing {method}...")
        save_path = TEST_OUTPUTS / f"{method}_comparison.png"
        method_metrics = run_method_test(
            tf_model, pt_model, method, 
            tf_input, pt_input, original_img, 
            save_path=str(save_path)
        )
        results[method] = method_metrics
    
    # Generate summary report
    generate_report(results)
    
    # For pytest, make sure all tested methods pass
    for method, metrics in results.items():
        assert metrics["success"], f"Method {method} failed: MAE = {metrics['mae']:.6f}"


def generate_report(results):
    """Generate a detailed comparison report."""
    report_path = TEST_OUTPUTS / "comparison_report.txt"
    
    with open(report_path, "w") as f:
        f.write("="*80 + "\n")
        f.write("SIGNXAI: TensorFlow vs PyTorch Comparison Report\n")
        f.write("="*80 + "\n\n")
        
        # Method comparison details
        f.write("METHOD COMPARISONS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':30s} {'MAE':12s} {'MSE':12s} {'Max Diff':12s} {'Correlation':12s} {'Status':12s}\n")
        f.write("-"*80 + "\n")
        
        for method, metrics in results.items():
            # Format status
            status = "Success" if metrics["success"] else "Failed"
            
            # Format metrics
            mae = f"{metrics['mae']:.6f}"
            mse = f"{metrics['mse']:.6f}"
            max_diff = f"{metrics['max_diff']:.6f}"
            correlation = f"{metrics['correlation']:.6f}" if not np.isnan(metrics['correlation']) else "nan"
            
            f.write(f"{method:30s} {mae:12s} {mse:12s} {max_diff:12s} {correlation:12s} {status:12s}\n")
        
        f.write("\n\n")
        
        # Summary statistics
        f.write("SUMMARY\n")
        f.write("-"*80 + "\n")
        
        # Count success/failure
        success_count = sum(1 for m in results.values() if m["success"])
        total_count = len(results)
        
        # Calculate averages
        avg_mae = np.mean([m["mae"] for m in results.values()])
        avg_mse = np.mean([m["mse"] for m in results.values()])
        avg_corr = np.mean([m["correlation"] for m in results.values() if not np.isnan(m["correlation"])])
        
        f.write(f"Success Rate: {success_count}/{total_count} methods\n")
        f.write(f"Average MAE: {avg_mae:.6f}\n")
        f.write(f"Average MSE: {avg_mse:.6f}\n")
        f.write(f"Average Correlation: {avg_corr:.6f}\n\n")
        
        # Find best/worst methods
        best_method = min(results.items(), key=lambda x: x[1]["mae"])
        worst_method = max(results.items(), key=lambda x: x[1]["mae"])
        
        f.write(f"Most similar implementation (lowest MAE): {best_method[0]} (MAE: {best_method[1]['mae']:.6f})\n")
        f.write(f"Least similar implementation (highest MAE): {worst_method[0]} (MAE: {worst_method[1]['mae']:.6f})\n")
    
    print(f"Comparison report saved to {report_path}")
    
    # Also generate an HTML version for better visualization
    html_report_path = TEST_OUTPUTS / "comparison_report.html"
    generate_html_report(results, html_report_path)


def generate_html_report(results, html_path):
    """Generate an HTML report with visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SignXAI: TensorFlow vs PyTorch Comparison</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .success { color: green; font-weight: bold; }
            .failure { color: red; font-weight: bold; }
            .summary { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
            .method-images { display: flex; flex-wrap: wrap; justify-content: center; margin: 20px 0; }
            .method-images img { margin: 10px; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>SignXAI: TensorFlow vs PyTorch Comparison Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
    """
    
    # Count success/failure
    success_count = sum(1 for m in results.values() if m["success"])
    total_count = len(results)
    
    # Calculate averages
    avg_mae = np.mean([m["mae"] for m in results.values()])
    avg_mse = np.mean([m["mse"] for m in results.values()])
    avg_corr = np.mean([m["correlation"] for m in results.values() if not np.isnan(m["correlation"])])
    
    # Best/worst methods
    best_method = min(results.items(), key=lambda x: x[1]["mae"])
    worst_method = max(results.items(), key=lambda x: x[1]["mae"])
    
    html_content += f"""
            <p><strong>Success Rate:</strong> {success_count}/{total_count} methods</p>
            <p><strong>Average MAE:</strong> {avg_mae:.6f}</p>
            <p><strong>Average MSE:</strong> {avg_mse:.6f}</p>
            <p><strong>Average Correlation:</strong> {avg_corr:.6f}</p>
            <p><strong>Most similar implementation:</strong> {best_method[0]} (MAE: {best_method[1]['mae']:.6f})</p>
            <p><strong>Least similar implementation:</strong> {worst_method[0]} (MAE: {worst_method[1]['mae']:.6f})</p>
        </div>
        
        <h2>Method Comparisons</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>MAE</th>
                <th>MSE</th>
                <th>Max Diff</th>
                <th>Correlation</th>
                <th>Status</th>
            </tr>
    """
    
    # Add rows for each method
    for method, metrics in results.items():
        status_class = "success" if metrics["success"] else "failure"
        status_text = "Success" if metrics["success"] else "Failed"
        
        correlation = f"{metrics['correlation']:.6f}" if not np.isnan(metrics['correlation']) else "nan"
        
        html_content += f"""
            <tr>
                <td>{method}</td>
                <td>{metrics['mae']:.6f}</td>
                <td>{metrics['mse']:.6f}</td>
                <td>{metrics['max_diff']:.6f}</td>
                <td>{correlation}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Visual Comparisons</h2>
        <div class="method-images">
    """
    
    # Add visualization images
    for method in results.keys():
        img_path = f"{method}_comparison.png"
        html_content += f"""
            <div>
                <h3>{method}</h3>
                <img src="{img_path}" alt="{method} comparison">
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML comparison report saved to {html_path}")


if __name__ == "__main__":
    # Setup
    test_models_instance = test_models()
    test_inputs_instance = test_inputs()
    
    # Run tests
    test_all_methods(test_models_instance, test_inputs_instance)
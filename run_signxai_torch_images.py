#!/usr/bin/env python3
"""
SignXAI PyTorch Image Explanation Example
==========================================
Demonstrates complex method parsing with parameters, variants, and modifiers.

Example usage:
    # Complex method with epsilon parameter and chained modifiers
    python run_signxai_torch_images.py --method lrp_epsilon_50_x_input_x_sign
    
    # Method with decimal parameter
    python run_signxai_torch_images.py --method lrp_epsilon_0_25_x_input
    
    # Alpha-beta method with parameters
    python run_signxai_torch_images.py --method lrp_alpha_2_beta_1_x_sign
    
    # With custom image
    python run_signxai_torch_images.py --image examples/data/images/example.jpg --method smoothgrad_noise_0_15_samples_50
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("Please install SignXAI2 with PyTorch support:")
    print("  pip install signxai2[pytorch]")
    sys.exit(1)

# Import SignXAI
try:
    from signxai.api import explain
    from signxai.torch_signxai.utils import remove_softmax
    from signxai.common.method_parser import MethodParser
except ImportError as e:
    print(f"ERROR: Failed to import SignXAI: {e}")
    print("Please install SignXAI2:")
    print("  pip install signxai2[pytorch]")
    sys.exit(1)


def demonstrate_method_parsing(method_name):
    """Demonstrate how method parsing works."""
    print("\n" + "="*60)
    print("Method Parsing Demonstration")
    print("="*60)
    print(f"Input method string: '{method_name}'")
    
    # Parse the method to show how it works
    parser = MethodParser()
    parsed = parser.parse(method_name)
    
    print(f"\nParsed components:")
    print(f"  Base method: {parsed['base_method']}")
    print(f"  Modifiers: {parsed['modifiers']}")
    print(f"  Parameters: {parsed['params']}")
    print(f"  Original name: {parsed['original_name']}")
    
    # Show how parameters are extracted
    if parsed['params']:
        print("\nExtracted parameters:")
        for key, value in parsed['params'].items():
            print(f"    {key}: {value} (type: {type(value).__name__})")
    
    # Show how modifiers will be applied
    if parsed['modifiers']:
        print("\nModifiers to be applied (in order):")
        for modifier in parsed['modifiers']:
            if modifier == 'x_input':
                print("    - Multiply by input")
            elif modifier == 'x_sign':
                print("    - Multiply by sign of input")
            elif modifier == 'std_x':
                print("    - Apply standard deviation normalization")
            else:
                print(f"    - Apply {modifier}")
    
    print("="*60)
    return parsed


def load_model(model_path=None):
    """Load and prepare model for XAI."""
    if model_path and os.path.exists(model_path):
        print(f"\nLoading model from: {model_path}")
        model = torch.load(model_path, map_location='cpu')
    else:
        print("\nLoading pretrained VGG16")
        model = models.vgg16(pretrained=True)
    
    model.eval()
    
    # Remove softmax for XAI methods
    print("Removing softmax activation for XAI...")
    model_no_softmax = remove_softmax(model)
    return model_no_softmax


def load_and_preprocess_image(image_path=None):
    """Load and preprocess image for VGG16."""
    # Use example image if none provided
    if not image_path or not os.path.exists(image_path):
        example_path = 'examples/data/images/example.jpg'
        if not os.path.exists(example_path):
            # Download example image
            os.makedirs('examples/data/images', exist_ok=True)
            import urllib.request
            url = 'https://raw.githubusercontent.com/keras-team/keras-applications/master/tests/data/elephant.jpg'
            urllib.request.urlretrieve(url, example_path)
        image_path = example_path
    
    print(f"Loading image: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Preprocess for VGG16
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    x = preprocess(img).unsqueeze(0)  # Add batch dimension
    
    print(f"Image preprocessed: shape {tuple(x.shape)}, dtype {x.dtype}")
    return img.resize((224, 224)), x


def get_prediction(model, x):
    """Get model prediction and target class."""
    print("\nGetting model prediction...")
    
    # Load ImageNet class labels
    import json
    import urllib.request
    
    labels_path = 'imagenet_class_index.json'
    if not os.path.exists(labels_path):
        url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        with urllib.request.urlopen(url) as response:
            labels = json.loads(response.read())
        with open(labels_path, 'w') as f:
            json.dump(labels, f)
    else:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    
    # Get prediction (temporarily add softmax)
    with torch.no_grad():
        output = torch.nn.functional.softmax(model(x), dim=1)
    
    # Get top predictions
    probs, indices = torch.topk(output[0], 3)
    
    print("Top 3 predictions:")
    for i, (prob, idx) in enumerate(zip(probs, indices), 1):
        if isinstance(labels, list):
            class_name = labels[idx.item()] if idx.item() < len(labels) else f"class_{idx.item()}"
        else:
            class_name = labels.get(str(idx.item()), [str(idx.item()), "Unknown"])[1]
        marker = "◄" if i == 1 else " "
        print(f"  {marker} {i}. {class_name}: {prob.item():.2%}")
    
    target_class = indices[0].item()
    if isinstance(labels, list):
        class_name = labels[target_class] if target_class < len(labels) else f"class_{target_class}"
    else:
        class_name = labels.get(str(target_class), [str(target_class), "Unknown"])
        if isinstance(class_name, list):
            class_name = class_name[1]
    
    print(f"\nUsing target class: {target_class} ({class_name})")
    return target_class, class_name


def calculate_explanation(model, x, method_name, parsed_method, target_class):
    """Calculate XAI explanation using the unified API."""
    print(f"\n" + "="*60)
    print("Calculating Explanation")
    print("="*60)
    
    # Show what parameters will be passed to the API
    print(f"Method: {method_name}")
    print(f"Target class: {target_class}")
    print(f"Input shape: {tuple(x.shape)}")
    
    # Prepare method-specific parameters
    kwargs = {}
    
    # Add parsed parameters
    if parsed_method['params']:
        kwargs.update(parsed_method['params'])
        print(f"Method parameters from parsing: {kwargs}")
    
    # Add framework-specific parameters for certain methods
    if 'grad_cam' in parsed_method['base_method'].lower():
        # Find last conv layer for VGG16
        last_conv_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name
        if last_conv_name:
            kwargs['target_layer'] = last_conv_name
            print(f"Added Grad-CAM layer: {kwargs['target_layer']}")
    
    print("\nCalling SignXAI API...")
    
    # Use the unified API
    explanation = explain(
        model=model,
        x=x,
        method_name=method_name,
        target_class=target_class,
        framework='pytorch',
        **kwargs
    )
    
    print(f"Explanation calculated successfully!")
    print(f"Output shape: {explanation.shape}")
    print(f"Output range: [{explanation.min():.3f}, {explanation.max():.3f}]")
    print("="*60)
    
    return explanation


def visualize_explanation(img, explanation, method_name, class_name, parsed_method):
    """Create visualization of the explanation."""
    print("\nCreating visualization...")
    
    # Convert to numpy if needed
    if hasattr(explanation, 'detach'):
        explanation = explanation.detach().cpu().numpy()
    
    # Process explanation for visualization
    if explanation.ndim == 4:
        explanation = explanation[0]  # Remove batch dimension
    
    # Create 2D heatmap
    if explanation.ndim == 3:
        # For PyTorch format (C, H, W), sum over channels
        heatmap = np.sum(explanation, axis=0)
        print(f"Aggregated {explanation.shape[0]} channels to create heatmap")
    else:
        heatmap = explanation
    
    # Normalize to [-1, 1]
    abs_max = np.max(np.abs(heatmap))
    if abs_max > 0:
        heatmap = heatmap / abs_max
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Explanation heatmap
    im = axes[1].imshow(heatmap, cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title(f'{method_name}')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # Overlay
    axes[2].imshow(img, alpha=0.7)
    axes[2].imshow(heatmap, cmap='seismic', alpha=0.3, vmin=-1, vmax=1)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add method info as subtitle
    info_text = f"Base: {parsed_method['base_method']}"
    if parsed_method['params']:
        params_str = ", ".join([f"{k}={v}" for k, v in parsed_method['params'].items()])
        info_text += f" | Params: {params_str}"
    if parsed_method['modifiers']:
        info_text += f" | Modifiers: {', '.join(parsed_method['modifiers'])}"
    
    plt.suptitle(f'SignXAI PyTorch: {class_name}\n{info_text}', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    os.makedirs('outputs', exist_ok=True)
    safe_method_name = method_name.replace("/", "_").replace(".", "_")
    output_path = f'outputs/pt_{safe_method_name}.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='SignXAI PyTorch - Demonstrating Complex Method Parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Method Name Format:
  The method name can include the base method, parameters, and modifiers:
  
  Format: <base_method>_<param1>_<value1>_<modifier1>_<modifier2>
  
  Examples:
    lrp_epsilon_50_x_input_x_sign  - LRP with epsilon=50, multiply by input and sign
    lrp_alpha_2_beta_1              - LRP with alpha=2, beta=1
    smoothgrad_noise_0_15_samples_50 - SmoothGrad with noise=0.15, samples=50
    gradient_x_input                - Gradient multiplied by input
    integrated_gradients_steps_100  - Integrated Gradients with 100 steps
    
  Parameters are automatically extracted:
    - epsilon_50 → epsilon=50.0
    - epsilon_0_25 → epsilon=0.25
    - alpha_2_beta_1 → alpha=2.0, beta=1.0
    - noise_0_15 → noise_level=0.15
    - samples_50 → num_samples=50
    
  Modifiers (applied in order):
    - x_input: multiply by input
    - x_sign: multiply by sign of input
    - std_x: standard deviation normalization
        """
    )
    
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (uses example if not specified)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (uses VGG16 if not specified)')
    parser.add_argument('--method', type=str, default='lrp_epsilon_50_x_input_x_sign',
                        help='XAI method name with parameters and modifiers')
    
    args = parser.parse_args()
    
    print("SignXAI PyTorch Explanation Demo")
    print("=================================")
    
    # Demonstrate method parsing
    parsed_method = demonstrate_method_parsing(args.method)
    
    # Load model
    model = load_model(args.model)
    
    # Load and preprocess image
    img, x = load_and_preprocess_image(args.image)
    
    # Get prediction
    target_class, class_name = get_prediction(model, x)
    
    # Calculate explanation
    explanation = calculate_explanation(model, x, args.method, parsed_method, target_class)
    
    # Visualize results
    visualize_explanation(img, explanation, args.method, class_name, parsed_method)
    
    print("\n" + "="*60)
    print("Explanation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
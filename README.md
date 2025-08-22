# SignXAI2

Cross-framework (TensorFlow + PyTorch) implementation of state-of-the-art XAI methods with dynamic method parsing.

## Key Features

✨ **Dynamic Method Parsing**: Parameters embedded in method names  
🔄 **Unified API**: Same interface for TensorFlow and PyTorch  
🚀 **No Wrappers**: Direct method calls for better performance  
📊 **Time Series Support**: Full ECG and time series visualization  

## Installation

```bash
pip install signxai2[all]  # Install with both frameworks
# OR
pip install signxai2[tensorflow]  # TensorFlow only
pip install signxai2[pytorch]  # PyTorch only
```

## Quick Start

### Image Classification

```python
from signxai.api import explain

# Basic gradient
explanation = explain(model, image, method_name="gradient")

# SmoothGrad with custom parameters
explanation = explain(model, image, method_name="smoothgrad_noise_0_3_samples_50")

# Complex combination
explanation = explain(model, image, method_name="gradient_x_input_x_sign_mu_neg_0_5")
```

### Time Series (ECG)

```python
from signxai.api import explain
from utils.ecg_visualization import plot_ecg

# Generate explanation for ECG
explanation = explain(model, ecg_data, method_name="gradient_x_input")

# Visualize with 12-lead plot
plot_ecg(ecg_data, explanation)
```

## Dynamic Method Parsing

Parameters are embedded directly in method names:

| Method Name | Description |
|------------|-------------|
| `gradient` | Basic gradient |
| `gradient_x_input` | Gradient × Input |
| `smoothgrad_noise_0_3_samples_50` | SmoothGrad (noise=0.3, samples=50) |
| `integrated_gradients_steps_100` | Integrated Gradients (100 steps) |
| `lrp_epsilon_0_25` | LRP (ε=0.25) |
| `gradient_x_input_x_sign_mu_neg_0_5` | Complex combination |

## Supported Methods

### Gradient-based
- gradient, smoothgrad, integrated_gradients, vargrad

### Backpropagation
- guided_backprop, deconvnet

### LRP Family
- lrp_epsilon, lrp_alpha_beta, lrp_gamma, lrp_z

### Feature Methods
- grad_cam, grad_cam++

## Examples

See the `quickstart_*.py` files for complete examples:
- `quickstart_tf_images.py` - TensorFlow image classification
- `quickstart_tf_timeseries.py` - TensorFlow ECG analysis
- `quickstart_torch_images.py` - PyTorch image classification
- `quickstart_torch_timeseries.py` - PyTorch ECG analysis

## Documentation

- [API Reference](API_REFERENCE.md)
- [Changelog](CHANGELOG.md)
- [Tutorials](examples/tutorials/)

## License

MIT License - See LICENSE file for details.

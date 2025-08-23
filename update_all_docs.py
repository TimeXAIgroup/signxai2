#!/usr/bin/env python
"""
Script to update all documentation files to use the new dynamic method parsing approach.
This script should be run from the signxai2 project root directory.
"""

import os
import json

# Get the current directory (should be project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def update_notebooks():
    """Update all Jupyter notebooks to use new dynamic parsing."""
    
    # PyTorch Advanced Usage Notebook
    pytorch_advanced = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SignXAI2 with PyTorch - Advanced Usage\n\n",
                    "This tutorial covers advanced features using dynamic method parsing.\n\n",
                    "## Advanced Features:\n",
                    "- Complex method combinations\n",
                    "- Custom parameter tuning\n",
                    "- Model-specific optimizations\n",
                    "- Batch processing"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import numpy as np\n",
                    "from signxai.api import explain\n",
                    "from signxai.torch_signxai.utils import remove_softmax"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Advanced Method Combinations\n\n",
                    "Dynamic parsing allows complex combinations:"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Complex combinations with parameters\n",
                    "advanced_methods = [\n",
                    "    'gradient_x_input_x_sign_mu_neg_0_5',\n",
                    "    'lrp_epsilon_50_x_sign',\n",
                    "    'lrpsign_epsilon_0_25_std_x',\n",
                    "    'smoothgrad_noise_0_3_samples_50_x_sign',\n",
                    "    'integrated_gradients_steps_100_x_input'\n",
                    "]\n\n",
                    "# Apply each method\n",
                    "for method in advanced_methods:\n",
                    "    explanation = explain(model, input_tensor, method_name=method)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # PyTorch Time Series Notebook
    pytorch_timeseries = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SignXAI2 PyTorch - Time Series (ECG) Analysis\n\n",
                    "This tutorial demonstrates ECG time series analysis with dynamic method parsing.\n\n",
                    "## Features:\n",
                    "- 12-lead ECG visualization\n",
                    "- Dynamic method parameters\n",
                    "- Time series specific methods"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import torch\n",
                    "import numpy as np\n",
                    "from signxai.api import explain\n",
                    "from signxai.torch_signxai.utils import remove_softmax\n",
                    "from utils.ecg_data import load_and_preprocess_ecg\n",
                    "from utils.ecg_visualization import plot_ecg"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Dynamic method examples for ECG\n",
                    "ecg_methods = [\n",
                    "    'gradient',\n",
                    "    'gradient_x_input_x_sign_mu_neg_0_5',\n",
                    "    'smoothgrad_noise_0_3_samples_50',\n",
                    "    'integrated_gradients_steps_100',\n",
                    "    'lrp_epsilon_0_25'\n",
                    "]\n\n",
                    "# Load ECG data\n",
                    "ecg_data = load_and_preprocess_ecg(record_id='03509_hr')\n\n",
                    "# Generate explanations\n",
                    "for method in ecg_methods:\n",
                    "    explanation = explain(model, ecg_data, method_name=method)\n",
                    "    plot_ecg(ecg_data, explanation, title=method)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # TensorFlow Basic Usage Notebook
    tensorflow_basic = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SignXAI2 with TensorFlow - Basic Usage\n\n",
                    "This tutorial demonstrates TensorFlow/Keras models with dynamic method parsing.\n\n",
                    "## Key Features:\n",
                    "- Dynamic method parsing\n",
                    "- Unified API\n",
                    "- No wrapper functions"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import tensorflow as tf\n",
                    "import numpy as np\n",
                    "from signxai.api import explain\n",
                    "from signxai.utils.utils import remove_softmax"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load a pre-trained model\n",
                    "from tensorflow.keras.applications import VGG16\n",
                    "model = VGG16(weights='imagenet')\n\n",
                    "# Dynamic method examples\n",
                    "methods = [\n",
                    "    'gradient',\n",
                    "    'gradient_x_input',\n",
                    "    'smoothgrad_noise_0_3_samples_50',\n",
                    "    'integrated_gradients_steps_100',\n",
                    "    'lrp_epsilon_0_25'\n",
                    "]\n\n",
                    "# Generate explanations\n",
                    "for method in methods:\n",
                    "    explanation = explain(model, input_image, method_name=method)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # TensorFlow Advanced Usage Notebook
    tensorflow_advanced = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SignXAI2 with TensorFlow - Advanced Usage\n\n",
                    "Advanced TensorFlow features with dynamic method parsing.\n\n",
                    "## Topics:\n",
                    "- Complex method combinations\n",
                    "- Custom layer handling\n",
                    "- Performance optimization"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import tensorflow as tf\n",
                    "from signxai.api import explain\n",
                    "from signxai.utils.utils import remove_softmax"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Advanced method combinations\n",
                    "complex_methods = [\n",
                    "    'gradient_x_input_x_sign_mu_neg_0_5',\n",
                    "    'lrp_epsilon_50_x_sign',\n",
                    "    'smoothgrad_noise_0_5_samples_100_x_input'\n",
                    "]\n\n",
                    "for method in complex_methods:\n",
                    "    explanation = explain(model, input_data, method_name=method)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # TensorFlow Time Series Notebook
    tensorflow_timeseries = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SignXAI2 TensorFlow - Time Series (ECG) Analysis\n\n",
                    "ECG analysis with TensorFlow using dynamic method parsing.\n\n",
                    "## Features:\n",
                    "- 12-lead ECG support\n",
                    "- Dynamic parameters\n",
                    "- Real-time visualization"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import tensorflow as tf\n",
                    "import numpy as np\n",
                    "from signxai.api import explain\n",
                    "from signxai.utils.utils import remove_softmax\n",
                    "from utils.ecg_data import load_and_preprocess_ecg\n",
                    "from utils.ecg_visualization import plot_ecg"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ECG-specific methods with dynamic parameters\n",
                    "ecg_methods = [\n",
                    "    'gradient',\n",
                    "    'gradient_x_input_x_sign_mu_neg_0_5',\n",
                    "    'smoothgrad_noise_0_3_samples_50',\n",
                    "    'integrated_gradients_steps_100'\n",
                    "]\n\n",
                    "# Process ECG data\n",
                    "for method in ecg_methods:\n",
                    "    explanation = explain(model, ecg_data, method_name=method)\n",
                    "    plot_ecg(ecg_data, explanation, title=f'TensorFlow: {method}')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save all notebooks with relative paths
    notebooks = {
        "examples/tutorials/pytorch/pytorch_advanced_usage.ipynb": pytorch_advanced,
        "examples/tutorials/pytorch/pytorch_time_series.ipynb": pytorch_timeseries,
        "examples/tutorials/tensorflow/tensorflow_basic_usage.ipynb": tensorflow_basic,
        "examples/tutorials/tensorflow/tensorflow_advanced_usage.ipynb": tensorflow_advanced,
        "examples/tutorials/tensorflow/tensorflow_time_series.ipynb": tensorflow_timeseries
    }

    for rel_path, content in notebooks.items():
        full_path = os.path.join(BASE_DIR, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            json.dump(content, f, indent=2)
        print(f"Updated: {rel_path}")

def update_documentation():
    """Update markdown documentation files."""
    
    # Update API_REFERENCE.md
    api_reference = """# SignXAI2 API Reference

## Core API

### `explain(model, x, method_name, target_class=None, **kwargs)`

Generate explanations using dynamic method parsing.

#### Parameters:
- `model`: TensorFlow or PyTorch model
- `x`: Input data (numpy array or tensor)
- `method_name`: Method name with embedded parameters
- `target_class`: Target class for explanation (optional)

#### Dynamic Method Parsing

Parameters are embedded directly in method names:

##### Basic Methods:
- `gradient` - Basic gradient
- `gradient_x_input` - Gradient × Input
- `gradient_x_sign` - Gradient × Sign(Input)

##### Methods with Parameters:
- `smoothgrad_noise_0_3_samples_50` - SmoothGrad with noise=0.3, samples=50
- `integrated_gradients_steps_100` - Integrated Gradients with 100 steps
- `lrp_epsilon_0_25` - LRP with epsilon=0.25
- `lrp_alpha_2_beta_1` - LRP with α=2, β=1

##### Complex Combinations:
- `gradient_x_input_x_sign_mu_neg_0_5` - Multiple operations with parameters
- `lrp_epsilon_50_x_sign` - LRP with transformation
- `lrpsign_epsilon_0_25_std_x` - LRP-Sign with normalization

#### Examples:

```python
from signxai.api import explain

# Basic gradient
explanation = explain(model, x, method_name="gradient")

# SmoothGrad with custom parameters
explanation = explain(model, x, method_name="smoothgrad_noise_0_3_samples_50")

# Complex combination
explanation = explain(model, x, method_name="gradient_x_input_x_sign_mu_neg_0_5")
```

## Framework-Specific Utilities

### TensorFlow
```python
from signxai.utils.utils import remove_softmax
model_no_softmax = remove_softmax(model)
```

### PyTorch
```python
from signxai.torch_signxai.utils import remove_softmax
model_no_softmax = remove_softmax(model)
```

## Method Name Format

```
base_method[_param_value][_operation][_param_value]...
```

### Supported Operations:
- `x_input` - Multiply by input
- `x_sign` - Multiply by sign
- `x_sign_mu_neg_VALUE` - Sign with threshold

### Supported Parameters:
- `noise_VALUE` - Noise level (SmoothGrad)
- `samples_VALUE` - Number of samples
- `steps_VALUE` - Integration steps
- `epsilon_VALUE` - Epsilon value (LRP)
- `alpha_VALUE_beta_VALUE` - Alpha-beta parameters (LRP)

## Batch Processing

```python
# Process multiple inputs
for input_batch in data_loader:
    explanations = explain(model, input_batch, method_name="gradient")
```

## Time Series Support

For ECG and time series data:

```python
from utils.ecg_data import load_and_preprocess_ecg
from utils.ecg_visualization import plot_ecg

# Load ECG data
ecg_data = load_and_preprocess_ecg(record_id="03509_hr")

# Generate explanation
explanation = explain(model, ecg_data, method_name="gradient_x_input")

# Visualize with 12-lead plot
plot_ecg(ecg_data, explanation)
```
"""

    # Update CHANGELOG.md
    changelog = """# Changelog

## [2.0.0] - 2024-01-22

### Added
- **Dynamic Method Parsing**: Parameters are now embedded directly in method names
- **Unified API**: Single `explain()` function for all methods
- **12-Lead ECG Support**: Full support for multi-channel time series visualization
- **Method Combinations**: Support for complex method combinations like `gradient_x_input_x_sign_mu_neg_0_5`

### Changed
- **Removed Wrapper Functions**: Direct method calls without intermediate wrappers
- **Simplified API**: All methods now use the same unified interface
- **Improved Performance**: Optimized method parsing and execution

### Deprecated
- `wrapper.py` functionality replaced by dynamic parsing
- Old parameter passing style (use embedded parameters instead)

### Examples
```python
# Old style (deprecated):
explain(model, x, method="smoothgrad", noise_level=0.3, num_samples=50)

# New style:
explain(model, x, method_name="smoothgrad_noise_0_3_samples_50")
```

## [1.0.0] - Previous Version
- Initial release with wrapper-based API
"""

    # Update README.md
    readme = """# SignXAI2

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
"""

    # Save documentation files
    docs = {
        "API_REFERENCE.md": api_reference,
        "CHANGELOG.md": changelog,
        "README.md": readme
    }

    for filename, content in docs.items():
        path = os.path.join(BASE_DIR, filename)
        with open(path, 'w') as f:
            f.write(content)
        print(f"Updated: {filename}")

def main():
    """Main function to update all documentation."""
    print("Updating SignXAI2 documentation for dynamic method parsing...")
    print(f"Working directory: {BASE_DIR}")
    print()
    
    print("Updating Jupyter notebooks...")
    update_notebooks()
    print()
    
    print("Updating documentation files...")
    update_documentation()
    print()
    
    print("✅ All documentation files updated successfully!")
    print("\nNext steps:")
    print("1. Review the updated files")
    print("2. Run: bash update_sphinx_docs.sh")
    print("3. Commit the changes")

if __name__ == "__main__":
    main()
# SignXAI2

## Project Description

**SIGNed explanations: Unveiling relevant features by reducing bias**

This repository and python package is an extended version of the published python package of the following journal article: https://doi.org/10.1016/j.inffus.2023.101883

If you use the code from this repository in your work, please cite:

```bibtex
@article{Gumpfer2023SIGN,
    title = {SIGNed explanations: Unveiling relevant features by reducing bias},
    author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
    journal = {Information Fusion},
    pages = {101883},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2023.101883},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253523001999}
}
```

## Documentation

The documentation for SignXAI2 is available at: https://TimeXAIgroup.github.io/signxai2/index.html

## Requirements

- Python 3.9 or 3.10 (Python 3.11+ is not supported)
- TensorFlow >=2.8.0,<=2.12.1
- PyTorch >=1.10.0
- NumPy, Matplotlib, SciPy

## 🚀 Installation

SignXAI2 requires you to explicitly choose which deep learning framework(s) to install. This ensures you only install what you need.

### Install from PyPI

For TensorFlow users:
```bash
pip install signxai2[tensorflow]
```

For PyTorch users:
```bash
pip install signxai2[pytorch]
```

For both frameworks:
```bash
pip install signxai2[all]
```

For development (includes all frameworks + dev tools):
```bash
pip install signxai2[dev]
```

**Note:** Installing `pip install signxai2` alone is not supported. You must specify at least one framework.

### Install from source

```bash
git clone https://github.com/TimeXAIgroup/signxai2.git
cd signxai2

# Choose your installation:
pip install -e .[tensorflow]    # TensorFlow only
pip install -e .[pytorch]       # PyTorch only  
pip install -e .[all]           # Both frameworks
pip install -e .[dev]           # Development (all frameworks + tools)
```

## Setup of Git LFS

Before you get started please set up Git LFS to download the large files in this repository. This is required to access the pre-trained models and example data.

```bash
git lfs install
```

## 📦 Load Data and Documentation

After installation, run the setup script to download documentation, examples, and sample data:

```bash
bash ./prepare.sh
```

This will download:
- 📚 Full documentation (viewable at docs/index.html)
- 📝 Example scripts and notebooks (examples/)
- 📊 Sample ECG data and images (examples/data/)

## Examples

To get started with SignXAI2 Methods, please follow the example tutorials ('examples/tutorials/').

## Features

- Support for TensorFlow and PyTorch models
- Consistent API across frameworks with dynamic method parsing
- Wide range of explanation methods:
  - Gradient-based: Vanilla gradient, Integrated gradients, SmoothGrad
  - Class activation maps: Grad-CAM
  - Guided backpropagation
  - Layer-wise Relevance Propagation (LRP)
  - Sign-based thresholding for binary relevance maps
- No wrapper classes - direct method calls with parameters embedded in method names

## Development version

To install with development dependencies for testing and documentation:

```bash
pip install signxai2[dev]
```

Or from source:
```bash
git clone https://github.com/TimeXAIgroup/signxai2.git
cd signxai2
pip install -e ".[dev]"
```

## Project Structure

- `signxai/`: Main package with unified API and framework detection
- `signxai/tf_signxai/`: TensorFlow implementation using modified iNNvestigate
- `signxai/torch_signxai/`: PyTorch implementation using zennit with custom hooks
- `examples/tutorials/`: Tutorials for both frameworks covering images and time series
- `examples/comparison/`: Implementation for reproducing results from the paper
- `utils/`: Helper scripts for model conversion (tf -> torch) and data preprocessing

## Usage

Please follow the example tutorials in the `examples/tutorials/` directory to get started with SignXAI2 methods. The new API uses dynamic method parsing where parameters are embedded directly in method names:

```python
from signxai import explain

# Basic gradient
explanation = explain(model, x, method_name="gradient")

# Complex method with parameter chaining
explanation = explain(model, x, method_name="gradient_x_input_x_sign_mu_neg_0_5")
```

## License

BSD 3-Clause License - See LICENSE file for details.

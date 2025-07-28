# SIGNed explanations: Unveiling relevant features by reducing bias

This repository and python package is an extended version of the published python package of the following journal article:
https://doi.org/10.1016/j.inffus.2023.101883

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

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1566253523001999-ga1_lrg.jpg" title="Graphical Abstract" width="900px"/>

## 🚀 Installation

### Install from PyPI
```bash
pip install signxai2
```

This installs the complete package with both TensorFlow and PyTorch support.

### Install from source

#### Option 1: Full installation (both frameworks)
```bash
git clone https://github.com/IRISlaboratory/signxai2.git
cd signxai2
pip install -e .
```

#### Option 2: Framework-specific installation
For users who want to install only specific framework support:

**TensorFlow only:**
```bash
git clone https://github.com/IRISlaboratory/signxai2.git
cd signxai2
pip install -r requirements/common.txt -r requirements/tensorflow.txt
pip install -e . --no-deps
```

**PyTorch only:**
```bash
git clone https://github.com/IRISlaboratory/signxai2.git
cd signxai2
pip install -r requirements/common.txt -r requirements/pytorch.txt
pip install -e . --no-deps
```

Note: Framework-specific installation is only available when installing from source. The PyPI package includes both frameworks for seamless compatibility.

## Setup of Git LFS

Before you get started please set up [Git LFS](https://git-lfs.github.com/) to download the large files in this repository. This is required to access the pre-trained models and example data.

```bash
git lfs install
```

## 📦 Load Data and Documentation

After installation, run the setup script to download documentation, examples, and sample data:

```bash
bash ./prepare.sh
```

This will download:
- 📚 Full documentation (viewable at `docs/index.html`)
- 📝 Example scripts and notebooks (`examples/`)  
- 📊 Sample ECG data and images (`examples/data/`)


## Examples

To get started with SignXAI2 Methods, please follow the example tutorials ('examples/tutorials/').

## Features

- Support for **TensorFlow** and **PyTorch** models
- Consistent API across frameworks
- Wide range of explanation methods:
  - Gradient-based: Vanilla gradient, Integrated gradients, SmoothGrad
  - Class activation maps: Grad-CAM
  - Guided backpropagation
  - Layer-wise Relevance Propagation (LRP)
  - Sign-based thresholding for binary relevance maps


### Development version

To install with development dependencies for testing and documentation:

```shell
pip install signxai2[dev]
```

Or from source:
```shell
git clone https://github.com/IRISlaboratory/signxai2.git
cd signxai2
pip install -e ".[dev]"
```

##  Project Structure

  - signxai/: Main package with unified API and framework detection
  - signxai/tf_signxai/: TensorFlow implementation using modified iNNvestigate
  - signxai/torch_signxai/: PyTorch implementation using zennit with custom hooks
  - examples/tutorials/: Tutorials for both frameworks covering images and time series
  - examples/comparison/: Implementation for reproducing results from the paper
  - utils/: Helper scripts for model conversion (tf -> torch) and data preprocessing


## Usage

Please follow the example tutorials in the `examples/tutorials/` directory to get started with SignXAI2 methods. The examples cover various use cases, including images and time series analysis.


## Methods

| Method | Base| Parameters |
|--------|-----------------------------------------|--------------------------------|
| gradient | Gradient | |
| input_t_gradient | Gradient x Input | |
| gradient_x_input | Gradient x Input | |
| gradient_x_sign | Gradient x SIGN  | mu = 0 |
| gradient_x_sign_mu | Gradient x SIGN  | requires *mu* parameter |
| gradient_x_sign_mu_0 | Gradient x SIGN  | mu = 0 |
| gradient_x_sign_mu_0_5 | Gradient x SIGN  | mu = 0.5 |
| gradient_x_sign_mu_neg_0_5 | Gradient x SIGN  | mu = -0.5 |
| guided_backprop | Guided Backpropagation | |
| guided_backprop_x_sign | Guided Backpropagation x SIGN  | mu = 0 |
| guided_backprop_x_sign_mu | Guided Backpropagation x SIGN  | requires *mu* parameter |
| guided_backprop_x_sign_mu_0 | Guided Backpropagation x SIGN  | mu = 0 |
| guided_backprop_x_sign_mu_0_5 | Guided Backpropagation x SIGN  | mu = 0.5 |
| guided_backprop_x_sign_mu_neg_0_5 | Guided Backpropagation x SIGN  | mu = -0.5 |
| integrated_gradients | Integrated Gradients | |
| smoothgrad | SmoothGrad | |
| smoothgrad_x_sign | SmoothGrad x SIGN  | mu = 0 |
| smoothgrad_x_sign_mu | SmoothGrad x SIGN  | requires *mu* parameter |
| smoothgrad_x_sign_mu_0 | SmoothGrad x SIGN  | mu = 0 |
| smoothgrad_x_sign_mu_0_5 | SmoothGrad x SIGN  | mu = 0.5  |
| smoothgrad_x_sign_mu_neg_0_5 | SmoothGrad x SIGN  | mu = -0.5  |
| vargrad | VarGrad  | |
| deconvnet | DeconvNet  | |
| deconvnet_x_sign | DeconvNet x SIGN | mu = 0 |
| deconvnet_x_sign_mu | DeconvNet x SIGN | requires *mu* parameter |
| deconvnet_x_sign_mu_0 | DeconvNet x SIGN | mu = 0 |
| deconvnet_x_sign_mu_0_5 | DeconvNet x SIGN | mu = 0.5 |
| deconvnet_x_sign_mu_neg_0_5 | DeconvNet x SIGN | mu = -0.5 |
| grad_cam | Grad-CAM| requires *last_conv* parameter |
| grad_cam_timeseries | Grad-CAM| (for time series data), requires *last_conv* parameter |
| grad_cam_VGG16ILSVRC | | *last_conv* based on VGG16 |
| guided_grad_cam_VGG16ILSVRC | | *last_conv* based on VGG16 |
| lrp_z | LRP-z  | |
| lrpsign_z | LRP-z / LRP-SIGN (Inputlayer-Rule) | |
| zblrp_z_VGG16ILSVRC | LRP-z / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet |
| w2lrp_z | LRP-z / LRP-w² (Inputlayer-Rule) | |
| flatlrp_z | LRP-z / LRP-flat (Inputlayer-Rule) | |
| lrp_epsilon_0_001 | LRP-epsilon | epsilon = 0.001 |
| lrpsign_epsilon_0_001 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.001 |
| zblrp_epsilon_0_001_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.001 |
| lrpz_epsilon_0_001 |LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.001 |
| lrp_epsilon_0_01 | LRP-epsilon | epsilon = 0.01 |
| lrpsign_epsilon_0_01 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.01 |
| zblrp_epsilon_0_01_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.01 |
| lrpz_epsilon_0_01 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.01 |
| w2lrp_epsilon_0_01 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 0.01 |
| flatlrp_epsilon_0_01 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 0.01 |
| lrp_epsilon_0_1 | LRP-epsilon | epsilon = 0.1 |
| lrpsign_epsilon_0_1 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.1 |
| zblrp_epsilon_0_1_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.1 |
| lrpz_epsilon_0_1 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.1 |
| w2lrp_epsilon_0_1 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 0.1 |
| flatlrp_epsilon_0_1 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 0.1 |
| lrp_epsilon_0_2 | LRP-epsilon | epsilon = 0.2 |
| lrpsign_epsilon_0_2 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.2 |
| zblrp_epsilon_0_2_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.2 |
| lrpz_epsilon_0_2 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.2 |
| lrp_epsilon_0_5 | LRP-epsilon | epsilon = 0.5 |
| lrpsign_epsilon_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.5 |
| zblrp_epsilon_0_5_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.5 |
| lrpz_epsilon_0_5 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 0.5 |
| lrp_epsilon_1 | LRP-epsilon | epsilon = 1 |
| lrpsign_epsilon_1 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 1 |
| zblrp_epsilon_1_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 1 |
| lrpz_epsilon_1 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 1 |
| w2lrp_epsilon_1 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 1 |
| flatlrp_epsilon_1 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 1 |
| lrp_epsilon_5 | LRP-epsilon | epsilon = 5 |
| lrpsign_epsilon_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 5 |
| zblrp_epsilon_5_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 5 |
| lrpz_epsilon_5 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 5 |
| lrp_epsilon_10 | LRP-epsilon | epsilon = 10 |
| lrpsign_epsilon_10 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 10 |
| zblrp_epsilon_10_VGG106ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 10 |
| lrpz_epsilon_10 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 10 |
| w2lrp_epsilon_10 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 10 |
| flatlrp_epsilon_10 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 10 |
| lrp_epsilon_20 | LRP-epsilon | epsilon = 20 |
| lrpsign_epsilon_20 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 20 |
| zblrp_epsilon_20_VGG206ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 20 |
| lrpz_epsilon_20 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 20 |
| w2lrp_epsilon_20 | LRP-epsilon / LRP-w² (Inputlayer-Rule)  | epsilon = 20 |
| flatlrp_epsilon_20 | LRP-epsilon / LRP-flat (Inputlayer-Rule)  | epsilon = 20 |
| lrp_epsilon_50 | LRP-epsilon | epsilon = 50 |
| lrpsign_epsilon_50 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 50 |
| lrpz_epsilon_50 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 50 |
| lrp_epsilon_75 | LRP-epsilon | epsilon = 75 |
| lrpsign_epsilon_75 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 75 |
| lrpz_epsilon_75 | LRP-epsilon / LRP-z (Inputlayer-Rule)  | epsilon = 75 |
| lrp_epsilon_100 | LRP-epsilon | epsilon = 100 |
| lrpsign_epsilon_100 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = 0 |
| lrpsign_epsilon_100_mu_0 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = 0 |
| lrpsign_epsilon_100_mu_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = 0.5 |
| lrpsign_epsilon_100_mu_neg_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 100, mu = -0.5 |
| lrpz_epsilon_100 | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 100 |
| zblrp_epsilon_100_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 100 |
| w2lrp_epsilon_100 | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 100 |
| flatlrp_epsilon_100 | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 100 |
| lrp_epsilon_0_1_std_x | LRP-epsilon | epsilon = 0.1 * std(x) |
| lrpsign_epsilon_0_1_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| lrpz_epsilon_0_1_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| zblrp_epsilon_0_1_std_x_VGG16ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.1 * std(x) |
| w2lrp_epsilon_0_1_std_x | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| flatlrp_epsilon_0_1_std_x | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 0.1 * std(x) |
| lrp_epsilon_0_25_std_x | LRP-epsilon | epsilon = 0.25 * std(x) |
| lrpsign_epsilon_0_25_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = 0 |
| lrpz_epsilon_0_25_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 0.25 * std(x) |
| zblrp_epsilon_0_25_std_x_VGG256ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.25 * std(x) |
| w2lrp_epsilon_0_25_std_x | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 0.25 * std(x) |
| flatlrp_epsilon_0_25_std_x | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 0.25 * std(x) |
| lrpsign_epsilon_0_25_std_x_mu_0 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = 0 |
| lrpsign_epsilon_0_25_std_x_mu_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = 0.5 |
| lrpsign_epsilon_0_25_std_x_mu_neg_0_5 | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.25 * std(x), mu = -0.5 |
| lrp_epsilon_0_5_std_x | LRP-epsilon | epsilon = 0.5 * std(x) |
| lrpsign_epsilon_0_5_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| lrpz_epsilon_0_5_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| zblrp_epsilon_0_5_std_x_VGG56ILSVRC | LRP-epsilon / LRP-ZB (Inputlayer-Rule) | bounds based on ImageNet, epsilon = 0.5 * std(x) |
| w2lrp_epsilon_0_5_std_x | LRP-epsilon / LRP-w² (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| flatlrp_epsilon_0_5_std_x | LRP-epsilon / LRP-flat (Inputlayer-Rule) | epsilon = 0.5 * std(x) |
| lrp_epsilon_1_std_x | LRP-epsilon | epsilon = 1 * std(x) |
| lrpsign_epsilon_1_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 1 * std(x), mu = 0 |
| lrpz_epsilon_1_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 1 * std(x) |
| lrp_epsilon_2_std_x | LRP-epsilon | epsilon = 2 * std(x) |
| lrpsign_epsilon_2_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 2 * std(x), mu = 0 |
| lrpz_epsilon_2_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 2 * std(x) |
| lrp_epsilon_3_std_x | LRP-epsilon | epsilon = 3 * std(x) |
| lrpsign_epsilon_3_std_x | LRP-epsilon / LRP-SIGN (Inputlayer-Rule) | epsilon = 3 * std(x), mu = 0 |
| lrpz_epsilon_3_std_x | LRP-epsilon / LRP-z (Inputlayer-Rule) | epsilon = 3 * std(x) |
| lrp_alpha_1_beta_0 | LRP-alpha-beta | alpha = 1, beta = 0 |
| lrpsign_alpha_1_beta_0 | LRP-alpha-beta / LRP-SIGN (Inputlayer-Rule) | alpha = 1, beta = 0, mu = 0 |
| lrpz_alpha_1_beta_0 | LRP-alpha-beta / LRP-z (Inputlayer-Rule) | alpha = 1, beta = 0 |
| zblrp_alpha_1_beta_0_VGG16ILSVRC |  | bounds based on ImageNet, alpha = 1, beta = 0 |
| w2lrp_alpha_1_beta_0 | LRP-alpha-beta / LRP-ZB (Inputlayer-Rule) | alpha = 1, beta = 0 |
| flatlrp_alpha_1_beta_0 | LRP-alpha-beta / LRP-flat (Inputlayer-Rule) | alpha = 1, beta = 0 |
| lrp_sequential_composite_a | LRP Comosite Variant A |  |
| lrpsign_sequential_composite_a | LRP Comosite Variant A / LRP-SIGN (Inputlayer-Rule) |  mu = 0 |
| lrpz_sequential_composite_a | LRP Comosite Variant A / LRP-z (Inputlayer-Rule) |  |
| zblrp_sequential_composite_a_VGG16ILSVRC |  | bounds based on ImageNet  |
| w2lrp_sequential_composite_a | LRP Comosite Variant A / LRP-ZB (Inputlayer-Rule) |  |
| flatlrp_sequential_composite_a | LRP Comosite Variant A / LRP-flat (Inputlayer-Rule) |  |
| lrp_sequential_composite_b | LRP Comosite Variant B |  |
| lrpsign_sequential_composite_b | LRP Comosite Variant B / LRP-SIGN (Inputlayer-Rule) |  mu = 0 |
| lrpz_sequential_composite_b | LRP Comosite Variant B / LRP-z (Inputlayer-Rule) |  |
| zblrp_sequential_composite_b_VGG16ILSVRC |  | bounds based on ImageNet  |
| w2lrp_sequential_composite_b | LRP Comosite Variant B / LRP-ZB (Inputlayer-Rule) |  |
| flatlrp_sequential_composite_b | LRP Comosite Variant B / LRP-flat (Inputlayer-Rule) |  |

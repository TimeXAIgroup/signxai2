# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2025-07-28

### Added

#### Core Features
- **Unified API**: Framework-agnostic `signxai.explain()` function that automatically detects TensorFlow or PyTorch models
- **Cross-framework compatibility**: Consistent API and results across TensorFlow and PyTorch implementations
- **200+ XAI methods**: Comprehensive collection of explainable AI methods including gradients, LRP variants, CAM methods, and backpropagation techniques

#### Framework Support
- **TensorFlow integration**: Full support for TensorFlow/Keras models with iNNvestigate backend for LRP methods
- **PyTorch integration**: Complete PyTorch support with Zennit backend for LRP implementations
- **Automatic framework detection**: Seamless switching between frameworks based on model type

#### XAI Methods
- **Gradient-based methods**: 
  - Vanilla gradients, Gradient × Input, Integrated Gradients
  - SmoothGrad, VarGrad for noise reduction
  - Guided Backpropagation and DeconvNet
- **Layer-wise Relevance Propagation (LRP)**:
  - Basic rules: LRP-Z, LRP-Epsilon, LRP-Gamma
  - Advanced rules: LRP-Alpha-Beta, LRP-Z+, LRP-ZBox
  - Composite rules with layer-specific configurations
- **Class Activation Mapping**:
  - Grad-CAM for convolutional networks
  - Specialized Grad-CAM for time series data
- **SIGN method**: Novel contribution for reducing bias in explanations
  - SIGN-enhanced variants of gradient and LRP methods
  - Customizable threshold parameter (μ) for flexible sign computation

#### Visualization and Analysis
- **Framework-agnostic visualization**: Consistent plotting utilities across both frameworks
- **Advanced visualization options**: Heatmaps, overlays, positive/negative contribution separation
- **Method comparison tools**: Side-by-side comparison of multiple explanation methods
- **Batch processing support**: Efficient handling of multiple inputs

#### Documentation and Tutorials
- **Comprehensive documentation**: Complete Sphinx-based documentation with API reference
- **Interactive tutorials**: Jupyter notebooks covering basic usage, advanced features, and cross-framework comparison
- **Installation guides**: Detailed setup instructions for both frameworks
- **Method comparison examples**: Practical examples showing framework interoperability

#### Development Tools
- **Comparison scripts**: Tools for validating consistency between TensorFlow and PyTorch implementations
- **Performance benchmarking**: Scripts for comparing method execution times and accuracy
- **Testing framework**: Comprehensive test suite ensuring reliability across frameworks

### Technical Details

#### Dependencies
- **Core**: NumPy ≥1.19.0, Matplotlib ≥3.7.0, SciPy ≥1.10.0, Pillow ≥8.0.0
- **TensorFlow**: TensorFlow ≥2.8.0, ≤2.12.1
- **PyTorch**: PyTorch ≥1.10.0, Zennit ≥0.5.1, scikit-image ≥0.19.0
- **Python**: Supports Python 3.9-3.10

#### Package Structure
- **signxai.tf_signxai**: TensorFlow-specific implementations
- **signxai.torch_signxai**: PyTorch-specific implementations  
- **signxai.common**: Shared utilities and visualization tools
- **signxai.api**: Unified API and framework detection

#### Performance Features
- **Memory-efficient processing**: Optimized for large models and datasets
- **GPU acceleration**: Support for CUDA-enabled explanations in PyTorch
- **Batch processing**: Efficient handling of multiple explanations
- **Parameter mapping**: Automatic translation between framework-specific parameters

### Known Issues
- Some advanced LRP composites may have slight differences between TensorFlow and PyTorch implementations due to underlying library differences
- Very large models may require additional memory optimization for certain methods

### Acknowledgments
- Built upon [iNNvestigate](https://github.com/albermax/innvestigate) for TensorFlow LRP implementations
- Built upon [Zennit](https://github.com/chr5tphr/zennit) for PyTorch LRP implementations
- SIGN method represents the novel contribution of the SignXAI project

[0.8.0]: https://github.com/IRISlaboratory/signxai2/blob/main/CHANGELOG.md
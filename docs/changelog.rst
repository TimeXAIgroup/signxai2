Changelog
=========

Version 2.0.0 (Current)
-----------------------

**Major Changes**

- Complete rewrite with dual-framework support (PyTorch and TensorFlow)
- Unified API for seamless framework switching
- Integration with zennit 0.5.1 for PyTorch LRP implementation
- Embedded iNNvestigate for TensorFlow support
- New SIGN method variants with configurable μ parameter

**New Features**

- Model conversion utilities (TensorFlow to PyTorch)
- Extended example notebooks for both frameworks
- Comprehensive documentation with Sphinx
- Support for time series (1D) and image (2D) data
- ECG pathology classification examples

**API Changes**

- New unified method naming convention
- Framework-agnostic visualization utilities
- Automatic framework detection

**Bug Fixes**

- Fixed numerical stability issues in LRP methods
- Improved gradient computation accuracy
- Better handling of edge cases in model conversion

Version 1.0.0
-------------

- Initial release with SIGN method implementation
- Basic TensorFlow support
- Core XAI methods: Gradient, LRP, Integrated Gradients
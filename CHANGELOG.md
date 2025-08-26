# Changelog

## [0.14.0] - 2025-08-26

### Added

### Changed
- **Decoupled Utils from requiring all frameworks**: Utils can now be used independently
- **Fixed Quickstart**: Fixed Batch dimension issues in PyTorch quickstart examples
- **Fixed Tutorials**: Fixed Image Loading to a valid image Link and added package installation, before processing.
- **Fixed Tutorials**: Updated depedencies to utils to be framework agnostic.

### Deprecated

## [0.13.9] - Previous Version
- Modified package to remove wrappers.py and work with direct method calls.
### Examples
```python
# Old style (deprecated):
explain(model, x, method="smoothgrad", noise_level=0.3, num_samples=50)

# New style:
explain(model, x, method_name="smoothgrad_noise_0_3_samples_50")
```
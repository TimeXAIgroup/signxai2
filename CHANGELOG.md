# Changelog

## [0.13.8] - 2025-08-26

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

## [0.13.7] - Previous Version
- Initial release with wrapper-based API

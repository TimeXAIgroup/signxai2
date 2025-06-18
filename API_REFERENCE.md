# 🔧 SignXAI PyTorch API Reference

## Overview

Complete API reference for the PyTorch implementation of explainable AI methods designed for TensorFlow iNNvestigate compatibility.

---

## Core Classes

### AnalyzerBase

Abstract base class for all analyzers.

```python
class AnalyzerBase(ABC):
    def __init__(self, model: nn.Module)
    
    @abstractmethod
    def analyze(self, input_tensor: torch.Tensor, 
                target_class: Optional[Union[int, torch.Tensor]] = None, 
                **kwargs) -> np.ndarray
```

**Parameters:**
- `model`: PyTorch model to analyze
- `input_tensor`: Input tensor for analysis
- `target_class`: Target class index (None for argmax)

**Returns:** Attribution as numpy array

---

## Gradient-Based Analyzers

### GradientAnalyzer

Basic vanilla gradients.

```python
analyzer = GradientAnalyzer(model)
attribution = analyzer.analyze(input_tensor, target_class=class_idx)
```

### IntegratedGradientsAnalyzer

Path integral method for attribution.

```python
analyzer = IntegratedGradientsAnalyzer(
    model=model,
    steps=50,                    # Number of integration steps
    baseline_type="zero"         # "zero", "black", "white", "gaussian"
)

attribution = analyzer.analyze(
    input_tensor,
    target_class=class_idx,
    baseline=custom_baseline,    # Optional custom baseline
    steps=100                    # Override instance steps
)
```

**Parameters:**
- `steps`: Integration steps (default: 50)
- `baseline_type`: Baseline generation method
- `baseline`: Custom baseline tensor (optional)

### SmoothGradAnalyzer

Gradient smoothing with noise.

```python
analyzer = SmoothGradAnalyzer(
    model=model,
    noise_level=0.2,            # Noise standard deviation factor
    num_samples=50              # Number of noisy samples
)

attribution = analyzer.analyze(
    input_tensor,
    target_class=class_idx,
    noise_level=0.15,           # Override noise level
    num_samples=100,            # Override sample count
    apply_sign=False,           # Apply sign function
    multiply_by_input=False     # Multiply by input
)
```

### GuidedBackpropAnalyzer

ReLU-modified gradients using Zennit's guided backprop composite.

```python
analyzer = GuidedBackpropAnalyzer(model)
attribution = analyzer.analyze(input_tensor, target_class=class_idx)
```

---

## LRP Analyzers

### AdvancedLRPAnalyzer

Main LRP implementation with custom hook support.

```python
analyzer = AdvancedLRPAnalyzer(
    model=model,
    variant="epsilon",          # LRP variant
    **kwargs                    # Variant-specific parameters
)

attribution = analyzer.analyze(input_tensor, target_class=class_idx)
```

#### Supported Variants

##### Epsilon Rule
```python
analyzer = AdvancedLRPAnalyzer(
    model, 
    variant="epsilon",
    epsilon=1e-6               # Stabilization parameter
)
```

##### Flat Rule (iNNvestigate Compatible)
```python
analyzer = AdvancedLRPAnalyzer(
    model,
    variant="flat",
    epsilon=1e-6               # For SafeDivide operation
)
```

##### WSquare Rule (iNNvestigate Compatible)
```python
analyzer = AdvancedLRPAnalyzer(
    model,
    variant="wsquare",
    epsilon=1e-6               # For SafeDivide operation
)
```

##### ZPlus Rule
```python
analyzer = AdvancedLRPAnalyzer(
    model,
    variant="zplus"
)
```

##### Alpha-Beta Rule
```python
analyzer = AdvancedLRPAnalyzer(
    model,
    variant="alpha2beta1",
    alpha=2,                   # Alpha parameter
    beta=1,                    # Beta parameter
    tf_compat_mode=True        # TensorFlow compatibility
)
```

---

## Custom Hooks

### InnvestigateFlatHook

Custom hook implementing iNNvestigate's FlatRule.

```python
from signxai.torch_signxai.methods.zennit_impl.innvestigate_compatible_hooks import (
    InnvestigateFlatHook
)

hook = InnvestigateFlatHook(stabilizer=1e-6)
```

### InnvestigateWSquareHook

Custom hook implementing iNNvestigate's WSquareRule.

```python
hook = InnvestigateWSquareHook(stabilizer=1e-6)
```

### InnvestigateEpsilonHook

Custom hook implementing iNNvestigate's EpsilonRule.

```python
hook = InnvestigateEpsilonHook(
    epsilon=1e-6,              # Stabilization parameter
    bias=True                  # Include bias in computation
)
```

---

## Composite Factories

### Custom Composite Creation

```python
from signxai.torch_signxai.methods.zennit_impl.innvestigate_compatible_hooks import (
    create_innvestigate_flat_composite,
    create_innvestigate_wsquare_composite,
    create_innvestigate_epsilon_composite
)

# Create composites with custom hooks
flat_composite = create_innvestigate_flat_composite()
wsquare_composite = create_innvestigate_wsquare_composite()
epsilon_composite = create_innvestigate_epsilon_composite(epsilon=1e-6)

# Use with Zennit Gradient
from zennit.attribution import Gradient
attributor = Gradient(model=model, composite=flat_composite)
```

---

## Method Registry

### Accessing Methods via Registry

```python
from signxai.torch_signxai.methods.zennit_impl import SUPPORTED_ZENNIT_METHODS

# Get available methods
available_methods = list(SUPPORTED_ZENNIT_METHODS.keys())

# Create analyzer via registry
analyzer_class = SUPPORTED_ZENNIT_METHODS["lrp_flat"]
analyzer = analyzer_class(model, variant="flat")
```

### Registered Method Names

#### Basic Methods
- `"gradient"`: Basic gradients
- `"integrated_gradients"`: Integrated gradients
- `"smoothgrad"`: SmoothGrad
- `"guided_backprop"`: Guided backpropagation

#### LRP Variants
- `"lrp.epsilon"`: Standard LRP epsilon
- `"lrp_flat"`: Flat rule
- `"lrp_w_square"`: WSquare rule
- `"flatlrp_epsilon_1"`: FlatLRP with epsilon=1
- `"flatlrp_epsilon_10"`: FlatLRP with epsilon=10
- `"flatlrp_epsilon_20"`: FlatLRP with epsilon=20
- `"flatlrp_epsilon_100"`: FlatLRP with epsilon=100

---

## Usage Examples

### Basic Usage

```python
import torch
import torch.nn as nn
from signxai.torch_signxai.methods.zennit_impl.lrp_variants import AdvancedLRPAnalyzer

# Load your model
model = your_model
model.eval()

# Prepare input
input_tensor = torch.randn(1, 3, 224, 224)

# Create analyzer
analyzer = AdvancedLRPAnalyzer(model, variant="flat", epsilon=1e-6)

# Generate attribution
attribution = analyzer.analyze(input_tensor, target_class=0)

print(f"Attribution shape: {attribution.shape}")
print(f"Attribution range: [{attribution.min():.6f}, {attribution.max():.6f}]")
```

### Batch Processing

```python
# Process multiple inputs
batch_inputs = torch.randn(8, 3, 224, 224)
batch_attributions = []

for i in range(batch_inputs.size(0)):
    single_input = batch_inputs[i:i+1]
    attribution = analyzer.analyze(single_input, target_class=0)
    batch_attributions.append(attribution)

# Stack results
all_attributions = np.stack(batch_attributions, axis=0)
```

### Custom Parameters

```python
# Integrated Gradients with custom parameters
ig_analyzer = IntegratedGradientsAnalyzer(
    model=model,
    steps=100,
    baseline_type="gaussian"
)

attribution = ig_analyzer.analyze(
    input_tensor,
    target_class=target_class,
    baseline=custom_baseline
)

# SmoothGrad with high quality settings
sg_analyzer = SmoothGradAnalyzer(
    model=model,
    noise_level=0.1,
    num_samples=100
)

attribution = sg_analyzer.analyze(input_tensor, target_class=target_class)
```

---

## Error Handling

### Common Exceptions

```python
try:
    analyzer = AdvancedLRPAnalyzer(model, variant="invalid_variant")
except ValueError as e:
    print(f"Invalid variant: {e}")

try:
    attribution = analyzer.analyze(input_tensor, target_class=999999)
except IndexError as e:
    print(f"Invalid target class: {e}")
```

### Validation

```python
# Check if method is supported
if "lrp_flat" in SUPPORTED_ZENNIT_METHODS:
    analyzer = AdvancedLRPAnalyzer(model, variant="flat")
else:
    print("Method not supported")

# Validate input tensor
assert input_tensor.dim() == 4, "Expected 4D input tensor (batch, channels, height, width)"
assert input_tensor.size(0) == 1, "Expected batch size of 1"
```

---

## Performance Considerations

### Memory Usage

```python
# For large models, consider using torch.no_grad() when possible
with torch.no_grad():
    # Model inference for target class determination
    output = model(input_tensor)
    target_class = output.argmax(dim=1).item()

# Attribution computation requires gradients
attribution = analyzer.analyze(input_tensor, target_class=target_class)
```

### Deterministic Behavior

```python
# Set seeds for reproducible results (except SmoothGrad)
torch.manual_seed(42)
np.random.seed(42)

# Note: SmoothGrad is non-deterministic by design due to noise
```

---

## Troubleshooting

### Common Issues

1. **Numerical Overflow in Epsilon Method**
   ```python
   # Use smaller epsilon values
   analyzer = AdvancedLRPAnalyzer(model, variant="epsilon", epsilon=1e-8)
   ```

2. **Weak Attribution Signal**
   ```python
   # Try different variants or check model architecture
   for variant in ["flat", "wsquare", "zplus"]:
       analyzer = AdvancedLRPAnalyzer(model, variant=variant)
       attribution = analyzer.analyze(input_tensor, target_class=0)
       print(f"{variant} std: {attribution.std():.8f}")
   ```

3. **Memory Issues with Large Models**
   ```python
   # Process smaller batches or use gradient checkpointing
   # Ensure model is on appropriate device
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)
   input_tensor = input_tensor.to(device)
   ```

---

## Version Compatibility

- **PyTorch**: ≥1.9.0
- **Zennit**: 0.5.1
- **NumPy**: ≥1.19.0
- **PIL**: ≥8.0.0

---

## See Also

- [Comprehensive Implementation Documentation](COMPREHENSIVE_IMPLEMENTATION_DOCUMENTATION.md)
- [Final Success Summary](FINAL_SUCCESS_SUMMARY.md)
- [Correlation Fix Summary](CORRELATION_FIX_SUMMARY.md)
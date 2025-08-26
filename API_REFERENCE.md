# SignXAI2 API Reference

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
from signxai.tf_signxai.tf_utils import remove_softmax
model_no_softmax = remove_softmax(model)
```

### PyTorch
```python
from signxai.torch_signxai.torch_utils import remove_softmax
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

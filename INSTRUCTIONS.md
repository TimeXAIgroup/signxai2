0 Instruction for Systematically Fixing TensorFlow-PyTorch XAI Method Discrepancies

  Overview

  You are tasked with systematically identifying and fixing discrepancies between TensorFlow (iNNvestigate) and PyTorch (Zennit) implementations of explainable AI methods. The goal is to achieve Mean Absolute Error (MAE) < 1e-04 between framework implementations by modifying only the PyTorch side with
   custom hooks while keeping TensorFlow implementations unchanged.

  Project Structure

  - Working Directory: /home/honormagicbook14/Projects/PyCharm/signxai2/examples/comparison
  - Main Script: run_image_method_comparison.py
  - PyTorch Wrappers: /home/honormagicbook14/Projects/PyCharm/signxai2/signxai/torch_signxai/methods/wrappers.py
  - TF-Exact Hooks: /home/honormagicbook14/Projects/PyCharm/signxai2/signxai/torch_signxai/methods/zennit_impl/tf_exact_*_hook.py
  - Results Directory: /home/honormagicbook14/Projects/PyCharm/signxai2/examples/comparison/results/

  Dependencies & Constraints

  ### Framework Versions
  - **TensorFlow**: 2.8.0 - 2.12.1 (with local iNNvestigate implementation) - **DO NOT MODIFY**
  - **PyTorch**: >= 1.10.0 with **Zennit 0.5.1** (older version) - **MODIFY ONLY PYTORCH SIDE**
  - **Python**: >= 3.8 (supports 3.8-3.11)

  ### Implementation Constraints
  - **TensorFlow Side**: Uses local iNNvestigate library (not pip-installed version)
  - **PyTorch Side**: Uses Zennit 0.5.1 (older version with limited features)
  - **Modification Rule**: Only modify PyTorch implementations to match TensorFlow behavior
  - **Approach**: Create custom TF-exact hooks in `/signxai/torch_signxai/methods/zennit_impl/`
  - **Goal**: Achieve perfect numerical alignment (MAE < 1e-04) between frameworks and visually identical heatmaps. heatmaps should be in red and white as e.g. examples/comparison/results/gradient_x_input_comparison_tf_pt_vgg16.jpg

  ## Architecture Overview

  ### TensorFlow Implementation
  - **Location**: `/signxai/tf_signxai/methods/`
  - **Core**: Local iNNvestigate library (modified version)
  - **Features**: Complete XAI method implementations
  - **Status**: Reference implementation - **DO NOT MODIFY**

  ### PyTorch Implementation
  - **Location**: `/signxai/torch_signxai/methods/`
  - **Core**: Zennit 0.5.1 + custom TF-exact hooks
  - **Wrappers**: `/signxai/torch_signxai/methods/wrappers.py`
  - **Custom Hooks**: `/signxai/torch_signxai/methods/zennit_impl/tf_exact_*_hook.py`
  - **Strategy**: Use custom hooks to replicate TensorFlow behavior exactly

  ### Custom Hooks Architecture
  The PyTorch implementation uses a sophisticated custom hooks system:
  
  1. **TF-Exact Hooks**: Custom Zennit hooks that replicate TensorFlow iNNvestigate behavior
  2. **Scaling Corrections**: Empirically determined scaling factors to match TF magnitudes
  3. **Parameter Routing**: Proper handling of method-specific parameters
  4. **Rule Application**: Correct application of LRP rules to match TF layer processing

  Available TF-Exact Hooks:
  - `tf_exact_epsilon_hook.py` - For LRP epsilon methods
  - `tf_exact_lrpsign_*.py` - For LRP sign-based methods
  - `tf_exact_lrpz_*.py` - For LRP-Z methods
  - `tf_exact_sequential_composite_*.py` - For composite methods
  - `tf_exact_stdx_*.py` - For standard deviation methods

  ## Key Implementation Differences

  ### TensorFlow (iNNvestigate) vs PyTorch (Zennit) Behavioral Differences
  
  1. **Numerical Precision**: TensorFlow uses different numerical precision handling
  2. **Layer Processing**: Different approaches to layer-wise relevance propagation
  3. **Parameter Handling**: Different default parameter values and scaling
  4. **Rule Application**: Variations in how LRP rules are applied to layers
  5. **Input Processing**: Different preprocessing and normalization approaches
  6. **Memory Layout**: Different tensor layout conventions (NHWC vs NCHW)

  ### Common Discrepancy Categories
  
  **1. Scaling Issues** (Most Common)
  - **Cause**: Different numerical implementations in TensorFlow vs PyTorch
  - **Symptoms**: High correlation (>0.99) but large MAE due to magnitude differences
  - **Solution**: Apply empirically determined scaling factors
  
  **2. Parameter Routing Issues**
  - **Cause**: Different parameter handling between frameworks
  - **Symptoms**: Parameter conflicts or incorrect rule application
  - **Solution**: Proper parameter filtering and routing in wrapper functions
  
  **3. Layer Rule Mapping Issues**
  - **Cause**: Different approaches to applying rules to specific layers
  - **Symptoms**: Incorrect attribution patterns, especially for input layers
  - **Solution**: Custom rule mapping and layer-specific handling
  
  **4. Composite Method Issues**
  - **Cause**: Different approaches to combining multiple rules
  - **Symptoms**: Incorrect relevance flow in multi-rule scenarios
  - **Solution**: Custom composite implementations that match TF behavior

  Methodology

  Step 1: Identify Methods to Fix

  Run the comparison script to get a list of methods with high MAE:
  python run_image_method_comparison.py

  Check the results in results/values_image_method_comparison.txt for methods with MAE > 1e-04.

  Step 2: Systematic Method Fixing Process

  For each problematic method (e.g., lrp_epsilon_0_25_std_x):

  2.1 Run Targeted Comparison

  python run_image_method_comparison.py lrp_epsilon_0_25_std_x

  2.2 Create Diagnostic Script

  Create a diagnostic script to measure exact scaling differences:

  #!/usr/bin/env python3
  """Diagnose scaling issues for [METHOD_NAME]."""

  import numpy as np
  import torch
  import os
  import sys

  # Add paths
  PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
  sys.path.append(PROJECT_ROOT)
  PT_MODEL_DEFINITION_DIR = os.path.join(PROJECT_ROOT, 'examples/data/models/pytorch/VGG16')
  sys.path.append(PT_MODEL_DEFINITION_DIR)

  # Imports
  from VGG16 import VGG16_PyTorch
  import tensorflow as tf
  from tensorflow.keras.models import load_model as tf_load_model
  from signxai.torch_signxai import calculate_relevancemap as pt_calculate_relevancemap
  from signxai.tf_signxai.methods.wrappers import calculate_relevancemap as tf_calculate_relevancemap
  from signxai.torch_signxai.utils import remove_softmax as torch_remove_softmax
  from signxai.utils.utils import remove_softmax as tf_remove_softmax
  from PIL import Image

  # Load models and image
  img_path = os.path.join(PROJECT_ROOT, 'examples/data/images/example.jpg')
  img = Image.open(img_path).resize((224, 224))
  img_array = np.array(img) / 255.0

  TF_MODEL_PATH = os.path.join(PROJECT_ROOT, 'examples/data/models/tensorflow/VGG16/model.h5')
  tf_model = tf_load_model(TF_MODEL_PATH)
  tf_model_no_softmax = tf_remove_softmax(tf_model)
  tf_input = img_array[np.newaxis, ...].astype(np.float32)

  PT_MODEL_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'examples/data/models/pytorch/VGG16/vgg16_ported_weights.pt')
  pt_model = VGG16_PyTorch(num_classes=1000)
  pt_model.load_state_dict(torch.load(PT_MODEL_WEIGHTS_PATH, map_location='cpu'))
  pt_model_no_softmax = torch_remove_softmax(pt_model)
  pt_model_no_softmax.eval()
  pt_input = torch.from_numpy(img_array.transpose(2, 0, 1)[np.newaxis, ...]).float()

  # Get target class
  with torch.no_grad():
      pt_logits = pt_model(pt_input)
      target_class = pt_logits.argmax(dim=1).item()

  # Calculate explanations with method-specific parameters
  METHOD_NAME = "[METHOD_NAME]"  # e.g., "lrp_epsilon_0_25_std_x"
  METHOD_PARAMS = {}  # e.g., {"epsilon": 0.25, "stdfactor": 0.25} for std_x methods

  tf_explanation = tf_calculate_relevancemap(METHOD_NAME, tf_input, tf_model_no_softmax,
                                           neuron_selection_mode=int(target_class), **METHOD_PARAMS)
  tf_aggregated = tf_explanation.sum(axis=-1) if tf_explanation.ndim == 3 else tf_explanation

  pt_explanation = pt_calculate_relevancemap(pt_model_no_softmax, pt_input, METHOD_NAME,
                                           target_class=int(target_class), **METHOD_PARAMS)
  if isinstance(pt_explanation, torch.Tensor):
      pt_aggregated = pt_explanation.sum(axis=1).squeeze().detach().numpy()
  else:
      pt_aggregated = pt_explanation.sum(axis=1) if pt_explanation.ndim == 3 else pt_explanation

  # Calculate exact scaling ratio
  tf_magnitude = np.abs(tf_aggregated).mean()
  pt_magnitude = np.abs(pt_aggregated).mean()
  scale_ratio = tf_magnitude / pt_magnitude if pt_magnitude > 0 else 0

  print(f"TF magnitude: {tf_magnitude:.6f}")
  print(f"PT magnitude: {pt_magnitude:.6f}")
  print(f"Scale ratio (TF/PT): {scale_ratio:.1f}x")

  mae_before = np.abs(tf_aggregated - pt_aggregated).mean()
  print(f"MAE before scaling: {mae_before:.6f}")

  # Test scaling correction
  pt_scaled = pt_aggregated * scale_ratio
  mae_after = np.abs(tf_aggregated - pt_scaled).mean()
  print(f"MAE after {scale_ratio:.1f}x scaling: {mae_after:.6f}")

  2.3 Identify Method Category and Required Fix

  Common Method Categories:

  1. LRP Epsilon Methods (lrp_epsilon_X):
    - Typically need TF-exact epsilon hooks with scaling corrections
    - Scale factors: 4.4x (ε=0.5), 19x (ε=0.2), 21x (ε=0.001), 30x (ε=0.01), 20.8x (ε=0.1)
  2. LRP Standard Deviation Methods (*_std_x):
    - Use direct Zennit approach with epsilon = factor * std(x)
    - May need scaling corrections
  3. LRP Input Layer Rule Methods (flat*, w2*, zb*):
    - Check for input layer rule routing issues
    - Need proper rule application to first layer
  4. Sign-based Methods (*_x_sign*):
    - May have thresholding function differences
    - Check calculate_sign_mu implementation

  2.4 Implement the Fix

  For LRP Epsilon Methods:
  def lrp_epsilon_X(model_no_softmax, x, **kwargs):
      """Calculate LRP epsilon X relevance map with TF-exact implementation."""
      # Force use of TF-exact epsilon implementation with scaling correction
      from signxai.torch_signxai.methods.zennit_impl.tf_exact_epsilon_hook import create_tf_exact_epsilon_composite
      from zennit.attribution import Gradient

      # Convert input to tensor if needed
      if not isinstance(x, torch.Tensor):
          x = torch.tensor(x, dtype=torch.float32)

      # Add batch dimension if needed
      needs_batch_dim = x.ndim == 3
      if needs_batch_dim:
          x = x.unsqueeze(0)

      # Create new kwargs without epsilon to avoid parameter conflict
      filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'epsilon'}

      # Use TF-exact epsilon implementation
      with create_tf_exact_epsilon_composite(model_no_softmax, epsilon=X):
          model_no_softmax.zero_grad()

          # Prepare target
          target_class = filtered_kwargs.get('target_class', None)
          if target_class is None:
              with torch.no_grad():
                  output = model_no_softmax(x)
                  target_class = output.argmax(dim=1)

          if isinstance(target_class, int):
              target_class = torch.tensor([target_class])

          # Calculate gradient-based attribution using exact TF approach
          gradient = Gradient()
          lrp = gradient(x, model_no_softmax, target_class, start_layer=0)

          # Apply scaling correction to match TensorFlow magnitude
          SCALE_CORRECTION_FACTOR = Y.Y  # Determined from diagnostic
          lrp = lrp * SCALE_CORRECTION_FACTOR

          # Remove batch dimension if it was added
          if needs_batch_dim:
              lrp = lrp[0]

          # Convert to numpy
          result = lrp.detach().cpu().numpy()

      return result

  For Input Layer Rule Methods:
  def method_with_input_layer_rule(model_no_softmax, x, **kwargs):
      """Method with proper input layer rule routing."""
      input_layer_rule = kwargs.get("input_layer_rule", None)

      if input_layer_rule is not None:
          # Use direct Zennit approach for methods with specific input layer rules
          # ... implement rule-specific logic
          SCALE_CORRECTION_FACTOR = Z.Z  # Method-specific scaling
          result = result * SCALE_CORRECTION_FACTOR
      else:
          # Use standard implementation
          # ... standard logic

      return result

  2.5 Test the Fix

  python run_image_method_comparison.py [method_name]

  Check that MAE < 1e-04 and correlation > 0.99.

  2.6 Verify Visually

  Check the generated comparison image in results/ to ensure heatmaps are visually identical.

  Step 3: Systematic Processing

  Process methods in order of complexity:

  1. Start with simple epsilon methods: lrp_epsilon_1, lrp_epsilon_2, etc.
  2. Move to epsilon std_x methods: lrp_epsilon_1_std_x, etc.
  3. Handle input layer rule methods: lrp_flat, lrp_w_square, etc.
  4. Fix sign-based methods: *_x_sign*, *_sign_mu*
  5. Address composite methods: *_sequential_composite_*

  Step 4: Quality Assurance

  For each fixed method:
  - ✅ MAE < 1e-04
  - ✅ Correlation > 0.99
  - ✅ Visual heatmaps match
  - ✅ No parameter conflicts or errors

  Step 5: Batch Validation

  After fixing multiple methods, run full comparison:
  python run_image_method_comparison.py

  Check results/values_image_method_comparison.txt for overall improvement.

  Key Implementation Patterns

  1. Always modify PyTorch wrappers, never TensorFlow
  2. Use TF-exact hooks for epsilon methods with empirically-determined scaling
  3. Handle parameter conflicts by filtering kwargs
  4. Preserve batch dimensions and tensor/numpy conversions
  5. Apply scaling corrections based on diagnostic measurements
  6. Route input layer rules correctly for composite methods

  Expected Outcome

  All methods should achieve:
  - MAE < 1e-04 between TensorFlow and PyTorch implementations
  - Correlation > 0.99
  - Visually identical heatmaps
  - No runtime errors or parameter conflicts

  This systematic approach ensures robust, reproducible fixes that maintain compatibility with the existing codebase while achieving exact framework alignment.
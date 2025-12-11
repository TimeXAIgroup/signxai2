===============================
Framework Interoperability
===============================

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

SignXAI offers a unique advantage by supporting both PyTorch and TensorFlow frameworks with a consistent API. This guide explains how to:

1. Use the framework-agnostic API
2. Switch between frameworks
3. Compare results across frameworks
4. Convert models between frameworks

Framework Detection
-------------------

SignXAI automatically detects which framework is being used based on the model type:

.. code-block:: python

    import signxai
    from signxai.api import explain
    
    # Check which backends are available
    print(f"Available backends: {signxai._AVAILABLE_BACKENDS}")
    
    # To use with automatic framework detection using the new API
    result = explain(model, input_tensor, method_name="gradient")
    
    # SignXAI will automatically determine if model is PyTorch or TensorFlow
    # and use the appropriate implementation

Framework-Agnostic API
----------------------

The framework-agnostic API provides a consistent interface regardless of which framework you're using:

.. code-block:: python

    from signxai.api import explain
    
    # Works with both PyTorch and TensorFlow models
    explanation = explain(
        model,              # Either tf.keras.Model or torch.nn.Module
        input_tensor,       # Either numpy array, tf.Tensor, or torch.Tensor
        method_name="lrp_z"  # Same method names across frameworks
    )
    
    # Multiple inputs
    explanations = []
    for input_tensor in [input1, input2, input3]:
        explanations.append(explain(model, input_tensor, method_name="gradient_x_input"))

Method Consistency Across Frameworks
------------------------------------

SignXAI ensures that the same method produces comparable results across frameworks:

=============================== ================== ==================
Method                          PyTorch            TensorFlow
=============================== ================== ==================
``gradient``                    ✓                  ✓
``gradient_x_input``            ✓                  ✓
``gradient_x_sign``             ✓                  ✓
``guided_backprop``             ✓                  ✓
``integrated_gradients_steps_50`` ✓                ✓
``smoothgrad_noise_0_2_samples_50`` ✓             ✓
``grad_cam``                    ✓                  ✓
``lrp_z``                       ✓                  ✓
``lrp_epsilon_0_1``             ✓                  ✓
``lrp_alpha_1_beta_0``          ✓                  ✓
=============================== ================== ==================

Implementation Differences
--------------------------

While SignXAI strives for consistent results, there are some implementation differences to be aware of:

1. **Backend Libraries**
   
   - TensorFlow: Uses iNNvestigate for LRP
   - PyTorch: Uses Zennit for LRP
   
2. **API Parameter Naming**
   
   - TensorFlow: Uses ``neuron_selection`` for target class
   - PyTorch: Uses ``target_class`` for target class (though both are accepted)
   
3. **Custom LRP Rules**
   
   - Layer-specific rules have slightly different implementation details
   - The core methods provide consistent results, but custom configurations may differ

Framework-Specific Workflow
---------------------------

If you prefer to work directly with a specific framework's implementation:

TensorFlow-Specific Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from tensorflow.keras.applications.vgg16 import VGG16
    from signxai.api import explain
    from signxai.utils.utils import load_image, normalize_heatmap
    
    # Load TensorFlow model
    model = VGG16(weights='imagenet')
    
    # Remove softmax
    model.layers[-1].activation = None
    
    # Load and preprocess input
    img, x = load_image('example.jpg')
    
    # Calculate explanation using the new dynamic parsing API
    explanation = explain(model, x, method_name='lrp_z')
    
    # Visualize
    import matplotlib.pyplot as plt
    plt.imshow(normalize_heatmap(explanation), cmap='seismic', clim=(-1, 1))
    plt.show()

PyTorch-Specific Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torchvision.models as models
    from signxai.api import explain
    from signxai.torch_signxai.utils import remove_softmax
    
    # Load PyTorch model
    model = models.vgg16(pretrained=True)
    model.eval()
    
    # Remove softmax
    model_no_softmax = remove_softmax(model)
    
    # Preprocess input
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = Image.open('example.jpg')
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Calculate explanation using the new dynamic parsing API
    explanation = explain(model_no_softmax, input_tensor, method_name="lrp_epsilon_0_1")
    
    # Visualize
    from signxai.common.visualization import normalize_relevance_map
    import matplotlib.pyplot as plt
    
    plt.imshow(normalize_relevance_map(explanation[0].sum(axis=0)), cmap='seismic', clim=(-1, 1))
    plt.show()

Converting Models Between Frameworks
------------------------------------

If you need to compare the exact same model across frameworks, SignXAI provides utilities for model conversion.

ONNX-Based Conversion
~~~~~~~~~~~~~~~~~~~~~

ONNX (Open Neural Network Exchange) provides a standard format for model conversion:

.. code-block:: python

    # TensorFlow to PyTorch via ONNX
    from signxai.converters.onnx_to_torch import convert_tf_to_torch_via_onnx
    
    # Convert TensorFlow model to PyTorch
    pytorch_model = convert_tf_to_torch_via_onnx(tensorflow_model, input_shape=(1, 224, 224, 3))
    
    # Now you can use the same model with both frameworks using the unified API
    from signxai.api import explain
    tf_explanation = explain(tensorflow_model, x, method_name='lrp_z')
    torch_explanation = explain(pytorch_model, torch_x, method_name="lrp_z")

Direct Conversion
~~~~~~~~~~~~~~~~~

For some simpler models, direct conversion without ONNX is possible:

.. code-block:: python

    from signxai.converters.direct_tf_to_torch import convert_tf_to_torch_direct
    
    # Direct conversion for compatible models
    pytorch_model = convert_tf_to_torch_direct(tensorflow_model)

Comparing Results Across Frameworks
-----------------------------------

To ensure consistency, you may want to compare explanation results from both frameworks:

.. code-block:: python

    import numpy as np
    from signxai.common.visualization import visualize_comparison
    
    # Get explanations from both frameworks using the unified API
    from signxai.api import explain
    tf_explanation = explain(tensorflow_model, x, method_name='lrp_z')
    torch_explanation = explain(pytorch_model, torch_x, method_name="lrp_z")
    
    # Convert to same format (numpy arrays)
    if torch.is_tensor(torch_explanation):
        torch_explanation = torch_explanation.detach().cpu().numpy()
    
    # Compute similarity metrics
    similarity = np.corrcoef(tf_explanation.flatten(), torch_explanation.flatten())[0, 1]
    print(f"Correlation between TensorFlow and PyTorch explanations: {similarity:.4f}")
    
    # Visualize differences
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(tf_explanation, cmap='seismic', clim=(-1, 1))
    axs[0].set_title("TensorFlow")
    axs[1].imshow(torch_explanation, cmap='seismic', clim=(-1, 1))
    axs[1].set_title("PyTorch")
    axs[2].imshow(np.abs(tf_explanation - torch_explanation), cmap='hot')
    axs[2].set_title("Absolute Difference")
    plt.tight_layout()
    plt.show()

Framework-Agnostic Visualization
--------------------------------

SignXAI provides framework-agnostic visualization utilities:

.. code-block:: python

    from signxai.common.visualization import (
        normalize_relevance_map,
        relevance_to_heatmap,
        overlay_heatmap,
        visualize_comparison
    )
    
    # Works with explanations from either framework
    normalized = normalize_relevance_map(explanation)
    heatmap = relevance_to_heatmap(normalized)
    overlaid = overlay_heatmap(original_image, heatmap)
    
    # Compare multiple methods
    fig = visualize_comparison(
        original_image,
        [method1_result, method2_result, method3_result],
        ["Method 1", "Method 2", "Method 3"]
    )
    plt.show()

Framework Differences in LRP Implementation
-------------------------------------------

Due to using different backend libraries (iNNvestigate vs. Zennit), there are some subtle differences in LRP implementations:

=========================== ================================= ===============================
Feature                     TensorFlow (iNNvestigate)         PyTorch (Zennit)
=========================== ================================= ===============================
Input layer rules           Z, SIGN, Bounded, WSquare, Flat   Handled through composites
Layer-specific rules        Via manual configuration          Via composite layer maps
Composite handling          Sequential composites A & B       Flexible layer mapping
Computation approach        Graph-based                       Hook-based
=========================== ================================= ===============================

Despite these implementation differences, SignXAI ensures that the core algorithms produce comparable results.

Tips for Seamless Framework Integration
---------------------------------------

1. **Consistent Input Format**
   
   - Use numpy arrays for inputs when possible
   - Ensure input dimensions match framework expectations
   
2. **Model Preparation**
   
   - Always remove the softmax layer
   - Ensure model is in evaluation mode
   
3. **Parameter Mapping**
   
   - Use common parameter names that work in both frameworks
   - Be explicit about target class specification
   
4. **Result Handling**
   
   - Convert results to numpy arrays for further processing
   - Use framework-agnostic visualization functions

Case Study: Analyzing the Same Model Across Frameworks
------------------------------------------------------

This example demonstrates analyzing the same model architecture (VGG16) in both frameworks:

.. code-block:: python

    import numpy as np
    import tensorflow as tf
    import torch
    import torchvision.models as torch_models
    from tensorflow.keras.applications.vgg16 import VGG16 as tf_VGG16
    import matplotlib.pyplot as plt
    
    from signxai.api import explain
    from signxai.utils.utils import load_image
    from signxai.common.visualization import normalize_relevance_map
    
    # Load example image
    img, x_np = load_image('example.jpg')
    
    # Prepare TensorFlow model and input
    tf_model = tf_VGG16(weights='imagenet')
    tf_model.layers[-1].activation = None  # Remove softmax
    x_tf = np.expand_dims(x_np, axis=0)  # Add batch dimension
    
    # Prepare PyTorch model and input
    torch_model = torch_models.vgg16(pretrained=True)
    torch_model.eval()
    # Convert numpy to torch format (C, H, W)
    x_torch = torch.from_numpy(x_np.transpose(2, 0, 1)).float().unsqueeze(0)
    
    # Calculate explanations using the unified API
    tf_explanation = explain(tf_model, x_tf, method_name='lrp_z')
    torch_explanation = explain(torch_model, x_torch, method_name="lrp_z")
    
    # Convert to numpy arrays
    if isinstance(torch_explanation, torch.Tensor):
        torch_explanation = torch_explanation.detach().cpu().numpy()
    
    # Visualize and compare
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[1].imshow(normalize_relevance_map(tf_explanation), cmap='seismic', clim=(-1, 1))
    axs[1].set_title("TensorFlow Explanation")
    axs[2].imshow(normalize_relevance_map(torch_explanation[0].sum(axis=0)), cmap='seismic', clim=(-1, 1))
    axs[2].set_title("PyTorch Explanation")
    plt.tight_layout()
    plt.show()
    
    # Calculate similarity
    tf_flat = tf_explanation.flatten()
    torch_flat = torch_explanation[0].sum(axis=0).flatten()
    correlation = np.corrcoef(tf_flat, torch_flat)[0, 1]
    print(f"Correlation between TensorFlow and PyTorch explanations: {correlation:.4f}")

Conclusion
----------

SignXAI provides a powerful toolkit for explainable AI that works seamlessly across both TensorFlow and PyTorch. Whether you're working exclusively with one framework or need to compare results across both, SignXAI offers a consistent experience with comparable results.
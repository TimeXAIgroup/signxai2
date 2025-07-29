===========================
PyTorch Implementation
===========================

.. py:module:: signxai.torch_signxai
   :no-index:

This guide provides a detailed explanation of SignXAI's PyTorch implementation, with a focus on how the package integrates with Zennit for Layer-wise Relevance Propagation (LRP) methods.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The PyTorch implementation in SignXAI provides powerful explainability methods for PyTorch models. It uses the Zennit library as the backend for Layer-wise Relevance Propagation (LRP) methods, providing state-of-the-art explanation capabilities with a clean API.

Key Components
--------------

1. **Dual API styles** - Both PyTorch-native and TensorFlow-compatible interfaces
2. **Gradient-based methods** - Vanilla gradient, Integrated gradients, SmoothGrad
3. **Guided Backpropagation** - Enhanced gradient visualization
4. **Grad-CAM** - Class activation mapping for CNNs
5. **LRP with Zennit** - Layer-wise Relevance Propagation variants
6. **SIGN methods** - Novel methods that use sign information

Zennit Integration
------------------

The most powerful aspect of the PyTorch implementation is its integration with Zennit for LRP methods. This section explains how SignXAI leverages Zennit's capabilities.

What is Zennit?
~~~~~~~~~~~~~~~

`Zennit <https://github.com/chr5tphr/zennit>`_ is a PyTorch library for interpreting neural networks through LRP and other relevance propagation methods, developed at TU Berlin by Christopher J. Anders and colleagues. It offers:

1. Efficient implementation of various LRP rules
2. Flexible composite rule system for layer-specific rules
3. Native PyTorch integration with hooks and autograd

.. admonition:: Citation

   If you use Zennit in your research through SignXAI, please consider citing the original work:

   .. code-block:: bibtex

      @article{anders2021software,
        author  = {Anders, Christopher J. and
                   Neumann, David and
                   Samek, Wojciech and
                   Müller, Klaus-Robert and
                   Lapuschkin, Sebastian},
        title   = {Software for Dataset-wide XAI: From Local Explanations to Global Insights with {Zennit}, {CoRelAy}, and {ViRelAy}},
        journal = {CoRR},
        volume  = {abs/2106.13200},
        year    = {2021}
      }

SignXAI integrates Zennit through a custom implementation in the ``signxai.torch_signxai.methods.zennit_impl`` module, allowing for:

1. Seamless integration with dependency management
2. Consistent API with the TensorFlow implementation
3. Implementation of SignXAI-specific features

How SignXAI Uses Zennit
~~~~~~~~~~~~~~~~~~~~~~~

The integration happens primarily through the various analyzer classes in ``signxai.torch_signxai.methods.zennit_impl``:

**Available Analyzer Classes:**

- ``GradientAnalyzer`` - Vanilla gradient calculation
- ``GradientXInputAnalyzer`` - Gradient × input method
- ``GradientXSignAnalyzer`` - Gradient × sign method with threshold
- ``IntegratedGradientsAnalyzer`` - Integrated gradients
- ``SmoothGradAnalyzer`` - SmoothGrad with noise averaging
- ``VarGradAnalyzer`` - Variance of gradients across noisy samples
- ``GuidedBackpropAnalyzer`` - Guided backpropagation
- ``GradCAMAnalyzer`` - Grad-CAM visualization
- ``LRPAnalyzer`` - Layer-wise Relevance Propagation
- ``AdvancedLRPAnalyzer`` - Advanced LRP with specialized rules
- ``LRPSequential`` - Sequential LRP with layer-specific rules
- ``DeepTaylorAnalyzer`` - Deep Taylor decomposition using LRP epsilon

.. code-block:: python

    class LRPAnalyzer:
        """Layer-wise Relevance Propagation (LRP) analyzer.
        
        Uses zennit's implementation of LRP with different rule variants.
        """
        
        def __init__(self, model, rule="epsilon", epsilon=1e-6):
            """Initialize LRP analyzer.
            
            Args:
                model: PyTorch model
                rule: LRP rule ('epsilon', 'zplus', 'alphabeta')
                epsilon: Stabilizing factor for epsilon rule
            """
            self.model = model
            self.rule = rule
            self.epsilon = epsilon
            
            # Map rule name to zennit composite
            if rule == "epsilon":
                self.composite = EpsilonGammaBox(epsilon=epsilon)
            elif rule == "zplus":
                self.composite = ZPlus()
            elif rule == "alphabeta":
                self.composite = AlphaBeta(alpha=1, beta=0)
            else:
                raise ValueError(f"Unknown LRP rule: {rule}")
        
        def analyze(self, input_tensor, target_class=None):
            """Generate LRP attribution.
            
            Args:
                input_tensor: Input tensor
                target_class: Target class index (None for argmax)
                
            Returns:
                LRP attribution
            """
            # Set up attributor
            attributor = Attributor(self.model, self.composite)
            
            # Ensure input is a tensor and detach previous gradients
            if isinstance(input_tensor, torch.Tensor):
                input_tensor = input_tensor.detach().requires_grad_(True)
            else:
                input_tensor = torch.tensor(input_tensor, requires_grad=True)
            
            # Forward pass
            with attributor:
                output = self.model(input_tensor)
                
                # Get target class
                if target_class is None:
                    target_class = output.argmax(dim=1)
                
                # Create one-hot tensor
                if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                    one_hot = torch.zeros_like(output)
                    one_hot[0, target_class] = 1.0
                else:
                    one_hot = torch.zeros_like(output)
                    for i, cls in enumerate(target_class):
                        one_hot[i, cls] = 1.0
                
                # Get attribution
                attribution = attributor.attribute(input_tensor, output, one_hot)
            
            # Return as numpy array
            return attribution.detach().cpu().numpy()

This function combines Zennit's powerful LRP implementation with SignXAI's consistent interface.

LRP Methods in Detail
---------------------

SignXAI provides several LRP variants through Zennit:

LRP-Epsilon
~~~~~~~~~~~

Adds a small epsilon value to stabilize the division operation:

.. code-block:: python

    # Using PyTorch-native API
    explanation = calculate_relevancemap(model, input_tensor, method="lrp_epsilon", epsilon=0.1)
    
    # Or via analyzer directly
    analyzer = LRPAnalyzer(model, rule="epsilon", epsilon=0.1)
    explanation = analyzer.analyze(input_tensor, target_class=class_idx)

LRP-AlphaBeta
~~~~~~~~~~~~~

Separates positive and negative contributions with different weights:

.. code-block:: python

    # Using PyTorch-native API
    explanation = calculate_relevancemap(model, input_tensor, method="lrp_alphabeta", alpha=1, beta=0)
    
    # Or via analyzer directly
    analyzer = LRPAnalyzer(model, rule="alphabeta")  # Default alpha=1, beta=0
    explanation = analyzer.analyze(input_tensor, target_class=class_idx)

Advanced LRP Rules
~~~~~~~~~~~~~~~~~~

For more complex LRP configurations, the ``AdvancedLRPAnalyzer`` can be used:

.. code-block:: python

    # Using PyTorch-native API
    explanation = calculate_relevancemap(
        model, 
        input_tensor, 
        method="lrp_advanced",
        rule_type="alpha1beta0"
    )
    
    # Or for more control
    analyzer = AdvancedLRPAnalyzer(
        model, 
        rule_type="zbox", 
        low=-123.68, 
        high=151.061
    )
    explanation = analyzer.analyze(input_tensor, target_class=class_idx)

LRP Composite Rules
~~~~~~~~~~~~~~~~~~~

Applies different LRP rules to different layers of the network:

.. code-block:: python

    # Using PyTorch-native API
    explanation = calculate_relevancemap(
        model, 
        input_tensor, 
        method="lrp_sequential",
        first_layer_rule="zbox",
        middle_layer_rule="alphabeta",
        last_layer_rule="epsilon"
    )
    
    # Or via analyzer directly
    analyzer = LRPSequential(
        model,
        first_layer_rule="zbox",
        middle_layer_rule="alphabeta",
        last_layer_rule="epsilon"
    )
    explanation = analyzer.analyze(input_tensor, target_class=class_idx)

Implementation of Other Methods
-------------------------------

In addition to LRP methods, SignXAI provides Zennit-based implementations of other explainability techniques:

Vanilla Gradient
~~~~~~~~~~~~~~~~

.. code-block:: python

    class GradientAnalyzer:
        """Vanilla gradient analyzer.
        
        Implements vanilla gradient calculation aligned with TensorFlow's implementation.
        """
        
        def __init__(self, model):
            """Initialize gradient analyzer.
            
            Args:
                model: PyTorch model
            """
            self.model = model
        
        def analyze(self, input_tensor, target_class=None):
            """Generate vanilla gradient attribution aligned with TensorFlow.
            
            Args:
                input_tensor: Input tensor
                target_class: Target class index (None for argmax)
                
            Returns:
                Gradient attribution
            """
            # Ensure input is a tensor with gradients
            if isinstance(input_tensor, torch.Tensor):
                input_tensor = input_tensor.detach().requires_grad_(True)
            else:
                input_tensor = torch.tensor(input_tensor, requires_grad=True)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(input_tensor)
            
            # Get target class
            if target_class is None:
                target_class = output.argmax(dim=1)
            
            # Create one-hot tensor
            if isinstance(target_class, int) or (hasattr(target_class, 'ndim') and target_class.ndim == 0):
                one_hot = torch.zeros_like(output)
                one_hot[0, target_class] = 1.0
            else:
                one_hot = torch.zeros_like(output)
                for i, cls in enumerate(target_class):
                    one_hot[i, cls] = 1.0
            
            # Backward pass
            output.backward(gradient=one_hot)
            
            # Get gradients
            attribution = input_tensor.grad.clone()
            
            # Return as numpy array
            return attribution.detach().cpu().numpy()

Integrated Gradients
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class IntegratedGradientsAnalyzer:
        """Integrated gradients analyzer.
        
        Implements the integrated gradients method by integrating gradients along a straight
        path from a baseline (typically zeros) to the input.
        """
        
        def __init__(self, model, steps=50, baseline=None):
            """Initialize integrated gradients analyzer.
            
            Args:
                model: PyTorch model
                steps: Number of steps for integration
                baseline: Baseline input (None for zeros)
            """
            self.model = model
            self.steps = steps
            self.baseline = baseline
        
        def analyze(self, input_tensor, target_class=None):
            """Generate integrated gradients attribution.
            
            Args:
                input_tensor: Input tensor
                target_class: Target class index (None for argmax)
                
            Returns:
                Integrated gradients attribution
            """
            # Implementation details...
            # ...
            return attribution

SmoothGrad
~~~~~~~~~~

.. code-block:: python

    class SmoothGradAnalyzer:
        """SmoothGrad analyzer.
        
        Implements SmoothGrad by adding Gaussian noise to the input multiple times and 
        averaging the resulting gradients.
        """
        
        def __init__(self, model, noise_level=0.2, num_samples=50):
            """Initialize SmoothGrad analyzer.
            
            Args:
                model: PyTorch model
                noise_level: Level of Gaussian noise to add
                num_samples: Number of noisy samples to average
            """
            self.model = model
            self.noise_level = noise_level
            self.num_samples = num_samples
        
        def analyze(self, input_tensor, target_class=None):
            """Generate SmoothGrad attribution.
            
            Args:
                input_tensor: Input tensor
                target_class: Target class index (None for argmax)
                
            Returns:
                SmoothGrad attribution
            """
            # Implementation details...
            # ...
            return smoothgrad

Removing Softmax for Explainability
-----------------------------------

Proper explainability often requires working with raw logits rather than softmax probabilities. SignXAI provides a wrapper to remove softmax from PyTorch models:

.. code-block:: python

    def remove_softmax(model: nn.Module) -> nn.Module:
        """Remove softmax layer from a PyTorch model.
        
        This function creates a copy of the model and removes the softmax activation,
        which is a common preprocessing step for explainability methods.
        
        Args:
            model: PyTorch model with softmax
            
        Returns:
            Model with softmax removed (copy)
        """
        # Create a copy of the model
        model_no_softmax = type(model)()
        model_no_softmax.load_state_dict(model.get_state_dict())
        model_no_softmax.eval()
        
        # Wrap the model with NoSoftmaxWrapper which simply returns logits
        return NoSoftmaxWrapper(model_no_softmax)

    class NoSoftmaxWrapper(nn.Module):
        """Wrapper class that removes softmax from a PyTorch model.
        
        This class wraps a PyTorch model and ensures the output is always logits,
        effectively removing any softmax activation in the forward pass.
        
        Args:
            model: PyTorch model with softmax
        """
        
        def __init__(self, model: nn.Module):
            """Initialize NoSoftmaxWrapper.
            
            Args:
                model: PyTorch model with softmax
            """
            super().__init__()
            self.model = model
            self.model.eval()  # Set to evaluation mode
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass that returns logits directly (no softmax).
            
            Args:
                x: Input tensor
                
            Returns:
                Model output before softmax
            """
            # Forward pass through the model
            output = self.model(x)
            
            # Return unmodified output (logits)
            return output

Dual API Styles
---------------

SignXAI provides two API styles for PyTorch users:

1. **PyTorch-Native API** - More intuitive for PyTorch users

.. code-block:: python

    from signxai.torch_signxai import calculate_relevancemap
    
    # PyTorch-style API: model first, then input
    explanation = calculate_relevancemap(model, input_tensor, method="gradients")

2. **TensorFlow-Compatible API** - Consistent with the TensorFlow implementation

.. code-block:: python

    from signxai.torch_signxai.methods.wrappers import calculate_relevancemap as tf_calculate_relevancemap
    
    # TensorFlow-style API: method name first, then input, then model
    explanation = tf_calculate_relevancemap("gradient", input_tensor, model)

This dual API allows for easier migration between frameworks and preference-based usage.

Usage Example
-------------

The following example demonstrates how to use SignXAI's PyTorch implementation with Zennit for generating LRP explanations:

.. code-block:: python

    import torch
    import torchvision.models as models
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    
    from signxai.torch_signxai import calculate_relevancemap
    from signxai.torch_signxai.utils import remove_softmax
    from signxai.common.visualization import normalize_relevance_map, relevance_to_heatmap, overlay_heatmap
    
    # Load pre-trained model
    model = models.vgg16(pretrained=True)
    model.eval()
    
    # Remove softmax (required for proper explanations)
    model_no_softmax = remove_softmax(model)
    
    # Load and preprocess image
    img = Image.open("example.jpg").resize((224, 224))
    img_tensor = torch.FloatTensor(np.array(img)).permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Calculate relevance maps using different LRP methods
    lrp_eps = calculate_relevancemap(model_no_softmax, img_tensor, method="lrp_epsilon", epsilon=0.1)
    lrp_ab = calculate_relevancemap(model_no_softmax, img_tensor, method="lrp_alphabeta")
    lrp_composite = calculate_relevancemap(
        model_no_softmax, 
        img_tensor, 
        method="lrp_sequential",
        first_layer_rule="zbox",
        middle_layer_rule="alphabeta",
        last_layer_rule="epsilon"
    )
    
    # Visualize relevance maps
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title("Original")
    
    # LRP-Epsilon
    norm_lrp_eps = normalize_relevance_map(lrp_eps[0].sum(axis=0))
    heatmap = relevance_to_heatmap(norm_lrp_eps)
    axs[1].imshow(overlay_heatmap(np.array(img)/255.0, heatmap))
    axs[1].set_title("LRP-Epsilon")
    
    # LRP-AlphaBeta
    norm_lrp_ab = normalize_relevance_map(lrp_ab[0].sum(axis=0))
    heatmap = relevance_to_heatmap(norm_lrp_ab)
    axs[2].imshow(overlay_heatmap(np.array(img)/255.0, heatmap))
    axs[2].set_title("LRP-AlphaBeta")
    
    # LRP-Composite
    norm_lrp_comp = normalize_relevance_map(lrp_composite[0].sum(axis=0))
    heatmap = relevance_to_heatmap(norm_lrp_comp)
    axs[3].imshow(overlay_heatmap(np.array(img)/255.0, heatmap))
    axs[3].set_title("LRP-Composite")
    
    plt.tight_layout()
    plt.show()

Advanced Zennit Configuration
-----------------------------

For advanced users, SignXAI exposes more detailed Zennit configurations:

.. code-block:: python

    from signxai.torch_signxai.methods.zennit_impl import AdvancedLRPAnalyzer
    from zennit.composites import EpsilonPlusFlat
    
    # Create custom composite with layer-specific rules
    from zennit.composites import LayerMapComposite
    from zennit.rules import Epsilon, ZPlus, Gamma
    from zennit.types import Convolution, Linear
    
    # Define layer mapping
    layer_map = {
        Convolution: ZPlus(),  # Use ZPlus for convolutional layers
        Linear: Epsilon(epsilon=0.1)  # Use Epsilon for linear layers
    }
    
    # Create analyzer with custom composite
    analyzer = AdvancedLRPAnalyzer(model, rule_type="custom", composite=LayerMapComposite(layer_map))
    explanation = analyzer.analyze(input_tensor, target_class=class_idx)

This flexibility allows for very fine-grained control over the explanation process.

SIGN Methods
------------

SignXAI implements the novel SIGN methods for PyTorch models:

.. code-block:: python

    from signxai.torch_signxai.methods.signed import calculate_sign_mu
    
    # Calculate sign with threshold mu
    sign = calculate_sign_mu(input_tensor, mu=0.0)
    
    # Use with gradient-based methods
    grad = GradientAnalyzer(model).analyze(input_tensor)
    grad_sign = grad * sign

This can be used with any of the analyzers to create SIGN variants of the methods.

Additional Analyzer Classes
---------------------------

The following analyzer classes were added to provide comprehensive XAI method coverage:

Gradient × Sign Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class GradientXSignAnalyzer:
        """Gradient × Sign analyzer with threshold parameter.
        
        Implements gradient × sign method with configurable mu threshold.
        """
        
        def __init__(self, model, mu=0.0):
            """Initialize Gradient × Sign analyzer.
            
            Args:
                model: PyTorch model
                mu: Threshold parameter for sign calculation
            """
            self.model = model
            self.mu = mu
        
        def analyze(self, input_tensor, target_class=None):
            """Generate gradient × sign attribution.
            
            Args:
                input_tensor: Input tensor
                target_class: Target class index (None for argmax)
                
            Returns:
                Gradient × sign attribution
            """
            # Implementation calculates gradient and applies sign with threshold
            return attribution

Gradient × Input Analyzer
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class GradientXInputAnalyzer:
        """Gradient × Input analyzer.
        
        Implements the gradient × input method for enhanced feature attribution.
        """
        
        def __init__(self, model):
            """Initialize Gradient × Input analyzer."""
            self.model = model
        
        def analyze(self, input_tensor, target_class=None):
            """Generate gradient × input attribution."""
            # Implementation multiplies gradient with input
            return attribution

VarGrad Analyzer
~~~~~~~~~~~~~~~~

.. code-block:: python

    class VarGradAnalyzer:
        """VarGrad analyzer.
        
        Implements variance of gradients across multiple noisy samples.
        """
        
        def __init__(self, model, num_samples=50, noise_level=0.2):
            """Initialize VarGrad analyzer.
            
            Args:
                model: PyTorch model
                num_samples: Number of noisy samples
                noise_level: Level of Gaussian noise
            """
            self.model = model
            self.num_samples = num_samples
            self.noise_level = noise_level
        
        def analyze(self, input_tensor, target_class=None):
            """Generate VarGrad attribution."""
            # Implementation calculates variance across noisy samples
            return attribution

Deep Taylor Analyzer
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class DeepTaylorAnalyzer:
        """Deep Taylor analyzer using LRP epsilon as proxy.
        
        Implements Deep Taylor decomposition by leveraging LRP epsilon rule.
        """
        
        def __init__(self, model, epsilon=1e-6):
            """Initialize Deep Taylor analyzer.
            
            Args:
                model: PyTorch model
                epsilon: Stabilizing factor for epsilon rule
            """
            self.model = model
            self.epsilon = epsilon
        
        def analyze(self, input_tensor, target_class=None):
            """Generate Deep Taylor attribution."""
            # Implementation uses LRP epsilon as Deep Taylor proxy
            return attribution

Performance Considerations
--------------------------

When using Zennit through SignXAI, consider these performance tips:

1. **Model Complexity** - LRP methods scale with model complexity
2. **Batch Size** - Process multiple examples simultaneously for efficiency
3. **GPU Acceleration** - Ensure PyTorch is using CUDA for better performance
4. **Memory Usage** - For large models or inputs, consider gradient checkpointing
5. **Parallelization** - Use DataParallel for multi-GPU setups

Extending with New Methods
--------------------------

To add new methods, you can create a new analyzer class in ``signxai.torch_signxai.methods.zennit_impl.py``:

.. code-block:: python

    class MyCustomAnalyzer:
        """Custom explanation method.
        
        Implements a custom explanation method using Zennit.
        """
        
        def __init__(self, model, **kwargs):
            """Initialize custom analyzer.
            
            Args:
                model: PyTorch model
                **kwargs: Additional parameters
            """
            self.model = model
            # Setup any necessary parameters
            
        def analyze(self, input_tensor, target_class=None):
            """Generate custom attribution.
            
            Args:
                input_tensor: Input tensor
                target_class: Target class index (None for argmax)
                
            Returns:
                Custom attribution
            """
            # Implement custom attribution logic
            # ...
            
            return attribution
===============
PyTorch Module
===============

.. py:module:: signxai.torch_signxai

The ``signxai.torch_signxai`` module provides explainability methods for PyTorch models, leveraging the Zennit library for LRP implementations.

.. contents:: Contents
   :local:
   :depth: 2

Main Functions
--------------

This module provides two API styles: a PyTorch-native style and a TensorFlow-compatible style.

PyTorch-Native API
~~~~~~~~~~~~~~~~~~

.. py:function:: explain(model, input_tensor, method_name, **kwargs)

   Calculates a relevance map using the new dynamic parsing API.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param input_tensor: Input tensor
   :type input_tensor: torch.Tensor or numpy.ndarray
   :param method_name: Name of the explanation method with embedded parameters (e.g., 'lrp_epsilon_0_25', 'smoothgrad_noise_0_3_samples_50')
   :type method_name: str
   :param kwargs: Additional arguments for the specific method
   :return: Relevance map as numpy array
   :rtype: numpy.ndarray
   
   **Note**: The old ``calculate_relevancemap(model, input_tensor, method="gradients")`` API is deprecated. Use this new unified API instead.
   
For multiple inputs, use ``explain()`` in a loop or with batch processing:

.. code-block:: python

   from signxai.api import explain
   
   # For multiple inputs
   explanations = []
   for input_tensor in input_tensors:
       explanations.append(explain(model, input_tensor, method_name="gradient"))
   
   # Or for batch processing (if supported by the method)
   batch_explanation = explain(model, batch_inputs, method_name="gradient")

TensorFlow-Compatible API
~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:module:: signxai.torch_signxai.methods.wrappers

**Deprecated**: The TensorFlow-compatible API style is now replaced by the unified ``explain()`` API.

Instead of ``calculate_relevancemap(method_name, x, model_no_softmax)`` use:

.. code-block:: python

   from signxai.api import explain
   explanation = explain(model_no_softmax, x, method_name=method_name)
   
**Deprecated**: Use the new unified ``explain()`` API instead:

.. code-block:: python

   from signxai.api import explain
   
   # For multiple inputs
   explanations = []
   for input_tensor in X:
       explanations.append(explain(model_no_softmax, input_tensor, method_name=method_name))

Zennit Integration
------------------

The module ``signxai.torch_signxai.methods.zennit_impl`` provides Zennit-based implementations of explanation methods.

.. py:module:: signxai.torch_signxai.methods.zennit_impl

.. py:class:: GradientAnalyzer(model)

   Implements vanilla gradient calculation aligned with TensorFlow's implementation.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate vanilla gradient attribution.
      
      **New API**: Use ``explain(model, input_tensor, method_name="gradient")``
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Gradient attribution
      :rtype: numpy.ndarray

.. py:class:: IntegratedGradientsAnalyzer(model, steps=50, baseline=None)

   Implements integrated gradients by integrating gradients along a straight path from baseline to input.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param steps: Number of steps for integration
   :type steps: int, optional
   :param baseline: Baseline input (None for zeros)
   :type baseline: torch.Tensor, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate integrated gradients attribution.
      
      **New API**: Use ``explain(model, input_tensor, method_name="integrated_gradients_steps_50")`` or other step values like ``integrated_gradients_steps_100``
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Integrated gradients attribution
      :rtype: numpy.ndarray

.. py:class:: SmoothGradAnalyzer(model, noise_level=0.2, num_samples=50)

   Implements SmoothGrad by adding Gaussian noise to the input multiple times and averaging the resulting gradients.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param noise_level: Level of Gaussian noise to add
   :type noise_level: float, optional
   :param num_samples: Number of noisy samples to average
   :type num_samples: int, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate SmoothGrad attribution.
      
      **New API**: Use ``explain(model, input_tensor, method_name="smoothgrad_noise_0_2_samples_50")`` or other parameter combinations
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: SmoothGrad attribution
      :rtype: numpy.ndarray

.. py:class:: GuidedBackpropAnalyzer(model)

   Implements guided backpropagation by modifying the backward pass of ReLU to only pass positive gradients.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate guided backpropagation attribution.
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Guided backpropagation attribution
      :rtype: numpy.ndarray

.. py:class:: GradientXInputAnalyzer(model)

   Implements gradient × input method for enhanced feature attribution.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate gradient × input attribution.
      
      **New API**: Use ``explain(model, input_tensor, method_name="gradient_x_input")``
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Gradient × input attribution
      :rtype: numpy.ndarray

.. py:class:: GradientXSignAnalyzer(model, mu=0.0)

   Implements gradient × sign method with configurable threshold parameter.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param mu: Threshold parameter for sign calculation
   :type mu: float, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate gradient × sign attribution.
      
      **New API**: Use ``explain(model, input_tensor, method_name="gradient_x_sign")`` or with mu values like ``gradient_x_sign_mu_0_5`` or ``gradient_x_sign_mu_neg_0_5``
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Gradient × sign attribution
      :rtype: numpy.ndarray

.. py:class:: VarGradAnalyzer(model, num_samples=50, noise_level=0.2)

   Implements variance of gradients across multiple noisy samples.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param num_samples: Number of noisy samples to average
   :type num_samples: int, optional
   :param noise_level: Level of Gaussian noise to add
   :type noise_level: float, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate VarGrad attribution.
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: VarGrad attribution
      :rtype: numpy.ndarray

.. py:class:: DeepTaylorAnalyzer(model, epsilon=1e-6)

   Implements Deep Taylor decomposition using LRP epsilon as proxy.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param epsilon: Stabilizing factor for epsilon rule
   :type epsilon: float, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate Deep Taylor attribution.
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Deep Taylor attribution
      :rtype: numpy.ndarray

.. py:class:: GradCAMAnalyzer(model, target_layer=None)

   Implements Grad-CAM by using the gradients of a target class with respect to feature maps of a convolutional layer.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param target_layer: Target convolutional layer (None to auto-detect)
   :type target_layer: torch.nn.Module, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate Grad-CAM attribution.
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: Grad-CAM attribution
      :rtype: numpy.ndarray

Layer-wise Relevance Propagation (LRP)
--------------------------------------

The Zennit library is used to implement various LRP variants.

.. py:class:: LRPAnalyzer(model, rule="epsilon", epsilon=1e-6)

   Layer-wise Relevance Propagation analyzer using Zennit's implementation.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param rule: LRP rule ('epsilon', 'zplus', 'alphabeta')
   :type rule: str, optional
   :param epsilon: Stabilizing factor for epsilon rule
   :type epsilon: float, optional
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate LRP attribution.
      
      **New API**: Use ``explain(model, input_tensor, method_name="lrp_epsilon_0_25")`` or other variants like ``lrp_alpha_1_beta_0``
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: LRP attribution
      :rtype: numpy.ndarray

.. py:class:: AdvancedLRPAnalyzer(model, rule_type, **kwargs)

   Advanced LRP analyzer with specialized rules and composites.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param rule_type: Type of LRP rule/composite
   :type rule_type: str
   :param kwargs: Additional parameters for specific rules
   
   **Available rule types**:
   
   - "alpha1beta0": Alpha-Beta rule with alpha=1, beta=0
   - "alpha2beta1": Alpha-Beta rule with alpha=2, beta=1
   - "epsilon": Epsilon rule with custom epsilon value
   - "gamma": Gamma rule with custom gamma value
   - "flat": Flat rule
   - "wsquare": W-Square rule
   - "zbox": Z-Box rule with custom low/high values
   - "sequential": Sequential application of different rules
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate advanced LRP attribution.
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: LRP attribution
      :rtype: numpy.ndarray

.. py:class:: LRPSequential(model, first_layer_rule="zbox", middle_layer_rule="alphabeta", last_layer_rule="epsilon", **kwargs)

   Implements LRP with sequential application of different rules to different layers.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :param first_layer_rule: Rule for first layers
   :type first_layer_rule: str, optional
   :param middle_layer_rule: Rule for middle layers
   :type middle_layer_rule: str, optional
   :param last_layer_rule: Rule for last layers
   :type last_layer_rule: str, optional
   :param kwargs: Additional parameters for specific rules
   
   .. py:method:: analyze(input_tensor, target_class=None)
      
      Generate sequential LRP attribution.
      
      :param input_tensor: Input tensor
      :type input_tensor: torch.Tensor or numpy.ndarray
      :param target_class: Target class index (None for argmax)
      :type target_class: int, optional
      :return: LRP attribution
      :rtype: numpy.ndarray

SIGN Methods
------------

The SIGN methods are implemented for PyTorch models as well.

.. py:module:: signxai.torch_signxai.methods.signed

.. py:function:: calculate_sign_mu(x, mu=0)

   Calculates the sign with a threshold parameter mu for PyTorch inputs.
   
   **New API**: This is typically used internally, but sign-based methods can be called with ``explain(model, input_tensor, method_name="gradient_x_sign_mu_0_5")`` or ``gradient_x_sign_mu_neg_0_5``
   
   :param x: Input tensor
   :type x: torch.Tensor or numpy.ndarray
   :param mu: Threshold parameter (default: 0)
   :type mu: float
   :return: Sign tensor
   :rtype: torch.Tensor or numpy.ndarray (matches input type)

Utility Functions
-----------------

.. py:module:: signxai.torch_signxai.utils

.. py:function:: remove_softmax(model)

   Removes the softmax activation from a PyTorch model.
   
   :param model: PyTorch model
   :type model: torch.nn.Module
   :return: Model with softmax removed (outputs raw logits)
   :rtype: torch.nn.Module
   
.. py:class:: NoSoftmaxWrapper(model)

   Wrapper class that removes softmax from a PyTorch model.
   
   :param model: PyTorch model with softmax
   :type model: torch.nn.Module
   
   .. py:method:: forward(x)
      
      Forward pass that returns logits directly (no softmax).
      
      :param x: Input tensor
      :type x: torch.Tensor
      :return: Model output before softmax
      :rtype: torch.Tensor
=========================
Explanation Methods List
=========================

SignXAI provides a comprehensive list of explanation methods for both TensorFlow and PyTorch models. This page details all available methods, their parameters, and framework compatibility.

.. contents:: Contents
   :local:
   :depth: 2

Library Attribution
------------------

SignXAI builds upon two powerful explainability libraries, each providing the backend implementation for various explanation methods:

- **iNNvestigate**: Powers the TensorFlow implementation of Layer-wise Relevance Propagation (LRP) and other methods
  
  - Developed at TU Berlin by M. Alber et al.
  - Original paper: `iNNvestigate neural networks! <https://doi.org/10.1007/s00521-019-04041-y>`_
  - Repository: `github.com/albermax/innvestigate <https://github.com/albermax/innvestigate>`_

- **Zennit**: Powers the PyTorch implementation of Layer-wise Relevance Propagation (LRP) variants
  
  - Developed at TU Berlin by C. Anders et al.
  - Original paper: `Software for Dataset-wide XAI <https://arxiv.org/abs/2106.13200>`_
  - Repository: `github.com/chr5tphr/zennit <https://github.com/chr5tphr/zennit>`_

The SIGN method, which is SignXAI's novel contribution, builds upon these libraries to provide improved explanations by reducing bias.

Method Overview
--------------

The table below shows all available explanation methods in SignXAI. Methods are implemented as follows:

- **Gradient-based methods**: Direct implementation in both TensorFlow and PyTorch
- **Guided Backpropagation**: Direct implementation in both TensorFlow and PyTorch
- **Grad-CAM**: Direct implementation in both TensorFlow and PyTorch
- **LRP methods (TensorFlow)**: Implemented using iNNvestigate
- **LRP methods (PyTorch)**: Implemented using Zennit
- **SIGN method**: Original SignXAI contribution, extending both iNNvestigate and Zennit

.. list-table:: Available Explanation Methods
   :widths: 20 50 10 10
   :header-rows: 1

   * - Method Name
     - Description
     - TensorFlow
     - PyTorch
   * - ``gradient``
     - Vanilla gradient
     - ✓
     - ✓
   * - ``input_t_gradient`` or ``gradient_x_input``
     - Gradient multiplied by input
     - ✓
     - ✓
   * - ``gradient_x_sign``
     - Gradient multiplied by sign of input
     - ✓
     - ✓
   * - ``gradient_x_sign_mu``
     - Gradient multiplied by sign with threshold mu
     - ✓
     - ✓
   * - ``guided_backprop``
     - Guided backpropagation
     - ✓
     - ✓
   * - ``guided_backprop_x_sign``
     - Guided backpropagation multiplied by sign
     - ✓
     - ✓
   * - ``guided_backprop_x_sign_mu``
     - Guided backpropagation multiplied by sign with threshold mu
     - ✓
     - ✓
   * - ``integrated_gradients``
     - Integrated gradients
     - ✓
     - ✓
   * - ``smoothgrad``
     - SmoothGrad
     - ✓
     - ✓
   * - ``smoothgrad_x_sign``
     - SmoothGrad multiplied by sign
     - ✓
     - ✓
   * - ``vargrad``
     - VarGrad (variance of gradients across noisy samples)
     - ✓
     - ✓
   * - ``deconvnet``
     - DeconvNet
     - ✓
     - ✓
   * - ``grad_cam``
     - Grad-CAM
     - ✓
     - ✓
   * - ``grad_cam_timeseries``
     - Grad-CAM for time series data
     - ✓
     - ✓
   * - ``lrp_z``
     - LRP-Z rule
     - ✓
     - ✓
   * - ``lrpsign_z``
     - LRP-Z with SIGN input layer rule
     - ✓
     - ✓
   * - ``lrp_epsilon_{value}``
     - LRP-epsilon with specified epsilon value
     - ✓
     - ✓
   * - ``lrpsign_epsilon_{value}``
     - LRP-epsilon with SIGN input layer rule
     - ✓
     - ✓
   * - ``lrp_epsilon_{value}_std_x``
     - LRP-epsilon with epsilon proportional to input std
     - ✓
     - ✓
   * - ``lrp_alpha_1_beta_0``
     - LRP-AlphaBeta with alpha=1, beta=0
     - ✓
     - ✓
   * - ``lrpsign_alpha_1_beta_0``
     - LRP-AlphaBeta with SIGN input layer rule
     - ✓
     - ✓
   * - ``lrp_sequential_composite_a``
     - LRP composite with layer-specific rules (variant A)
     - ✓
     - ✓
   * - ``lrp_sequential_composite_b``
     - LRP composite with layer-specific rules (variant B)
     - ✓
     - ✓

Gradient-Based Methods
---------------------

*Implemented directly in SignXAI with framework-specific optimizations*

Vanilla Gradient
~~~~~~~~~~~~~~

Method name: ``gradient``

Computes the gradient of the target output with respect to the input, highlighting features that have the greatest effect on the output.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Gradient x Input
~~~~~~~~~~~~~~

Method name: ``input_t_gradient`` or ``gradient_x_input``

Element-wise multiplication of the gradient with the input to reduce noise and improve visualization.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Gradient x SIGN
~~~~~~~~~~~~~

Method name: ``gradient_x_sign``

Multiplies the gradient with the sign of the input, focusing on the input's direction rather than magnitude.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Method name: ``gradient_x_sign_mu``

Includes a threshold parameter mu for more flexible sign thresholding.

**Parameters:**

- ``mu``: Threshold parameter (default: 0)
- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Integrated Gradients
~~~~~~~~~~~~~~~~~~

Method name: ``integrated_gradients``

Computes gradients along a straight-line path from a baseline to the input to better attribute feature importance.

**Parameters:**

- ``steps``: Number of steps for integration (default: 50)
- ``reference_inputs``: Baseline input (default: zeros)
- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

SmoothGrad
~~~~~~~~~

Method name: ``smoothgrad``

Computes average gradients from multiple input samples with added noise to produce smoother, more visually interpretable heatmaps.

**Parameters:**

- ``augment_by_n``: Number of noisy samples (default: 50)
- ``noise_scale``: Scale of Gaussian noise (default: 0.2)
- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

VarGrad
~~~~~~~

Method name: ``vargrad``

Computes the variance of gradients across multiple noisy samples to identify unstable attributions.

**Parameters:**

- ``num_samples``: Number of noisy samples (default: 50)
- ``noise_level``: Level of Gaussian noise (default: 0.2)
- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Guided Backpropagation
---------------------

*Implemented directly in SignXAI with framework-specific optimizations*

Method name: ``guided_backprop``

Modifies the ReLU gradient to only pass positive gradients, producing sharper visualization.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Method name: ``guided_backprop_x_sign``

Multiplies guided backpropagation with the sign of the input for enhanced visualization.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Method name: ``guided_backprop_x_sign_mu``

Includes a threshold parameter mu for more flexible sign thresholding.

**Parameters:**

- ``mu``: Threshold parameter
- ``neuron_selection``: Target output neuron (class) for which to compute the gradient (default: argmax)

Grad-CAM
-------

*Implemented directly in SignXAI with framework-specific optimizations*

Method name: ``grad_cam``

Generates a localization map highlighting important regions by using the gradients flowing into the final convolutional layer.

**Parameters:**

- ``last_conv_layer_name``: Name of the last convolutional layer (auto-detected if None)
- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``resize``: Whether to resize the output to match input dimensions (default: True)

Method name: ``grad_cam_timeseries``

Specialized version of Grad-CAM for time series data.

**Parameters:**

- ``last_conv_layer_name``: Name of the last convolutional layer (auto-detected if None)
- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``resize``: Whether to resize the output to match input dimensions (default: True)

Layer-wise Relevance Propagation (LRP)
-------------------------------------

*TensorFlow implementation provided by iNNvestigate; PyTorch implementation provided by Zennit*

LRP Base Methods
~~~~~~~~~~~~~~

Method name: ``lrp_z``

Basic LRP implementation following the z-rule.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``input_layer_rule``: Rule for the input layer (default: None)

Method name: ``lrpsign_z``

LRP-Z with SIGN input layer rule.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)

LRP with Epsilon
~~~~~~~~~~~~~~

Methods: ``lrp_epsilon_{value}`` (e.g., ``lrp_epsilon_0_1``, ``lrp_epsilon_1``, etc.)

LRP with epsilon stabilization factor.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``input_layer_rule``: Rule for the input layer (default: None)

Methods: ``lrpsign_epsilon_{value}`` (e.g., ``lrpsign_epsilon_0_1``)

LRP-epsilon with SIGN input layer rule.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)

Methods: ``lrp_epsilon_{value}_std_x`` (e.g., ``lrp_epsilon_0_1_std_x``)

LRP with epsilon proportional to the standard deviation of the input.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``input_layer_rule``: Rule for the input layer (default: None)

LRP with Alpha-Beta
~~~~~~~~~~~~~~~~~

Method name: ``lrp_alpha_1_beta_0``

LRP with separate treatment of positive and negative contributions.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``input_layer_rule``: Rule for the input layer (default: None)

Method name: ``lrpsign_alpha_1_beta_0``

LRP Alpha-Beta with SIGN input layer rule.

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)

LRP Composite Rules
~~~~~~~~~~~~~~~~

Method name: ``lrp_sequential_composite_a``

LRP with layer-specific rules (variant A).

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``input_layer_rule``: Rule for the input layer (default: None)

Method name: ``lrp_sequential_composite_b``

LRP with layer-specific rules (variant B).

**Parameters:**

- ``neuron_selection``: Target output neuron (class) (default: argmax)
- ``input_layer_rule``: Rule for the input layer (default: None)

Deep Taylor Decomposition
~~~~~~~~~~~~~~~~~~~~~~~

Method name: ``deep_taylor``

Implements Deep Taylor decomposition using LRP epsilon as a proxy method.

**Parameters:**

- ``epsilon``: Stabilizing factor for epsilon rule (default: 1e-6)
- ``neuron_selection``: Target output neuron (class) (default: argmax)

Framework-Specific Parameters
---------------------------

Some parameters have different meanings or implementations between TensorFlow and PyTorch.

TensorFlow-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``model_no_softmax``: Model with softmax removed (done automatically)
- ``input_layer_rule``: Input layer rule for LRP methods ('Z', 'SIGN', 'Bounded', 'WSquare', 'Flat')

PyTorch-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

- ``target_layer``: Target layer for Grad-CAM (auto-detected if None)
- ``rule``: LRP rule in Zennit implementation ('epsilon', 'zplus', 'alphabeta')
- ``rule_type``: Advanced LRP rule type ('alpha1beta0', 'epsilon', 'gamma', etc.)

Common Parameters
~~~~~~~~~~~~~~

- ``target_class``: Target class index (used in PyTorch implementation)
- ``neuron_selection``: Target neuron/class (used in TensorFlow implementation)

Both have the same meaning and can be used interchangeably depending on the framework.
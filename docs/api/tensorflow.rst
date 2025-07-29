==================
TensorFlow Module
==================

.. py:module:: signxai.tf_signxai

The ``signxai.tf_signxai`` module provides explainability methods for TensorFlow models, leveraging the iNNvestigate library for LRP implementations.

.. contents:: Contents
   :local:
   :depth: 2

Main Functions
--------------

.. py:function:: calculate_relevancemap(method, input_tensor, model, **kwargs)

   Calculates a relevance map for a given method, input, and model.
   
   :param method: Name of the explanation method
   :type method: str
   :param input_tensor: Input tensor (can be numpy array or TensorFlow tensor)
   :type input_tensor: numpy.ndarray or tf.Tensor
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param kwargs: Additional arguments for the specific method
   :return: Relevance map as numpy array
   :rtype: numpy.ndarray
   
.. py:function:: calculate_relevancemaps(method, inputs, model, **kwargs)

   Calculates relevance maps for multiple inputs.
   
   :param method: Name of the explanation method
   :type method: str
   :param inputs: List or batch of input tensors
   :type inputs: list or numpy.ndarray or tf.Tensor
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param kwargs: Additional arguments for the specific method
   :return: List or batch of relevance maps
   :rtype: numpy.ndarray

Gradient-Based Methods
----------------------

.. py:module:: signxai.tf_signxai.methods.wrappers

The methods module provides implementations of various explainability methods.

Vanilla Gradient
~~~~~~~~~~~~~~~~

.. py:function:: gradient(model_no_softmax, x, **kwargs)

   Computes vanilla gradients of the model output with respect to the input.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param kwargs: Additional arguments including neuron_selection for specifying target class
   :return: Gradient-based attribution
   :rtype: numpy.ndarray

Input x Gradient
~~~~~~~~~~~~~~~~

.. py:function:: input_t_gradient(model_no_softmax, x, **kwargs)

   Computes the element-wise product of gradients and input.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param kwargs: Additional arguments including neuron_selection for specifying target class
   :return: Gradient x Input attribution
   :rtype: numpy.ndarray

Integrated Gradients
~~~~~~~~~~~~~~~~~~~~

.. py:function:: integrated_gradients(model_no_softmax, x, **kwargs)

   Computes integrated gradients by integrating gradients along a straight path from reference to input.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param kwargs: Additional arguments including:
   
      - steps: Number of integration steps (default: 50)
      - reference_inputs: Baseline input (default: zeros)
      - neuron_selection: Target class
      
   :return: Integrated gradients attribution
   :rtype: numpy.ndarray

SmoothGrad
~~~~~~~~~~

.. py:function:: smoothgrad(model_no_softmax, x, **kwargs)

   Computes smoothgrad by adding noise to the input and averaging the resulting gradients.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param kwargs: Additional arguments including:
   
      - augment_by_n: Number of noisy samples (default: 50)
      - noise_scale: Scale of Gaussian noise (default: 0.2)
      - neuron_selection: Target class
      
   :return: SmoothGrad attribution
   :rtype: numpy.ndarray

SIGN Methods
------------

The Sign module provides implementations of the SIGN explainability methods.

.. py:module:: signxai.tf_signxai.methods.signed

.. py:function:: calculate_sign_mu(x, mu=0, **kwargs)

   Calculates the sign with a threshold parameter mu.
   
   :param x: Input tensor
   :type x: numpy.ndarray
   :param mu: Threshold parameter (default: 0)
   :type mu: float
   :param kwargs: Additional arguments
   :return: Sign tensor
   :rtype: numpy.ndarray

Gradient x SIGN
~~~~~~~~~~~~~~~

.. py:function:: gradient_x_sign(model_no_softmax, x, **kwargs)

   Computes the element-wise product of gradients and sign of the input.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param kwargs: Additional arguments including neuron_selection for specifying target class
   :return: Gradient x SIGN attribution
   :rtype: numpy.ndarray

.. py:function:: gradient_x_sign_mu(model_no_softmax, x, mu, **kwargs)

   Computes the element-wise product of gradients and sign of the input with threshold parameter mu.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param mu: Threshold parameter
   :type mu: float
   :param kwargs: Additional arguments including neuron_selection for specifying target class
   :return: Gradient x SIGN attribution with threshold
   :rtype: numpy.ndarray

Guided Backpropagation
----------------------

.. py:module:: signxai.tf_signxai.methods.guided_backprop

.. py:function:: guided_backprop(model_no_softmax, x, **kwargs)

   Computes guided backpropagation by modifying the ReLU gradient to only pass positive gradients.
   
   :param model_no_softmax: TensorFlow model with softmax removed
   :type model_no_softmax: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param kwargs: Additional arguments including neuron_selection for specifying target class
   :return: Guided backpropagation attribution
   :rtype: numpy.ndarray

.. py:function:: guided_backprop_on_guided_model(model, x, layer_name=None, **kwargs)

   Creates a guided model and computes guided backpropagation.
   
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param layer_name: Target layer name (for GradCAM)
   :type layer_name: str, optional
   :param kwargs: Additional arguments
   :return: Guided backpropagation attribution
   :rtype: numpy.ndarray

Grad-CAM
--------

.. py:module:: signxai.tf_signxai.methods.grad_cam

.. py:function:: calculate_grad_cam_relevancemap(x, model, last_conv_layer_name=None, neuron_selection=None, resize=True, **kwargs)

   Computes Grad-CAM by using the gradients of a target class with respect to feature maps of a convolutional layer.
   
   :param x: Input tensor
   :type x: numpy.ndarray
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param last_conv_layer_name: Name of the last convolutional layer
   :type last_conv_layer_name: str, optional
   :param neuron_selection: Target class
   :type neuron_selection: int, optional
   :param resize: Whether to resize the output to match input dimensions
   :type resize: bool, optional
   :param kwargs: Additional arguments
   :return: Grad-CAM attribution
   :rtype: numpy.ndarray

.. py:function:: calculate_grad_cam_relevancemap_timeseries(x, model, last_conv_layer_name=None, neuron_selection=None, resize=True, **kwargs)

   Computes Grad-CAM specifically for time series data.
   
   :param x: Input tensor (time series)
   :type x: numpy.ndarray
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param last_conv_layer_name: Name of the last convolutional layer
   :type last_conv_layer_name: str, optional
   :param neuron_selection: Target class
   :type neuron_selection: int, optional
   :param resize: Whether to resize the output to match input dimensions
   :type resize: bool, optional
   :param kwargs: Additional arguments
   :return: Grad-CAM attribution for time series
   :rtype: numpy.ndarray

Layer-wise Relevance Propagation (LRP)
--------------------------------------

The iNNvestigate module provides LRP implementations for TensorFlow. This is the key integration point for iNNvestigate in SignXAI.

.. py:module:: signxai.utils.utils

.. py:function:: calculate_explanation_innvestigate(model, x, method, **kwargs)

   Interface to iNNvestigate for LRP and other methods.
   
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param method: iNNvestigate method name (e.g., 'lrp.z', 'lrp.epsilon', etc.)
   :type method: str
   :param kwargs: Additional arguments including:
   
      - neuron_selection: Target class
      - input_layer_rule: Input layer rule ('Z', 'SIGN', 'Bounded', etc.)
      - epsilon: Epsilon value for LRP-epsilon
      - stdfactor: Standard deviation factor for LRP with varying epsilon
      
   :return: LRP attribution
   :rtype: numpy.ndarray

LRP Variants
~~~~~~~~~~~~

The module provides various LRP variants through iNNvestigate. Key implemented variants include:

1. **LRP-z**: Basic LRP implementation
2. **LRP-epsilon**: LRP with a stabilizing factor (epsilon)
3. **LRP-alpha-beta**: LRP with separate treatment of positive and negative contributions
4. **LRP with SIGN Input Layer Rule**: The novel SIGN method applied to LRP
5. **LRP Composite**: Layer-specific LRP rules

Utility Functions
-----------------

.. py:function:: remove_softmax(model)

   Removes the softmax activation from a TensorFlow model.
   
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :return: Model with softmax removed (outputs raw logits)
   :rtype: tf.keras.Model
==============================
TensorFlow Implementation
==============================

.. py:module:: signxai.tf_signxai
   :no-index:

This guide provides a detailed explanation of SignXAI's TensorFlow implementation, with a focus on how the package integrates with iNNvestigate for Layer-wise Relevance Propagation (LRP) methods.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The TensorFlow implementation in SignXAI provides a comprehensive set of explainability methods for TensorFlow models. It uses the iNNvestigate library as a backend for LRP methods, providing a seamless integration and an extensive array of explanation techniques.

iNNvestigate Integration
------------------------

The most powerful aspect of the TensorFlow implementation is its integration with iNNvestigate for LRP methods. This section explains how SignXAI leverages iNNvestigate's capabilities.

About iNNvestigate
~~~~~~~~~~~~~~~~~~

`iNNvestigate <https://github.com/albermax/innvestigate>`_ is a powerful library for explaining neural networks developed at the Technical University of Berlin by Maximilian Alber and colleagues. It offers a comprehensive framework for analyzing and interpreting decisions of neural networks and is particularly renowned for its implementation of various LRP methods.

.. admonition:: Citation

   If you use iNNvestigate in your research through SignXAI, please consider citing the original work:

   .. code-block:: bibtex

      @article{alber2019innvestigate,
        title={iNNvestigate neural networks!},
        author={Alber, Maximilian and Lapuschkin, Sebastian and Seegerer, Philipp and H{\"a}gele, Miriam and Sch{\"u}tt, Kristof T and Montavon, Gr{\'e}goire and Samek, Wojciech and M{\"u}ller, Klaus-Robert and D{\"a}hne, Sven and Kindermans, Pieter-Jan},
        journal={Journal of Machine Learning Research},
        volume={20},
        number={93},
        pages={1--8},
        year={2019}
      }

SignXAI embeds a carefully curated version of iNNvestigate directly within the package, allowing for:

1. Seamless integration without external dependencies
2. Customizations specific to SignXAI's use cases
3. Compatibility with modern TensorFlow versions

How SignXAI Uses iNNvestigate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integration happens primarily through the ``calculate_explanation_innvestigate`` function in ``signxai.utils.utils``:

.. code-block:: python

    def calculate_explanation_innvestigate(model, x, method, **kwargs):
        # Create analyzer based on method
        analyzer = create_analyzer(model, method, **kwargs)
        
        # Generate explanation
        explanation = analyzer.analyze(x, **kwargs)
        
        return explanation

This function serves as a bridge between SignXAI's API and iNNvestigate's internal methods.

Embedded iNNvestigate Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SignXAI includes a tailored version of iNNvestigate at ``signxai.tf_signxai.methods.innvestigate``. This module contains:

1. **analyzer/** - Core analysis algorithms
   - **base.py** - Base analyzer class
   - **gradient_based.py** - Gradient-based methods
   - **relevance_based/** - LRP implementation
   - **reverse_map.py** - Reverse mapping utilities

2. **applications/** - Model-specific utilities
   - **imagenet.py** - ImageNet specific utilities
   - **mnist.py** - MNIST specific utilities

3. **backend/** - Framework backend implementation
   - **tensorflow.py** - TensorFlow-specific functions

4. **utils/** - Helper functions and utilities
   - **keras/** - Keras graph utilities
   - **visualizations.py** - Visualization tools

This embedded structure ensures that SignXAI is self-contained and doesn't require external installations or version management for iNNvestigate.

LRP Methods in Detail
---------------------

LRP methods are implemented through iNNvestigate. The key method variants include:

LRP-Z
~~~~~

The basic LRP rule, which distributes relevance based on the ratio of positive activations.

.. code-block:: python

    def lrp_z(model_no_softmax, x, **kwargs):
        return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.z', **kwargs)

LRP-Epsilon
~~~~~~~~~~~

Adds a small epsilon value to stabilize the division operation, preventing numerical instabilities.

.. code-block:: python

    def lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
        return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.epsilon', epsilon=0.1, **kwargs)

LRP-Alpha-Beta
~~~~~~~~~~~~~~

Separates positive and negative contributions with different weights.

.. code-block:: python

    def lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
        return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.alpha_1_beta_0', **kwargs)

LRP with SIGN Input Layer Rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The novel contribution of SignXAI, combining LRP with sign information.

.. code-block:: python

    def lrpsign_z(model_no_softmax, x, **kwargs):
        return lrp_z(model_no_softmax, x, input_layer_rule='SIGN', **kwargs)

LRP Composite Rules
~~~~~~~~~~~~~~~~~~~

Applies different LRP rules to different layers of the network.

.. code-block:: python

    def lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
        return calculate_explanation_innvestigate(model_no_softmax, x, method='lrp.sequential_composite_a', **kwargs)

Customizing Input Layer Rules
-----------------------------

One of the key features of iNNvestigate is the ability to use different rules for the input layer. SignXAI leverages this to implement the SIGN method.

The available input layer rules are:

1. **Z-Rule** (default) - Basic propagation rule
2. **SIGN** - The novel SIGN method from SignXAI
3. **Bounded** - Uses bounded input range
4. **WSquare** - Uses squared weights
5. **Flat** - Equal distribution

Example usage:

.. code-block:: python

    # Use LRP-Epsilon with SIGN input layer rule using dynamic parsing
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name='lrpsign_epsilon_0_1')
    
    # Or explicitly:
    explanation = lrp_epsilon_0_1(model, input_tensor, input_layer_rule='SIGN')

Implementation of Gradient Methods
----------------------------------

While LRP methods use iNNvestigate, gradient-based methods are implemented directly in SignXAI:

Vanilla Gradient
~~~~~~~~~~~~~~~~

.. code-block:: python

    def gradient(model_no_softmax, x, **kwargs):
        return calculate_explanation_innvestigate(model_no_softmax, x, method='gradient', **kwargs)

Gradient x Input
~~~~~~~~~~~~~~~~

.. code-block:: python

    def gradient_x_input(model_no_softmax, x, **kwargs):
        g = gradient(model_no_softmax, x, **kwargs)
        return g * x

SIGN Methods
~~~~~~~~~~~~

The SIGN methods apply sign thresholding to the input:

.. code-block:: python

    def gradient_x_sign(model_no_softmax, x, **kwargs):
        g = gradient(model_no_softmax, x, **kwargs)
        s = np.nan_to_num(x / np.abs(x), nan=1.0)
        return g * s

With threshold parameter mu:

.. code-block:: python

    def gradient_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
        if batchmode:
            # Batch implementation
            G = []
            S = []
            for xi in x:
                G.append(gradient(model_no_softmax, xi, **kwargs))
                S.append(calculate_sign_mu(xi, mu, **kwargs))
            return np.array(G) * np.array(S)
        else:
            # Single input implementation
            return gradient(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)

Grad-CAM Implementation
-----------------------

Grad-CAM is implemented directly in SignXAI without using iNNvestigate:

.. code-block:: python

    def calculate_grad_cam_relevancemap(x, model, last_conv_layer_name=None, neuron_selection=None, resize=True, **kwargs):
        # Implementation that follows the standard Grad-CAM algorithm
        # 1. Get the last convolutional layer
        # 2. Compute gradients of target class output with respect to features
        # 3. Pool gradients to get importance weights
        # 4. Weight the feature maps and apply ReLU
        # 5. Resize to input dimensions if needed
        
        # Returns a heatmap highlighting important regions

Removing Softmax for Explainability
-----------------------------------

Proper explainability often requires working with raw logits rather than softmax probabilities. SignXAI implements a utility to remove softmax:

.. code-block:: python

    def remove_softmax(model):
        """Remove softmax activation from model.
        
        Args:
            model: TensorFlow model
            
        Returns:
            Model with softmax removed (outputs raw logits)
        """
        # Create a copy of the model
        model_copy = tf.keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        
        # Check if last layer has softmax activation
        if hasattr(model_copy.layers[-1], 'activation'):
            # Replace with linear activation
            model_copy.layers[-1].activation = tf.keras.activations.linear
            
        return model_copy

Usage Example
-------------

The following example demonstrates how to use SignXAI's TensorFlow implementation with iNNvestigate for generating LRP explanations:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications.vgg16 import VGG16
    from signxai.api import explain
    from signxai.utils.utils import load_image, normalize_heatmap, download_image

    # Load model
    model = VGG16(weights='imagenet')

    # Remove softmax (required for proper explanations)
    model.layers[-1].activation = None

    # Load example image
    path = 'example.jpg'
    download_image(path)
    img, x = load_image(path)

    # Calculate relevance maps using different LRP methods with dynamic parsing
    R1 = explain(model, x, method_name='lrp_z')  # Basic LRP-Z
    R2 = explain(model, x, method_name='lrpsign_z')  # LRP-Z with SIGN
    R3 = explain(model, x, method_name='lrp_epsilon_0_1')  # LRP-Epsilon
    R4 = explain(model, x, method_name='lrpsign_epsilon_0_1')  # LRP-Epsilon with SIGN

    # Visualize relevance maps
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(18, 12))
    axs[0][0].imshow(img)
    axs[1][0].imshow(img)
    axs[0][1].matshow(normalize_heatmap(R1), cmap='seismic', clim=(-1, 1))
    axs[0][2].matshow(normalize_heatmap(R2), cmap='seismic', clim=(-1, 1))
    axs[1][1].matshow(normalize_heatmap(R3), cmap='seismic', clim=(-1, 1))
    axs[1][2].matshow(normalize_heatmap(R4), cmap='seismic', clim=(-1, 1))

    plt.show()

Advanced iNNvestigate Configuration
-----------------------------------

For advanced users, SignXAI exposes the ability to directly configure the iNNvestigate analyzer:

.. code-block:: python

    from signxai.api import explain
    
    # Configure custom LRP parameters using dynamic parsing
    custom_lrp = explain(
        model, 
        input_tensor,
        method_name='lrpsign_sequential_composite_a'  # SIGN with sequential composite
    )
    
    # For more advanced customization, use the raw interface:
    from signxai.utils.utils import calculate_explanation_innvestigate
    custom_lrp_advanced = calculate_explanation_innvestigate(
        model, 
        input_tensor,
        method='lrp.sequential_composite_a',
        input_layer_rule='SIGN',
        neuron_selection=predicted_class
    )

This flexibility allows for very fine-grained control over the explanation process.

Extending with New Methods
--------------------------

To add new methods based on iNNvestigate, you can create a wrapper function in ``signxai.tf_signxai.methods.wrappers.py``:

.. code-block:: python

    def my_custom_method(model_no_softmax, x, **kwargs):
        # Custom pre-processing
        # ...
        
        # Use the new explain API or raw iNNvestigate
        from signxai.api import explain
        result = explain(model_no_softmax, x, method_name='custom_method_name')
        
        # Or for advanced cases, use the raw interface:
        from signxai.utils.utils import calculate_explanation_innvestigate
        result = calculate_explanation_innvestigate(
            model_no_softmax, 
            x, 
            method='...',  # Use existing iNNvestigate method
            **kwargs
        )
        
        # Custom post-processing
        # ...
        
        return result

Performance Considerations
--------------------------

When using iNNvestigate through SignXAI, consider these performance tips:

1. **Model Size** - LRP methods can be memory-intensive for large models
2. **Input Resolution** - Higher resolution inputs require more computation
3. **Batch Processing** - Use batched inputs for multiple samples
4. **GPU Acceleration** - Ensure TensorFlow is configured for GPU
5. **Memory Management** - For large models, consider reducing batch size
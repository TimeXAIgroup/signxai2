Unified API
===========

The SignXAI unified API provides a framework-agnostic interface for explainable AI methods.

.. currentmodule:: signxai

Overview
--------

The unified API automatically detects whether you're using TensorFlow or PyTorch models and routes your requests to the appropriate backend implementation.

Main Interface
--------------

Main Function
~~~~~~~~~~~~~

**explain(model, input_data, method, \*\*kwargs)**

    Get explanations for a model's predictions using any supported XAI method.

    :param model: The neural network model (TensorFlow or PyTorch)
    :param input_data: Input data for which to generate explanations  
    :param method: The explanation method to use
    :param kwargs: Additional method-specific parameters
    :return: Explanation array with same shape as input

Framework Detection
-------------------

**get_framework(model)**

    Automatically detect the framework of a given model.

    :param model: The model to check
    :return: 'tensorflow' or 'pytorch'

Common Parameters
-----------------

All methods support these common parameters:

- **neuron_selection**: Target neuron/class for explanation
- **batchsize**: Batch size for processing (PyTorch only)
- **postprocess**: Post-processing function to apply

Method-Specific Parameters
--------------------------

Different methods accept additional parameters:

**Gradient-based methods:**

- **postprocess**: 'abs', 'square', or custom function
- **mu**: SIGN threshold parameter (for SIGN variants)

**LRP methods:**

- **epsilon**: Stabilization parameter for LRP-ε
- **alpha/beta**: Parameters for LRP-α/β rule
- **layer_rule**: Custom rules for specific layers

**Integrated Gradients:**

- **reference**: Reference/baseline input
- **steps**: Number of integration steps

**SmoothGrad:**

- **noise**: Noise level for sampling
- **samples**: Number of samples

**Grad-CAM:**

- **last_conv**: Name/index of last convolutional layer

Usage Examples
--------------

Basic usage with automatic framework detection::

    import signxai
    
    # TensorFlow model
    explanation_tf = signxai.explain(
        tf_model, 
        input_data, 
        method='gradient'
    )
    
    # PyTorch model  
    explanation_pt = signxai.explain(
        torch_model,
        input_data,
        method='gradient'
    )

Using method-specific parameters::

    # LRP with epsilon
    explanation = signxai.explain(
        model,
        input_data, 
        method='lrp_epsilon',
        epsilon=0.01
    )
    
    # SIGN variant with custom mu
    explanation = signxai.explain(
        model,
        input_data,
        method='gradient_x_sign_mu',
        mu=0.5
    )

Error Handling
--------------

The unified API provides consistent error messages across frameworks:

- **UnsupportedMethodError**: Method not available for the detected framework
- **InvalidParameterError**: Invalid parameter for the chosen method
- **FrameworkDetectionError**: Unable to determine model framework
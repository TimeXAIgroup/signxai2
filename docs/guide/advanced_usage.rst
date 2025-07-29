==============
Advanced Usage
==============

This guide covers advanced usage patterns and features of SignXAI.

.. contents:: Contents
   :local:
   :depth: 2

Advanced LRP Configuration
--------------------------

Layer-wise Relevance Propagation (LRP) is highly configurable. Here's how to use advanced settings:

TensorFlow Advanced LRP
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from tensorflow.keras.applications.vgg16 import VGG16
    from signxai.utils.utils import calculate_explanation_innvestigate
    
    # Load model
    model = VGG16(weights='imagenet')
    model.layers[-1].activation = None
    
    # Load and preprocess input
    # ...
    
    # Basic LRP-Z
    lrp_z = calculate_explanation_innvestigate(
        model, 
        input_tensor, 
        method='lrp.z'
    )
    
    # LRP-Z with SIGN input layer rule
    lrp_sign = calculate_explanation_innvestigate(
        model, 
        input_tensor, 
        method='lrp.z',
        input_layer_rule='SIGN'
    )
    
    # LRP-Epsilon with custom epsilon
    lrp_eps = calculate_explanation_innvestigate(
        model, 
        input_tensor, 
        method='lrp.epsilon',
        epsilon=0.01
    )
    
    # LRP with bounded input range
    lrp_bounded = calculate_explanation_innvestigate(
        model, 
        input_tensor, 
        method='lrp.z',
        input_layer_rule='Bounded',
        low=-123.68,  # ImageNet mean values
        high=151.061
    )
    
    # LRP-AlphaBeta with custom parameters
    lrp_alphabeta = calculate_explanation_innvestigate(
        model, 
        input_tensor, 
        method='lrp.alpha_beta',
        alpha=2,
        beta=1
    )
    
    # LRP Sequential Composite - different rules for different layers
    lrp_composite = calculate_explanation_innvestigate(
        model, 
        input_tensor, 
        method='lrp.sequential_composite_a'
    )

PyTorch Advanced LRP
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torchvision.models as models
    from signxai.torch_signxai.methods.zennit_impl import (
        LRPAnalyzer,
        AdvancedLRPAnalyzer,
        LRPSequential
    )
    from zennit.composites import EpsilonPlusFlat, LayerMapComposite
    from zennit.rules import Epsilon, ZPlus, ZBox, Gamma
    from zennit.types import Convolution, Linear
    
    # Load model
    model = models.vgg16(pretrained=True)
    model.eval()
    
    # Load and preprocess input
    # ...
    
    # Basic LRP-Epsilon
    analyzer_epsilon = LRPAnalyzer(model, rule="epsilon", epsilon=0.1)
    lrp_epsilon = analyzer_epsilon.analyze(input_tensor)
    
    # LRP Alpha-Beta
    analyzer_alphabeta = LRPAnalyzer(model, rule="alphabeta")  # Default alpha=1, beta=0
    lrp_alphabeta = analyzer_alphabeta.analyze(input_tensor)
    
    # Advanced: LRP with custom rules
    analyzer_advanced = AdvancedLRPAnalyzer(
        model, 
        rule_type="zbox",
        low=-123.68,
        high=151.061
    )
    lrp_advanced = analyzer_advanced.analyze(input_tensor)
    
    # LRP Composite with layer-specific rules
    layer_map = {
        Convolution: ZPlus(),         # Use ZPlus for convolutional layers
        Linear: Epsilon(epsilon=0.1)  # Use Epsilon for linear layers
    }
    
    # Create a custom composite
    custom_composite = LayerMapComposite(layer_map)
    
    # Use the custom composite
    analyzer_custom = AdvancedLRPAnalyzer(model, rule_type="custom", composite=custom_composite)
    lrp_custom = analyzer_custom.analyze(input_tensor)
    
    # LRP Sequential (layer-specialized composite)
    analyzer_sequential = LRPSequential(
        model,
        first_layer_rule="zbox",
        middle_layer_rule="alphabeta",
        last_layer_rule="epsilon"
    )
    lrp_sequential = analyzer_sequential.analyze(input_tensor)

Custom Target Class Selection
-----------------------------

By default, explanations target the class with the highest predicted probability, but you can specify any class:

TensorFlow Custom Target
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get predictions
    preds = model.predict(x)
    
    # Get top 5 predicted classes
    top_classes = np.argsort(preds[0])[-5:][::-1]
    
    # Generate explanations for each class
    class_explanations = {}
    for idx in top_classes:
        class_explanations[idx] = calculate_relevancemap(
            'input_t_gradient', 
            x, 
            model, 
            neuron_selection=idx  # Specific class
        )
    
    # Visualize explanations for different classes
    fig, axs = plt.subplots(1, len(top_classes) + 1, figsize=(15, 4))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    # Class-specific explanations
    for i, idx in enumerate(top_classes):
        class_name = decode_predictions(preds, top=5)[0][i][1]
        axs[i+1].imshow(normalize_heatmap(class_explanations[idx][0]), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(f'{class_name}')
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

PyTorch Custom Target
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Get predictions
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get top 5 predicted classes
    _, top_indices = torch.topk(output, 5, dim=1)
    top_classes = top_indices[0].tolist()
    
    # Generate explanations for each class
    class_explanations = {}
    for idx in top_classes:
        class_explanations[idx] = calculate_relevancemap(
            model_no_softmax, 
            input_tensor, 
            method="input_t_gradient",
            target_class=idx  # Specific class
        )
    
    # Visualize explanations for different classes
    fig, axs = plt.subplots(1, len(top_classes) + 1, figsize=(15, 4))
    
    # Original image
    axs[0].imshow(img_np)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    # Class-specific explanations
    for i, idx in enumerate(top_classes):
        explanation = class_explanations[idx][0].sum(axis=0)
        axs[i+1].imshow(normalize_relevance_map(explanation), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(f'Class {idx}')
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

Working with Time Series Data
-----------------------------

SignXAI supports time series data such as ECG signals:

TensorFlow Time Series
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from signxai.tf_signxai import calculate_relevancemap
    
    # Load a pre-trained ECG model
    model = tf.keras.models.load_model('ecg_model.h5')
    model.layers[-1].activation = None
    
    # Load an ECG signal
    ecg_signal = np.load('ecg_sample.npy')
    ecg_input = ecg_signal.reshape(1, -1, 1)  # Add batch and channel dimensions
    
    # Calculate explanation
    explanation = calculate_relevancemap(
        'input_t_gradient', 
        ecg_input, 
        model
    )
    
    # Plot original signal and explanation
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(ecg_signal)
    plt.title('Original ECG Signal')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(explanation[0, :, 0])
    plt.title('Explanation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # For 1D time series, GradCAM requires a specific implementation
    gradcam_explanation = calculate_relevancemap(
        'grad_cam_timeseries', 
        ecg_input, 
        model,
        last_conv_layer_name='conv1d_3'  # Specify convolutional layer
    )
    
    # Plot GradCAM explanation
    plt.figure(figsize=(12, 3))
    plt.plot(gradcam_explanation[0, :, 0])
    plt.title('GradCAM Explanation for Time Series')
    plt.grid(True)
    plt.show()

PyTorch Time Series
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    from signxai.torch_signxai import calculate_relevancemap
    from signxai.torch_signxai.utils import remove_softmax
    
    # Define a simple 1D CNN for time series
    class ECG_CNN(nn.Module):
        def __init__(self):
            super(ECG_CNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=5)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=5)
            self.pool = nn.MaxPool1d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32*123, 64)
            self.fc2 = nn.Linear(64, 5)  # 5 classes
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Load model and weights
    model = ECG_CNN()
    model.load_state_dict(torch.load('ecg_model.pt'))
    model.eval()
    
    # Remove softmax
    model_no_softmax = remove_softmax(model)
    
    # Load an ECG signal
    ecg_signal = np.load('ecg_sample.npy')
    
    # Convert to PyTorch tensor with shape [batch, channels, time]
    ecg_input = torch.tensor(ecg_signal, dtype=torch.float32).reshape(1, 1, -1)
    
    # Calculate explanation
    explanation = calculate_relevancemap(
        model_no_softmax, 
        ecg_input, 
        method="input_t_gradient"
    )
    
    # Plot original signal and explanation
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(ecg_signal)
    plt.title('Original ECG Signal')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(explanation[0, 0, :])
    plt.title('Explanation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

Custom SIGN Methods
-------------------

The SIGN method is a key innovation in SignXAI. Here's how to use it with custom parameters:

TensorFlow SIGN
~~~~~~~~~~~~~~~

.. code-block:: python

    from signxai.tf_signxai.methods.signed import calculate_sign_mu
    
    # Standard SIGN with mu=0
    sign = calculate_sign_mu(input_tensor, mu=0)
    
    # Custom SIGN methods with different mu values
    sign_pos = calculate_sign_mu(input_tensor, mu=0.5)     # Focus on positive values
    sign_neg = calculate_sign_mu(input_tensor, mu=-0.5)    # Focus on negative values
    
    # Apply SIGN with gradient
    gradient = calculate_relevancemap('gradient', input_tensor, model)
    
    # Manually apply SIGN
    gradient_sign = gradient * sign
    gradient_sign_pos = gradient * sign_pos
    gradient_sign_neg = gradient * sign_neg
    
    # Or use built-in methods
    gradient_sign_direct = calculate_relevancemap('gradient_x_sign', input_tensor, model)
    gradient_sign_mu = calculate_relevancemap('gradient_x_sign_mu', input_tensor, model, mu=0.5)

PyTorch SIGN
~~~~~~~~~~~~

.. code-block:: python

    from signxai.torch_signxai.methods.signed import calculate_sign_mu
    
    # Standard SIGN with mu=0
    sign = calculate_sign_mu(input_tensor, mu=0)
    
    # Custom SIGN methods with different mu values
    sign_pos = calculate_sign_mu(input_tensor, mu=0.5)     # Focus on positive values
    sign_neg = calculate_sign_mu(input_tensor, mu=-0.5)    # Focus on negative values
    
    # Apply SIGN with gradient
    gradient = calculate_relevancemap(model_no_softmax, input_tensor, method="gradients")
    
    # Convert to tensor if needed
    if isinstance(gradient, np.ndarray):
        gradient = torch.tensor(gradient)
    
    # Manually apply SIGN
    gradient_sign = gradient * sign
    gradient_sign_pos = gradient * sign_pos
    gradient_sign_neg = gradient * sign_neg

Integrating with Other Libraries
--------------------------------

SignXAI can be used alongside other explainability libraries:

SHAP Integration
~~~~~~~~~~~~~~~~

.. code-block:: python

    import shap
    
    # TensorFlow
    # Create a SHAP explainer
    explainer = shap.GradientExplainer(model, background_dataset)
    shap_values = explainer.shap_values(x)
    
    # Calculate SignXAI explanation
    signxai_explanation = calculate_relevancemap('input_t_gradient', x, model)
    
    # Compare explanations
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    shap.image_plot(shap_values, x, show=False)
    plt.title('SHAP Explanation')
    
    plt.subplot(1, 2, 2)
    plt.imshow(normalize_heatmap(signxai_explanation[0]), cmap='seismic', clim=(-1, 1))
    plt.title('SignXAI Explanation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

Captum Integration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from captum.attr import IntegratedGradients
    
    # PyTorch with Captum
    ig = IntegratedGradients(model)
    captum_attr = ig.attribute(input_tensor, target=predicted_idx)
    
    # SignXAI explanation
    signxai_attr = calculate_relevancemap(
        model, 
        input_tensor, 
        method="integrated_gradients"
    )
    
    # Compare explanations
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(captum_attr.sum(dim=1)[0].detach().cpu().numpy(), cmap='seismic')
    plt.title('Captum Explanation')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(signxai_attr[0].sum(axis=0), cmap='seismic')
    plt.title('SignXAI Explanation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

Advanced Visualization Techniques
---------------------------------

SignXAI provides advanced visualization options:

Overlay with Transparency
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from signxai.common.visualization import (
        normalize_relevance_map,
        relevance_to_heatmap,
        overlay_heatmap
    )
    
    # Generate explanation
    explanation = calculate_relevancemap(
        model, 
        input_tensor, 
        method="lrp_z"
    )
    
    # Normalize explanation
    normalized = normalize_relevance_map(explanation[0].sum(axis=0))
    
    # Create heatmap
    heatmap = relevance_to_heatmap(normalized, cmap='seismic')
    
    # Create overlays with different transparency levels
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    for i, alpha in enumerate([0.3, 0.5, 0.7]):
        overlaid = overlay_heatmap(original_image, heatmap, alpha=alpha)
        axs[i+1].imshow(overlaid)
        axs[i+1].set_title(f'Overlay (alpha={alpha})')
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

Positive and Negative Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Separate positive and negative contributions
    explanation = calculate_relevancemap(
        model, 
        input_tensor, 
        method="lrp_epsilon", 
        epsilon=0.1
    )
    
    # Extract positive and negative values
    explanation_flat = explanation[0].sum(axis=0)
    pos_explanation = np.maximum(0, explanation_flat)
    neg_explanation = np.minimum(0, explanation_flat)
    
    # Normalize separately
    pos_norm = pos_explanation / np.max(pos_explanation) if np.max(pos_explanation) > 0 else pos_explanation
    neg_norm = neg_explanation / np.min(neg_explanation) if np.min(neg_explanation) < 0 else neg_explanation
    
    # Visualize
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(normalize_relevance_map(explanation_flat), cmap='seismic', clim=(-1, 1))
    axs[1].set_title('Combined')
    axs[1].axis('off')
    
    axs[2].imshow(pos_norm, cmap='Reds')
    axs[2].set_title('Positive Contributions')
    axs[2].axis('off')
    
    axs[3].imshow(-neg_norm, cmap='Blues')
    axs[3].set_title('Negative Contributions')
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()

Performance Optimization
------------------------

For large models or datasets, consider these performance optimizations:

TensorFlow Performance
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tensorflow as tf
    
    # Enable mixed precision (for TensorFlow 2.x with GPU)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Use memory-efficient computation
    @tf.function
    def compute_gradients(model, inputs, target_class):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = model(inputs)
            loss = predictions[:, target_class]
        return tape.gradient(loss, inputs)
    
    # Batch processing
    def process_large_dataset(model, dataset, batch_size=32):
        all_explanations = []
        
        for batch in dataset.batch(batch_size):
            batch_explanations = calculate_relevancemap('input_t_gradient', batch, model)
            all_explanations.append(batch_explanations)
        
        return np.concatenate(all_explanations, axis=0)

PyTorch Performance
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    
    # Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    # Memory-efficient computation
    def efficient_gradient(model, inputs, target_class):
        inputs = inputs.to('cuda')
        model = model.to('cuda')
        
        inputs.requires_grad = True
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            
            # One-hot encoding
            one_hot = torch.zeros_like(outputs)
            one_hot[:, target_class] = 1
            
        # Compute gradients
        model.zero_grad()
        outputs.backward(gradient=one_hot)
        
        return inputs.grad.detach().cpu()
    
    # Batch processing with DataLoader
    def process_large_dataset(model, dataset_loader):
        all_explanations = []
        
        for batch in dataset_loader:
            inputs, _ = batch
            explanations = calculate_relevancemap(model, inputs, method="gradients")
            all_explanations.append(explanations)
        
        return np.concatenate(all_explanations, axis=0)

Next Steps
----------

After exploring these advanced techniques, you may want to:

1. Read about the specific implementation details in :doc:`pytorch` and :doc:`tensorflow`
2. Learn how to convert models between frameworks in :doc:`framework_interop`
3. Explore the complete API reference in :doc:`/api/common`, :doc:`/api/pytorch`, and :doc:`/api/tensorflow`
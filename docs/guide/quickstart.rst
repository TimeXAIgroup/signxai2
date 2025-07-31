==========
Quickstart
==========

This quickstart guide will help you get up and running with SignXAI2 quickly for both PyTorch and TensorFlow models.

.. contents:: Contents
   :local:
   :depth: 2

Installation
------------

SignXAI2 requires you to explicitly choose which deep learning framework(s) to install:

.. code-block:: bash

    # For TensorFlow users:
    pip install signxai2[tensorflow]
    
    # For PyTorch users:
    pip install signxai2[pytorch]
    
    # For both frameworks:
    pip install signxai2[all]
    
    # Note: Requires Python 3.9 or 3.10
    # Installing pip install signxai2 alone is NOT supported

TensorFlow Quickstart
---------------------

Here's a complete example using TensorFlow:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from signxai.tf_signxai.methods.wrappers import calculate_relevancemap
    
    # Step 1: Load a pre-trained model
    model = VGG16(weights='imagenet')
    
    # Step 2: Remove softmax (critical for explanations)
    model.layers[-1].activation = None
    
    # Step 3: Load and preprocess an image
    img_path = 'path/to/image.jpg'
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Step 4: Get prediction
    preds = model.predict(x)
    top_pred_idx = np.argmax(preds[0])
    print(f"Predicted class: {decode_predictions(preds, top=1)[0][0][1]}")
    
    # Step 5: Calculate explanation with input × gradient method
    explanation = calculate_relevancemap('input_t_gradient', x, model, neuron_selection=top_pred_idx)
    
    # Step 6: Normalize and visualize
    # Sum over channels to create 2D heatmap
    heatmap = explanation[0].sum(axis=-1)
    abs_max = np.max(np.abs(heatmap))
    if abs_max > 0:
        normalized = heatmap / abs_max
    else:
        normalized = heatmap
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(normalized, cmap='seismic', clim=(-1, 1))
    plt.title('Explanation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

PyTorch Quickstart
------------------

Here's a complete example using PyTorch:

.. code-block:: python

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms
    from signxai.torch_signxai import calculate_relevancemap
    from signxai.torch_signxai.utils import remove_softmax
    
    # Step 1: Load a pre-trained model
    model = models.vgg16(pretrained=True)
    model.eval()
    
    # Step 2: Remove softmax
    model_no_softmax = remove_softmax(model)
    
    # Step 3: Load and preprocess an image
    img_path = 'path/to/image.jpg'
    img = Image.open(img_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    
    # Step 4: Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get the most likely class
    _, predicted_idx = torch.max(output, 1)
    
    # Step 5: Calculate explanation with Gradient x Input method
    explanation = calculate_relevancemap(
        model_no_softmax, 
        input_tensor, 
        method="input_t_gradient",
        target_class=predicted_idx.item()
    )
    
    # Step 6: Normalize and visualize
    # Convert back to numpy for visualization
    explanation_np = explanation.detach().cpu().numpy() if hasattr(explanation, 'detach') else explanation
    # Sum over channels to create 2D heatmap
    heatmap = explanation_np.sum(axis=0)
    abs_max = np.max(np.abs(heatmap))
    if abs_max > 0:
        normalized = heatmap / abs_max
    else:
        normalized = heatmap
    
    # Convert the original image for display
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(normalized, cmap='seismic', clim=(-1, 1))
    plt.title('Explanation')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

Framework-Agnostic Approach
---------------------------

You can also use the framework-agnostic API:

.. code-block:: python

    from signxai import explain, list_methods
    
    # List available methods
    print(f"Available methods: {list_methods()}")
    
    # Will work with either PyTorch or TensorFlow model
    explanation = explain(model, input_data, method="gradient")
    
    # SignXAI will automatically detect the framework

Multiple Explanation Methods
----------------------------

Compare different explanation methods for the same input:

.. code-block:: python

    # For PyTorch
    from signxai.torch_signxai import calculate_relevancemap
    
    methods = ['gradient', 'input_t_gradient', 'integrated_gradients', 'smoothgrad', 'lrp_z']
    explanations = []
    
    for method in methods:
        explanation = calculate_relevancemap(
            model=model_no_softmax,
            input_tensor=input_tensor,
            method=method,
            target_class=predicted_idx.item()
        )
        # Convert to numpy for visualization
        if hasattr(explanation, 'detach'):
            explanation = explanation.detach().cpu().numpy()
        explanations.append(explanation)
    
    # Visualize all methods
    fig, axs = plt.subplots(1, len(methods) + 1, figsize=(15, 4))
    axs[0].imshow(img_np)
    axs[0].set_title('Original')
    axs[0].axis('off')
    
    for i, (method, expl) in enumerate(zip(methods, explanations)):
        # Sum over channels and normalize
        heatmap = expl.sum(axis=0)  # PyTorch format: (C, H, W)
        abs_max = np.max(np.abs(heatmap))
        if abs_max > 0:
            normalized = heatmap / abs_max
        else:
            normalized = heatmap
        axs[i+1].imshow(normalized, cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(method)
        axs[i+1].axis('off']
    
    plt.tight_layout()
    plt.show()

LRP Variants
------------

Layer-wise Relevance Propagation (LRP) has several variants:

.. code-block:: python

    # For PyTorch
    lrp_methods = [
        'lrp_z',                  # Basic LRP-Z
        'lrpsign_z',              # LRP-Z with SIGN
        'lrp_epsilon_0_1',        # LRP with epsilon=0.1
        'lrp_alpha_1_beta_0'      # LRP with alpha=1, beta=0
    ]
    
    lrp_explanations = []
    for method in lrp_methods:
        explanation = calculate_relevancemap(
            model=model_no_softmax,
            input_tensor=input_tensor,
            method=method,
            target_class=predicted_idx.item()
        )
        if hasattr(explanation, 'detach'):
            explanation = explanation.detach().cpu().numpy()
        lrp_explanations.append(explanation)
    
    # Visualize LRP variants
    fig, axs = plt.subplots(1, len(lrp_methods), figsize=(12, 3))
    for i, (method, expl) in enumerate(zip(lrp_methods, lrp_explanations)):
        heatmap = expl.sum(axis=0)
        abs_max = np.max(np.abs(heatmap))
        if abs_max > 0:
            normalized = heatmap / abs_max
        else:
            normalized = heatmap
        axs[i].imshow(normalized, cmap='seismic', clim=(-1, 1))
        axs[i].set_title(method)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

Working with Different Method Parameters
----------------------------------------

Many methods support additional parameters:

.. code-block:: python

    # For PyTorch
    # LRP with different epsilon values
    epsilons = [0.01, 0.1, 1.0]
    for eps in epsilons:
        explanation = calculate_relevancemap(
            model=model_no_softmax,
            input_tensor=input_tensor,
            method='lrp_epsilon',
            target_class=predicted_idx.item(),
            epsilon=eps
        )
        # Visualize...
    
    # SmoothGrad with custom parameters
    explanation = calculate_relevancemap(
        model=model_no_softmax,
        input_tensor=input_tensor,
        method='smoothgrad',
        target_class=predicted_idx.item(),
        num_samples=50,    # Number of samples
        noise_level=0.1    # Noise level
    )
    
    # Integrated Gradients with custom steps
    explanation = calculate_relevancemap(
        model=model_no_softmax,
        input_tensor=input_tensor,
        method='integrated_gradients',
        target_class=predicted_idx.item(),
        steps=100  # Integration steps
    )
    
    # Grad-CAM with specific layer
    explanation = calculate_relevancemap(
        model=model_no_softmax,
        input_tensor=input_tensor,
        method='grad_cam',
        target_class=predicted_idx.item(),
        target_layer=model.features[28]  # Last conv layer for VGG16
    )

Next Steps
----------

After this quickstart, you can:

1. Explore different explanation methods in the :doc:`../api/methods_list`
2. Learn about framework-specific features in :doc:`pytorch` and :doc:`tensorflow`
3. Check out complete tutorials in the :doc:`/tutorials/image_classification` and :doc:`/tutorials/time_series`
4. Understand the framework interoperability options in :doc:`framework_interop`
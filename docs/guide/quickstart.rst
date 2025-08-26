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
    from signxai.api import explain
    from signxai.utils.utils import remove_softmax
    
    # Step 1: Load a pre-trained model
    model = VGG16(weights='imagenet')
    
    # Step 2: Remove softmax (critical for explanations)
    model = remove_softmax(model)
    
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
    
    # Step 5: Calculate explanation with advanced gradient method
    # This demonstrates dynamic method parsing with multiple operations:
    # gradient × input × sign with mu parameter of -0.5
    explanation = explain(model, x, method_name='gradient_x_input_x_sign_mu_neg_0_5', target_class=top_pred_idx)
    
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
    from signxai.api import explain
    from signxai.torch_signxai.utils import remove_softmax

    # Step 1: Load a pre-trained model
    model = models.vgg16(pretrained=True)
    model.eval()

    # Step 2: Remove softmax
    model_no_softmax = remove_softmax(model)

    # Step 3: Load and preprocess an image
    img_path = 'path/to/image.jpg' # Please replace with actual path
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

    # Step 5: Calculate explanation with advanced gradient method
    explanation = explain(
        model_no_softmax,
        input_tensor,
        method_name="gradient_x_input_x_sign_mu_neg_0_5",
        target_class=predicted_idx.item()
    )

    # Step 6: Normalize and visualize
    # Convert to numpy for visualization
    explanation_np = explanation.detach().cpu().numpy() if hasattr(explanation, 'detach') else explanation
    # Sum over channels to create 2D heatmap
    if explanation_np.ndim == 4:
        explanation_np = explanation_np[0]
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

    from signxai.api import explain
    
    # Will work with either PyTorch or TensorFlow model
    # Using dynamic method parsing - parameters embedded in method names
    
    # Simple gradient method
    explanation = explain(model, input_data, method_name="gradient")
    
    # Advanced method with parameters
    explanation = explain(model, input_data, method_name="gradient_x_input_x_sign_mu_neg_0_5")
    
    # SignXAI will automatically detect the framework

Multiple Explanation Methods
----------------------------

Compare different explanation methods using dynamic method parsing:

.. code-block:: python

    from signxai.api import explain
    
    # Dynamic method names with embedded parameters
    methods = [
        'gradient',                                    # Basic gradient
        'gradient_x_input',                           # Gradient × Input
        'gradient_x_input_x_sign_mu_neg_0_5',        # Advanced combination
        'integrated_gradients_steps_100',             # Integrated Gradients (100 steps)
        'smoothgrad_noise_0_3_samples_50',           # SmoothGrad with parameters
        'lrp_epsilon_0_25'                           # LRP with epsilon=0.25
    ]
    explanations = []
    
    for method_name in methods:
        explanation = explain(
            model=model_no_softmax,
            x=input_tensor,
            method_name=method_name,
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
    
    for i, (method_name, expl) in enumerate(zip(methods, explanations)):
        # Sum over channels and normalize
        heatmap = expl.sum(axis=0)  # PyTorch format: (C, H, W)
        abs_max = np.max(np.abs(heatmap))
        if abs_max > 0:
            normalized = heatmap / abs_max
        else:
            normalized = heatmap
        axs[i+1].imshow(normalized, cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(method_name)
        axs[i+1].axis('off']
    
    plt.tight_layout()
    plt.show()

LRP Variants
------------

Layer-wise Relevance Propagation (LRP) variants using dynamic method parsing:

.. code-block:: python

    from signxai.api import explain
    
    # LRP methods with parameters embedded in names
    lrp_methods = [
        'lrp_z',                          # Basic LRP-Z
        'lrp_z_x_sign',                   # LRP-Z with SIGN
        'lrp_epsilon_0_1',                # LRP with epsilon=0.1
        'lrp_epsilon_0_25',               # LRP with epsilon=0.25
        'lrp_alpha_2_beta_1',             # LRP with alpha=2, beta=1
        'lrp_gamma_0_25'                  # LRP with gamma=0.25
    ]
    
    lrp_explanations = []
    for method_name in lrp_methods:
        explanation = explain(
            model=model_no_softmax,
            x=input_tensor,
            method_name=method_name,
            target_class=predicted_idx.item()
        )
        if hasattr(explanation, 'detach'):
            explanation = explanation.detach().cpu().numpy()
        lrp_explanations.append(explanation)
    
    # Visualize LRP variants
    fig, axs = plt.subplots(1, len(lrp_methods), figsize=(12, 3))
    for i, (method_name, expl) in enumerate(zip(lrp_methods, lrp_explanations)):
        heatmap = expl.sum(axis=0)
        abs_max = np.max(np.abs(heatmap))
        if abs_max > 0:
            normalized = heatmap / abs_max
        else:
            normalized = heatmap
        axs[i].imshow(normalized, cmap='seismic', clim=(-1, 1))
        axs[i].set_title(method_name)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

Working with Dynamic Method Parameters
---------------------------------------

Parameters are embedded directly in method names:

.. code-block:: python

    from signxai.api import explain
    
    # LRP with different epsilon values (embedded in method name)
    epsilon_methods = [
        'lrp_epsilon_0_01',    # epsilon=0.01
        'lrp_epsilon_0_1',     # epsilon=0.1
        'lrp_epsilon_1'        # epsilon=1.0
    ]
    
    for method_name in epsilon_methods:
        explanation = explain(
            model=model_no_softmax,
            x=input_tensor,
            method_name=method_name,
            target_class=predicted_idx.item()
        )
        # Visualize...
    
    # SmoothGrad with custom parameters (embedded in name)
    explanation = explain(
        model=model_no_softmax,
        x=input_tensor,
        method_name='smoothgrad_noise_0_1_samples_50',  # noise=0.1, samples=50
        target_class=predicted_idx.item()
    )
    
    # Integrated Gradients with custom steps
    explanation = explain(
        model=model_no_softmax,
        x=input_tensor,
        method_name='integrated_gradients_steps_100',  # 100 integration steps
        target_class=predicted_idx.item()
    )
    
    # Complex combinations with multiple operations
    explanation = explain(
        model=model_no_softmax,
        x=input_tensor,
        method_name='gradient_x_input_x_sign_mu_neg_0_5',  # gradient × input × sign(mu=-0.5)
        target_class=predicted_idx.item()
    )

Next Steps
----------

After this quickstart, you can:

1. Explore different explanation methods in the :doc:`../api/methods_list`
2. Learn about framework-specific features in :doc:`pytorch` and :doc:`tensorflow`
3. Check out complete tutorials in the :doc:`/tutorials/image_classification` and :doc:`/tutorials/time_series`
4. Understand the framework interoperability options in :doc:`framework_interop`
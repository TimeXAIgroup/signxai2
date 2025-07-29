===========
Basic Usage
===========

This guide covers the basic usage of SignXAI with both PyTorch and TensorFlow.

.. contents:: Contents
   :local:
   :depth: 2

Common Workflow
---------------

Regardless of which framework you use, the general workflow for generating explanations is similar:

1. Load your model
2. Remove softmax activation
3. Prepare input data
4. Calculate explanations
5. Visualize and analyze results

TensorFlow Basic Usage
----------------------

Working with a TensorFlow model:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from signxai.tf_signxai.methods.wrappers import calculate_relevancemap
    from signxai.common.visualization import normalize_relevance_map
    
    # Step 1: Load a pre-trained model
    model = VGG16(weights='imagenet')
    
    # Step 2: Remove softmax - critical for proper explanations
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
    
    # Step 5: Calculate explanation with different methods
    explanations = {}
    methods = [
        'gradient',              # Vanilla gradient
        'input_t_gradient',      # Input × Gradient
        'gradient_x_sign',       # Gradient × Sign
        'integrated_gradients',  # Integrated gradients
        'smoothgrad',            # SmoothGrad
        'grad_cam',              # Grad-CAM
        'lrp_z',                 # LRP-Z
        'lrp_epsilon_0_1'        # LRP-Epsilon
    ]
    
    for method in methods:
        explanations[method] = calculate_relevancemap(
            method, 
            x, 
            model, 
            neuron_selection=top_pred_idx
        )
    
    # Step 6: Visualize explanations
    n_methods = len(methods)
    fig, axs = plt.subplots(2, (n_methods // 2) + 1, figsize=(15, 8))
    axs = axs.flatten()
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Explanation methods
    for i, method in enumerate(methods):
            # Sum over channels and normalize
        heatmap = explanations[method][0].sum(axis=-1)
        abs_max = np.max(np.abs(heatmap))
        if abs_max > 0:
            normalized = heatmap / abs_max
        else:
            normalized = heatmap
        axs[i+1].imshow(normalized, cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(method)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

PyTorch Basic Usage
-------------------

Working with a PyTorch model:

.. code-block:: python

    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.models as models
    import torchvision.transforms as transforms
    from signxai.torch_signxai import calculate_relevancemap
    from signxai.torch_signxai.utils import remove_softmax
    from signxai.common.visualization import normalize_relevance_map
    
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
    
    # Step 5: Calculate explanation with different methods
    explanations = {}
    methods = [
        "gradients",             # Vanilla gradient
        "input_t_gradient",      # Gradient × Input
        "integrated_gradients",  # Integrated gradients
        "smoothgrad",            # SmoothGrad
        "grad_cam",              # Grad-CAM
        "lrp_epsilon",           # LRP with epsilon rule
        "lrp_alphabeta"          # LRP with alpha-beta rule
    ]
    
    for method in methods:
        explanations[method] = calculate_relevancemap(
            model_no_softmax, 
            input_tensor, 
            method=method,
            target_class=predicted_idx.item()
        )
    
    # Step 6: Visualize explanations
    # Convert the original image for display
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    n_methods = len(methods)
    fig, axs = plt.subplots(2, (n_methods // 2) + 1, figsize=(15, 8))
    axs = axs.flatten()
    
    # Original image
    axs[0].imshow(img_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Explanation methods
    for i, method in enumerate(methods):
        # Handle channel dimension for PyTorch explanations
        explanation = explanations[method][0].sum(axis=0)
        axs[i+1].imshow(normalize_relevance_map(explanation), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(method)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

Visualizing Explanations
------------------------

SignXAI provides several visualization utilities:

.. code-block:: python

    from signxai.common.visualization import (
        normalize_relevance_map,
        relevance_to_heatmap, 
        overlay_heatmap
    )
    
    # Normalize explanation
    normalized = normalize_relevance_map(explanation[0].sum(axis=0))
    
    # Convert to heatmap
    heatmap = relevance_to_heatmap(normalized, cmap='seismic')
    
    # Overlay on original image
    overlaid = overlay_heatmap(original_image, heatmap, alpha=0.6)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(overlaid)
    plt.title('Explanation Overlay')
    plt.axis('off')
    plt.show()

Working with Custom Models
--------------------------

You can use SignXAI with your own custom models:

TensorFlow Custom Model
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import tensorflow as tf
    from signxai.tf_signxai import calculate_relevancemap
    
    # Define a custom model
    def create_custom_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)  # No activation (logits)
        ])
        return model
    
    # Create model
    model = create_custom_model()
    
    # Load weights if needed
    # model.load_weights('my_model_weights.h5')
    
    # Generate explanation for a custom input
    input_data = np.random.random((1, 28, 28, 1))
    explanation = calculate_relevancemap('lrp_z', input_data, model, neuron_selection=5)
    
    # Visualize
    plt.matshow(explanation[0, :, :, 0], cmap='seismic', clim=(-1, 1))
    plt.colorbar()
    plt.title('Explanation for Class 5')
    plt.show()

PyTorch Custom Model
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from signxai.torch_signxai import calculate_relevancemap
    from signxai.torch_signxai.utils import remove_softmax
    
    # Define a custom model
    class CustomCNN(nn.Module):
        def __init__(self):
            super(CustomCNN, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
            self.fc1 = nn.Linear(1600, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create model
    model = CustomCNN()
    
    # Load weights if needed
    # model.load_state_dict(torch.load('my_model_weights.pth'))
    model.eval()
    
    # Remove softmax
    model_no_softmax = remove_softmax(model)
    
    # Generate explanation for a custom input
    input_data = torch.randn(1, 1, 28, 28)
    explanation = calculate_relevancemap(model_no_softmax, input_data, method="lrp_epsilon")
    
    # Visualize
    plt.matshow(explanation[0, 0], cmap='seismic', clim=(-1, 1))
    plt.colorbar()
    plt.title('Explanation')
    plt.show()

Batch Processing
----------------

Process multiple inputs at once:

TensorFlow Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process a batch of inputs
    batch_size = 4
    batch_inputs = np.random.random((batch_size, 224, 224, 3))
    
    # Calculate explanations for each image in batch
    batch_explanations = calculate_relevancemap('input_t_gradient', batch_inputs, model)
    
    # Visualize batch results
    fig, axs = plt.subplots(2, batch_size, figsize=(12, 6))
    
    # Top row: Input images
    for i in range(batch_size):
        axs[0, i].imshow(batch_inputs[i])
        axs[0, i].set_title(f'Input {i+1}')
        axs[0, i].axis('off')
    
    # Bottom row: Explanations
    for i in range(batch_size):
        heatmap = batch_explanations[i].sum(axis=-1)
        abs_max = np.max(np.abs(heatmap))
        if abs_max > 0:
            normalized = heatmap / abs_max
        else:
            normalized = heatmap
        axs[1, i].imshow(normalized, cmap='seismic', clim=(-1, 1))
        axs[1, i].set_title(f'Explanation {i+1}')
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

PyTorch Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Process a batch of inputs
    batch_size = 4
    batch_inputs = torch.randn(batch_size, 3, 224, 224)
    
    # Calculate explanations for the batch
    batch_explanations = calculate_relevancemap(model_no_softmax, batch_inputs, method="gradients")
    
    # Visualize batch results
    fig, axs = plt.subplots(2, batch_size, figsize=(12, 6))
    
    # Convert inputs for visualization
    input_np = batch_inputs.permute(0, 2, 3, 1).detach().cpu().numpy()
    
    # Normalize for display
    for i in range(batch_size):
        img = input_np[i]
        img = (img - img.min()) / (img.max() - img.min())
        
        # Top row: Input images
        axs[0, i].imshow(img)
        axs[0, i].set_title(f'Input {i+1}')
        axs[0, i].axis('off')
        
        # Bottom row: Explanations
        explanation = batch_explanations[i].sum(axis=0)
        axs[1, i].imshow(normalize_relevance_map(explanation), cmap='seismic', clim=(-1, 1))
        axs[1, i].set_title(f'Explanation {i+1}')
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

Next Steps
----------

After mastering the basics, you can:

1. Explore advanced usage in the :doc:`advanced_usage` guide
2. Learn about framework-specific features in :doc:`pytorch` and :doc:`tensorflow`
3. Try different explanation methods from the :doc:`/api/methods_list`
4. Work with time series data using the examples in :doc:`/tutorials/time_series`
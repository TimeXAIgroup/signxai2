=======================
Image Classification
=======================

This tutorial demonstrates how to use SignXAI2 for explaining image classification models.

.. contents:: Contents
   :local:
   :depth: 2

Introduction
------------

Image classification is one of the most common applications of deep learning, and understanding how these models make decisions is crucial. In this tutorial, we'll use SignXAI2 to explain predictions from image classification models built with both TensorFlow and PyTorch.

We'll use a pre-trained VGG16 model that classifies images into 1000 categories from the ImageNet dataset.

Setup
-----

First, let's install the required packages:

.. code-block:: bash

    # For TensorFlow
    pip install signxai2[tensorflow]
    
    # For PyTorch
    pip install signxai2[pytorch]
    
    # For both
    pip install signxai2[tensorflow,pytorch]

Let's also download a sample image to work with:

.. code-block:: python

    # Download an example image
    import urllib.request
    
    # Download an image of a dog
    url = "https://farm1.staticflickr.com/148/414245159_7549a49046_z.jpg"
    urllib.request.urlretrieve(url, "dog.jpg")

TensorFlow Implementation
-------------------------

Let's use a pre-trained VGG16 model with TensorFlow:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from signxai.tf_signxai import calculate_relevancemap
    from signxai.utils.utils import normalize_heatmap
    
    # Load the pre-trained model
    model = VGG16(weights='imagenet')
    
    # Remove softmax layer (critical for explanations)
    model.layers[-1].activation = None
    
    # Load and preprocess the image
    img_path = "dog.jpg"
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Make prediction
    preds = model.predict(x)
    top_pred_idx = np.argmax(preds[0])
    print(f"Predicted class: {decode_predictions(preds, top=1)[0][0][1]}")
    
    # Calculate explanations with different methods
    methods = [
        'gradient',
        'gradient_x_input',
        'integrated_gradients',
        'smoothgrad',
        'grad_cam',
        'lrp_z',
        'lrp_epsilon_0_1',
        'lrpsign_z'  # The SIGN method
    ]
    
    explanations = {}
    for method in methods:
        explanations[method] = calculate_relevancemap(
            method, 
            x, 
            model, 
            neuron_selection=top_pred_idx
        )
    
    # Visualize explanations
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image', fontsize=14)
    axs[0].axis('off')
    
    # Explanations
    for i, method in enumerate(methods[:7]):
        axs[i+1].imshow(normalize_heatmap(explanations[method][0]), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(method, fontsize=14)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Highlight the difference between standard LRP and SIGN
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(normalize_heatmap(explanations['lrp_z'][0]), cmap='seismic', clim=(-1, 1))
    plt.title('LRP-Z', fontsize=14)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(normalize_heatmap(explanations['lrpsign_z'][0]), cmap='seismic', clim=(-1, 1))
    plt.title('LRP-SIGN', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

PyTorch Implementation
----------------------

Now let's do the same with PyTorch:

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
    
    # Load the pre-trained model
    model = models.vgg16(pretrained=True)
    model.eval()
    
    # Remove softmax
    model_no_softmax = remove_softmax(model)
    
    # Load and preprocess the image
    img_path = "dog.jpg"
    img = Image.open(img_path).convert('RGB')
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    img_np = np.array(img.resize((224, 224))) / 255.0  # For visualization
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Get the predicted class
    _, predicted_idx = torch.max(output, 1)
    
    # Calculate explanations with different methods
    methods = [
        "gradients",
        "input_t_gradient",
        "integrated_gradients",
        "smoothgrad",
        "grad_cam",
        "lrp_epsilon",  # Equivalent to lrp_epsilon_0_1
        "lrp_alphabeta"  # Equivalent to lrp_alpha_1_beta_0
    ]
    
    explanations = {}
    for method in methods:
        explanations[method] = calculate_relevancemap(
            model_no_softmax, 
            input_tensor, 
            method=method,
            target_class=predicted_idx.item()
        )
    
    # Visualize explanations
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    # Original image
    axs[0].imshow(img_np)
    axs[0].set_title('Original Image', fontsize=14)
    axs[0].axis('off')
    
    # Explanations
    for i, method in enumerate(methods[:7]):
        explanation = explanations[method][0].sum(axis=0)
        axs[i+1].imshow(normalize_relevance_map(explanation), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(method, fontsize=14)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

Advanced Analysis
-----------------

Let's compare class-specific explanations:

.. code-block:: python

    # TensorFlow
    # Get top 3 predicted classes
    top_classes = np.argsort(preds[0])[-3:][::-1]
    class_names = [decode_predictions(preds, top=3)[0][i][1] for i in range(3)]
    
    # Calculate explanations for each class
    class_explanations = {}
    for idx in top_classes:
        class_explanations[idx] = calculate_relevancemap(
            'lrp_epsilon_0_1', 
            x, 
            model, 
            neuron_selection=idx
        )
    
    # Visualize
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image', fontsize=14)
    axs[0].axis('off')
    
    # Class-specific explanations
    for i, (idx, name) in enumerate(zip(top_classes, class_names)):
        axs[i+1].imshow(normalize_heatmap(class_explanations[idx][0]), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(f'Class: {name}', fontsize=14)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

We can also highlight the positive and negative contributions separately:

.. code-block:: python

    # Choose a method
    method = 'lrpsign_z'  # TensorFlow example
    explanation = explanations[method][0]
    
    # Separate positive and negative contributions
    pos_expl = np.maximum(0, explanation)
    neg_expl = np.minimum(0, explanation)
    
    # Normalize
    pos_norm = pos_expl / np.max(pos_expl) if np.max(pos_expl) > 0 else pos_expl
    neg_norm = neg_expl / np.min(neg_expl) if np.min(neg_expl) < 0 else neg_expl
    
    # Visualize
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axs[0].imshow(img)
    axs[0].set_title('Original Image', fontsize=14)
    axs[0].axis('off')
    
    # Combined explanation
    axs[1].imshow(normalize_heatmap(explanation), cmap='seismic', clim=(-1, 1))
    axs[1].set_title(f'{method} - Combined', fontsize=14)
    axs[1].axis('off')
    
    # Positive contributions
    axs[2].imshow(pos_norm, cmap='Reds')
    axs[2].set_title('Positive Contributions', fontsize=14)
    axs[2].axis('off')
    
    # Negative contributions
    axs[3].imshow(-neg_norm, cmap='Blues')
    axs[3].set_title('Negative Contributions', fontsize=14)
    axs[3].axis('off')
    
    plt.tight_layout()
    plt.show()

Conclusion
----------

In this tutorial, we've seen how to:

1. Use SignXAI with pre-trained image classification models
2. Generate explanations using various methods
3. Visualize and compare these explanations
4. Analyze class-specific attributions
5. Separate positive and negative contributions

The explanations reveal which parts of the image influenced the model's prediction, helping us understand and trust the model's decision-making process.

You can apply these techniques to your own image classification models to gain insights into their behavior and improve their performance and trustworthiness.

Interactive Notebooks
---------------------

For hands-on experience with image classification explanations, check out these interactive Jupyter notebooks:

**TensorFlow:**
- `examples/tutorials/tensorflow/tensorflow_basic_usage.ipynb` - Basic usage with VGG16
- `examples/tutorials/tensorflow/tensorflow_advanced_usage.ipynb` - Advanced techniques and LRP methods

**PyTorch:**
- `examples/tutorials/pytorch/pytorch_basic_usage.ipynb` - Basic usage with VGG16  
- `examples/tutorials/pytorch/pytorch_advanced_usage.ipynb` - Advanced techniques and Zennit integration

These notebooks provide step-by-step implementations with code you can run and modify.
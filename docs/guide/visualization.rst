=============
Visualization
=============

SignXAI provides powerful visualization utilities to help you interpret and present explanation results effectively.

.. contents:: Contents
   :local:
   :depth: 2

Basic Visualization
-------------------

The simplest way to visualize explanations is using matplotlib:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    
    # Generate an explanation using the new API (placeholder)
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name="gradient")
    
    # Simple visualization
    plt.figure(figsize=(10, 6))
    plt.imshow(explanation[0].sum(axis=0), cmap='seismic', clim=(-1, 1))
    plt.colorbar(label='Attribution Value')
    plt.title('Explanation Heatmap')
    plt.axis('off')
    plt.show()

SignXAI Visualization Utilities
-------------------------------

SignXAI provides several visualization utilities in the ``signxai.common.visualization`` module:

Normalizing Explanation Maps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from signxai.common.visualization import normalize_relevance_map
    
    # Normalize explanation map to range [-1, 1]
    normalized = normalize_relevance_map(explanation[0].sum(axis=0))
    
    plt.figure(figsize=(10, 6))
    plt.imshow(normalized, cmap='seismic', clim=(-1, 1))
    plt.colorbar(label='Normalized Attribution')
    plt.title('Normalized Explanation')
    plt.axis('off')
    plt.show()

Creating Heatmaps
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from signxai.common.visualization import relevance_to_heatmap
    
    # Convert normalized relevance map to RGB heatmap
    heatmap = relevance_to_heatmap(normalized, cmap='seismic')
    
    plt.figure(figsize=(10, 6))
    plt.imshow(heatmap)
    plt.title('Heatmap Visualization')
    plt.axis('off')
    plt.show()

Overlaying Heatmaps
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from signxai.common.visualization import overlay_heatmap
    
    # Overlay heatmap on original image
    overlaid = overlay_heatmap(original_image, heatmap, alpha=0.6)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(overlaid)
    plt.title('Heatmap Overlay')
    plt.axis('off')
    plt.show()

Multiple Method Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from signxai.common.visualization import visualize_comparison
    
    # Generate explanations with different methods using dynamic parsing
    explanations = {
        'gradient': explain(model, input_tensor, method_name="gradient"),
        'integrated_gradients': explain(model, input_tensor, method_name="integrated_gradients_steps_50"),
        'smoothgrad': explain(model, input_tensor, method_name="smoothgrad_noise_0_2_samples_50"),
        'lrp_epsilon': explain(model, input_tensor, method_name="lrp_epsilon_0_1")
    }
    
    # Convert explanations to suitable format for comparison
    original_image = np.array(img) / 255.0
    
    processed_explanations = []
    method_names = []
    
    for name, expl in explanations.items():
        # Process explanation for visualization (sum across channels)
        processed = expl[0].sum(axis=0) if expl.ndim == 4 else expl.sum(axis=0)
        processed_explanations.append(processed)
        method_names.append(name)
    
    # Visualize comparison
    fig = visualize_comparison(
        original_image,
        processed_explanations,
        method_names,
        figsize=(15, 4),
        cmap='seismic'
    )
    
    plt.tight_layout()
    plt.show()

Advanced Visualization Techniques
---------------------------------

Separating Positive and Negative Contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separate visualization of positive and negative attributions:

.. code-block:: python

    # Get explanation using the new API
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name="gradient")
    
    # Sum across channels
    explanation_flat = explanation[0].sum(axis=0) if explanation.ndim == 4 else explanation[0]
    
    # Separate positive and negative contributions
    pos_explanation = np.maximum(0, explanation_flat)
    neg_explanation = np.minimum(0, explanation_flat)
    
    # Normalize separately
    pos_norm = pos_explanation / np.max(pos_explanation) if np.max(pos_explanation) > 0 else pos_explanation
    neg_norm = neg_explanation / np.min(neg_explanation) if np.min(neg_explanation) < 0 else neg_explanation
    
    # Visualize
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Combined visualization
    axs[0].imshow(normalize_relevance_map(explanation_flat), cmap='seismic', clim=(-1, 1))
    axs[0].set_title('Combined Attribution')
    axs[0].axis('off')
    
    # Positive contributions
    axs[1].imshow(pos_norm, cmap='Reds')
    axs[1].set_title('Positive Contributions')
    axs[1].axis('off')
    
    # Negative contributions
    axs[2].imshow(-neg_norm, cmap='Blues')
    axs[2].set_title('Negative Contributions')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.show()

Channel-Specific Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize attributions for different input channels individually:

.. code-block:: python

    # Get explanation (assuming RGB image, 3 channels) using the new API
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name="gradient")
    
    # Get channel-specific explanations
    r_channel = explanation[0, 0]  # Red channel
    g_channel = explanation[0, 1]  # Green channel
    b_channel = explanation[0, 2]  # Blue channel
    
    # Visualize
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axs[0].imshow(original_image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Channel-specific visualizations
    channels = [r_channel, g_channel, b_channel]
    titles = ['Red Channel', 'Green Channel', 'Blue Channel']
    
    for i, (channel, title) in enumerate(zip(channels, titles)):
        axs[i+1].imshow(normalize_relevance_map(channel), cmap='seismic', clim=(-1, 1))
        axs[i+1].set_title(title)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()

Class Activation Mapping Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Special visualization for Grad-CAM results:

.. code-block:: python

    # Generate Grad-CAM explanation using dynamic parsing
    from signxai.api import explain
    gradcam = explain(model, input_tensor, method_name="grad_cam")
    
    # Normalize Grad-CAM (it's usually positive-only)
    normalized_gradcam = gradcam[0, :, :, 0] if gradcam.ndim == 4 else gradcam[0]
    normalized_gradcam = normalized_gradcam / np.max(normalized_gradcam)
    
    # Create heatmap and overlay
    import cv2
    
    # Convert to heatmap using cv2's colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * normalized_gradcam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize to match original image if necessary
    if heatmap.shape[:2] != original_image.shape[:2]:
        heatmap = cv2.resize(heatmap, original_image.shape[:2][::-1])
    
    # Overlay
    alpha = 0.4
    overlaid = heatmap * alpha + original_image * (1 - alpha)
    overlaid = overlaid / np.max(overlaid)  # Normalize to [0, 1]
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.imshow(overlaid)
    plt.title('Grad-CAM Visualization')
    plt.axis('off')
    plt.show()

Time Series Visualization
-------------------------

For time series data, the visualization differs from images:

.. code-block:: python

    # Generate explanation for time series using the new API
    time_series = np.load('ecg_sample.npy')
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name="gradient")
    
    # For time series, the explanation usually has shape [batch, time, channels]
    # or [batch, channels, time] depending on framework
    
    # Reshape if needed to get a 1D array
    explanation_1d = explanation[0, :, 0] if explanation.ndim == 3 else explanation[0]
    
    plt.figure(figsize=(12, 8))
    
    # Plot original signal
    plt.subplot(2, 1, 1)
    plt.plot(time_series)
    plt.title('Original Time Series')
    plt.grid(True)
    
    # Plot explanation
    plt.subplot(2, 1, 2)
    plt.plot(explanation_1d)
    plt.title('Explanation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Alternative visualization: Colored time series based on explanation
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    plt.figure(figsize=(12, 4))
    
    # Create colormap
    norm = Normalize(vmin=-1, vmax=1)
    cmap = plt.cm.seismic
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Plot time series with color based on explanation
    for i in range(len(time_series) - 1):
        plt.plot(
            [i, i+1], 
            [time_series[i], time_series[i+1]], 
            color=cmap(norm(explanation_1d[i])), 
            linewidth=2
        )
    
    plt.colorbar(sm, label='Attribution Value')
    plt.title('Time Series with Attribution Coloring')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

Interactive Visualization
-------------------------

For more interactive visualization, you can use libraries like Plotly:

.. code-block:: python

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Original Image', 'Explanation'))
    
    # Add original image
    fig.add_trace(
        go.Image(z=original_image),
        row=1, col=1
    )
    
    # Add explanation heatmap
    fig.add_trace(
        go.Heatmap(
            z=explanation[0].sum(axis=0),
            colorscale='RdBu_r',
            zmid=0
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive Explanation Visualization',
        height=500,
        width=1000
    )
    
    # Show figure
    fig.show()

Batch Visualization
-------------------

Visualize multiple inputs and their explanations:

.. code-block:: python

    # Assuming batch_inputs and batch_explanations
    batch_size = len(batch_inputs)
    
    # Create subplot grid
    fig, axs = plt.subplots(2, batch_size, figsize=(4*batch_size, 8))
    
    # Plot each input and its explanation
    for i in range(batch_size):
        # Original input
        axs[0, i].imshow(batch_inputs[i])
        axs[0, i].set_title(f'Input {i+1}')
        axs[0, i].axis('off')
        
        # Explanation
        explanation = batch_explanations[i].sum(axis=0) if batch_explanations[i].ndim == 3 else batch_explanations[i]
        axs[1, i].imshow(normalize_relevance_map(explanation), cmap='seismic', clim=(-1, 1))
        axs[1, i].set_title(f'Explanation {i+1}')
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()

Saving Visualizations
---------------------

Save your visualizations for later use:

.. code-block:: python

    # Create visualization
    plt.figure(figsize=(10, 6))
    # Note: explanation shape depends on framework - handle both cases
    if explanation.ndim == 4:  # TensorFlow format [batch, height, width, channels]
        explanation_viz = explanation[0].sum(axis=-1)
    else:  # PyTorch format [batch, channels, height, width]
        explanation_viz = explanation[0].sum(axis=0)
    
    plt.imshow(normalize_relevance_map(explanation_viz), cmap='seismic', clim=(-1, 1))
    plt.colorbar(label='Attribution Value')
    plt.title('Explanation')
    plt.axis('off')
    
    # Save to file
    plt.savefig('explanation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save all explanations from a method comparison
    for method, expl in explanations.items():
        plt.figure(figsize=(8, 8))
        # Handle both TensorFlow and PyTorch formats
        if expl.ndim == 4:  # TensorFlow format
            explanation_viz = expl[0].sum(axis=-1)
        else:  # PyTorch format
            explanation_viz = expl[0].sum(axis=0)
        plt.imshow(normalize_relevance_map(explanation_viz), cmap='seismic', clim=(-1, 1))
        plt.title(method)
        plt.axis('off')
        plt.savefig(f'explanation_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()

Visualization Best Practices
----------------------------

1. **Use a diverging colormap** (like 'seismic', 'RdBu', or 'coolwarm') for signed explanations.
2. **Normalize explanations** to a fixed range like [-1, 1] for consistent visualization.
3. **Include the original input** alongside explanations for context.
4. **Choose appropriate overlays** - too transparent and you'll miss details, too opaque and you'll hide the original.
5. **Consider channel aggregation carefully** - summing across RGB channels can help visualization but may hide channel-specific details.
6. **Add a colorbar** to indicate the meaning of colors.
7. **Use the same scale** when comparing different methods to ensure fair comparison.
8. **Provide proper titles and annotations** to help viewers understand what they're seeing.

Framework-Specific Considerations
---------------------------------

TensorFlow Outputs
~~~~~~~~~~~~~~~~~~

TensorFlow explanations typically have shape ``[batch, height, width, channels]`` for images:

.. code-block:: python

    # For TensorFlow using the new unified API
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name='gradient')
    
    # Sum across channels for visualization
    explanation_viz = explanation[0].sum(axis=-1)
    
    plt.imshow(normalize_heatmap(explanation_viz), cmap='seismic', clim=(-1, 1))
    plt.show()

PyTorch Outputs
~~~~~~~~~~~~~~~

PyTorch explanations typically have shape ``[batch, channels, height, width]`` for images:

.. code-block:: python

    # For PyTorch using the new unified API
    from signxai.api import explain
    explanation = explain(model, input_tensor, method_name="gradient")
    
    # Sum across channels for visualization
    explanation_viz = explanation[0].sum(axis=0)
    
    plt.imshow(normalize_relevance_map(explanation_viz), cmap='seismic', clim=(-1, 1))
    plt.show()

Custom Colormaps
----------------

Create custom colormaps for specific visualization needs:

.. code-block:: python

    import matplotlib.colors as colors
    
    # Create a custom colormap for positive-only contributions
    def create_pos_cmap():
        return colors.LinearSegmentedColormap.from_list(
            'pos_cmap', 
            [(0, 'white'), (1, 'red')]
        )
    
    # Create a custom colormap for SIGN-specific visualization
    def create_sign_cmap():
        return colors.LinearSegmentedColormap.from_list(
            'sign_cmap', 
            [(0, 'blue'), (0.5, 'white'), (1, 'red')]
        )
    
    # Use custom colormaps
    pos_cmap = create_pos_cmap()
    sign_cmap = create_sign_cmap()
    
    # Visualize with custom colormaps
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(normalize_relevance_map(explanation[0].sum(axis=0)), cmap='seismic', clim=(-1, 1))
    plt.title('Standard Colormap')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.maximum(0, explanation[0].sum(axis=0)), cmap=pos_cmap)
    plt.title('Positive-Only Colormap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(normalize_relevance_map(explanation[0].sum(axis=0)), cmap=sign_cmap, clim=(-1, 1))
    plt.title('SIGN Colormap')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
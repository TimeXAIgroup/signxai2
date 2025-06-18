=============
Common Module
=============

The ``signxai.common`` module contains framework-agnostic utilities and functions used by both the TensorFlow and PyTorch implementations.

.. contents:: Contents
   :local:
   :depth: 2

Visualization Tools
------------------

.. py:module:: signxai.common.visualization

The visualization module provides utilities for visualizing and displaying explanation results.

.. py:function:: normalize_relevance_map(relevance_map, percentile=99)

   Normalizes a relevance map to a specified percentile range.
   
   :param relevance_map: The relevance map to normalize
   :type relevance_map: numpy.ndarray
   :param percentile: The percentile value for upper/lower bounds normalization
   :type percentile: int, optional
   :return: Normalized relevance map with values in range [-1, 1]
   :rtype: numpy.ndarray

.. py:function:: relevance_to_heatmap(relevance_map, cmap='seismic', clip_values=(-1, 1))

   Converts a relevance map to a heatmap visualization.
   
   :param relevance_map: The relevance map to visualize
   :type relevance_map: numpy.ndarray
   :param cmap: The colormap to use for visualization (default: 'seismic')
   :type cmap: str, optional
   :param clip_values: The lower and upper bounds for clipping the relevance values
   :type clip_values: tuple, optional
   :return: RGB heatmap image
   :rtype: numpy.ndarray

.. py:function:: overlay_heatmap(original_image, heatmap, alpha=0.5)

   Overlays a heatmap on an original image.
   
   :param original_image: The original input image
   :type original_image: numpy.ndarray
   :param heatmap: The heatmap to overlay
   :type heatmap: numpy.ndarray
   :param alpha: The blending factor between original image and heatmap
   :type alpha: float, optional
   :return: Overlaid image
   :rtype: numpy.ndarray

.. py:function:: aggregate_and_normalize_relevancemap_rgb(relevance_map)

   Aggregates and normalizes a relevance map for RGB images.
   
   :param relevance_map: The relevance map to aggregate (with channels)
   :type relevance_map: numpy.ndarray
   :return: Aggregated and normalized relevance map
   :rtype: numpy.ndarray

.. py:function:: visualize_comparison(original_image, relevance_maps, method_names, figsize=(15, 5), cmap='seismic')

   Creates a side-by-side visualization of multiple relevance maps.
   
   :param original_image: The original input image
   :type original_image: numpy.ndarray
   :param relevance_maps: List of relevance maps to visualize
   :type relevance_maps: list of numpy.ndarray
   :param method_names: List of method names for the relevance maps
   :type method_names: list of str
   :param figsize: Figure size for the visualization
   :type figsize: tuple, optional
   :param cmap: Colormap to use
   :type cmap: str, optional
   :return: Matplotlib figure
   :rtype: matplotlib.figure.Figure

Common Validation Functions
--------------------------

.. py:module:: signxai.common.validation

The validation module contains utility functions for validating inputs and ensuring compatibility between frameworks.

.. py:function:: validate_model(model, backend=None)

   Validates that a model is compatible with the specified backend.
   
   :param model: The model to validate
   :param backend: The backend to validate against ('tensorflow' or 'pytorch')
   :type backend: str, optional
   :return: True if the model is valid for the backend
   :rtype: bool
   :raises: ValueError: If the model is not valid for the backend

.. py:function:: validate_input(input_tensor, model, backend=None)

   Validates that an input tensor is compatible with the model and backend.
   
   :param input_tensor: The input tensor to validate
   :param model: The model to validate against
   :param backend: The backend to validate against ('tensorflow' or 'pytorch')
   :type backend: str, optional
   :return: Validated input tensor, possibly converted to the appropriate format
   :raises: ValueError: If the input tensor is not valid for the model/backend
   
Framework Detection
-----------------

.. py:module:: signxai.common

The common module provides functions for detecting and handling different frameworks.

.. py:function:: detect_framework(model)

   Detects the framework (TensorFlow or PyTorch) based on the model.
   
   :param model: The model to check
   :return: Framework name ('tensorflow' or 'pytorch')
   :rtype: str
   :raises: ValueError: If the framework cannot be determined
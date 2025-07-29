=============
Utilities
=============

.. py:module:: signxai.utils

The ``signxai.utils`` module provides utility functions used by both the TensorFlow and PyTorch implementations.

.. contents:: Contents
   :local:
   :depth: 2

General Utilities
-----------------

.. py:function:: load_image(path, target_size=(224, 224), preprocess=True)

   Loads and preprocesses an image for use with models.
   
   :param path: Path to the image file
   :type path: str
   :param target_size: Target size for the image (height, width)
   :type target_size: tuple, optional
   :param preprocess: Whether to preprocess the image (ImageNet normalization)
   :type preprocess: bool, optional
   :return: Tuple of (original image, preprocessed image)
   :rtype: tuple

.. py:function:: normalize_heatmap(heatmap, percentile=99)

   Normalizes a heatmap to the specified percentile for visualization.
   
   :param heatmap: The heatmap to normalize
   :type heatmap: numpy.ndarray
   :param percentile: The percentile value for normalization
   :type percentile: float, optional
   :return: Normalized heatmap
   :rtype: numpy.ndarray

.. py:function:: download_image(path, url=None)

   Downloads an example image if it doesn't exist.
   
   :param path: Path to save the image
   :type path: str
   :param url: URL to download from (default: example image)
   :type url: str, optional
   :return: None

.. py:function:: download_model(path, url=None)

   Downloads an example model if it doesn't exist.
   
   :param path: Path to save the model
   :type path: str
   :param url: URL to download from (default: example model)
   :type url: str, optional
   :return: None

.. py:function:: ensure_dir(file_path)

   Ensures that a directory exists.
   
   :param file_path: Path to check/create
   :type file_path: str
   :return: None

.. py:function:: remove_softmax(model)

   Removes the softmax activation from a model.
   
   :param model: Model to process
   :return: Model with softmax removed
   :raises: NotImplementedError: If the model framework is not supported

Visualization Utilities
-----------------------

.. py:function:: plot_relevancemap(relevance_map, ax=None, colorbar_ax=None, colorbar_kw=None, **kwargs)

   Plots a relevance map as a heatmap.
   
   :param relevance_map: The relevance map to visualize
   :type relevance_map: numpy.ndarray
   :param ax: Matplotlib axes to plot on
   :type ax: matplotlib.axes.Axes, optional
   :param colorbar_ax: Axes for the colorbar
   :type colorbar_ax: matplotlib.axes.Axes, optional
   :param colorbar_kw: Additional keyword arguments for the colorbar
   :type colorbar_kw: dict, optional
   :param kwargs: Additional keyword arguments for imshow
   :return: Matplotlib image object
   :rtype: matplotlib.image.AxesImage

.. py:function:: plot_comparison(original_image, explanations, method_names, figsize=(15, 6), cmap='seismic')

   Plots multiple explanation methods side by side for comparison.
   
   :param original_image: The original input image
   :type original_image: numpy.ndarray
   :param explanations: List of explanations to compare
   :type explanations: list
   :param method_names: Names of the methods for the explanations
   :type method_names: list
   :param figsize: Figure size
   :type figsize: tuple, optional
   :param cmap: Colormap for the heatmaps
   :type cmap: str, optional
   :return: Matplotlib figure
   :rtype: matplotlib.figure.Figure

Data Handling
-------------

.. py:function:: batch_to_numpy(batch)

   Converts a batch of tensors to numpy arrays.
   
   :param batch: Batch of tensors
   :type batch: torch.Tensor or tf.Tensor or numpy.ndarray
   :return: Batch as numpy array
   :rtype: numpy.ndarray

.. py:function:: ensure_batch_dimension(x)

   Ensures input has a batch dimension.
   
   :param x: Input tensor
   :type x: numpy.ndarray
   :return: Input with batch dimension
   :rtype: numpy.ndarray

Framework-Specific Utilities
----------------------------

TensorFlow Utilities
~~~~~~~~~~~~~~~~~~~~

.. py:function:: calculate_explanation_innvestigate(model, x, method, **kwargs)

   Interface to iNNvestigate for explanation generation.
   
   :param model: TensorFlow model
   :type model: tf.keras.Model
   :param x: Input tensor
   :type x: numpy.ndarray
   :param method: iNNvestigate method name
   :type method: str
   :param kwargs: Additional parameters for the method
   :return: Explanation
   :rtype: numpy.ndarray

PyTorch Utilities
~~~~~~~~~~~~~~~~~

.. py:function:: numpy_to_torch(array, requires_grad=True)

   Converts a numpy array to a PyTorch tensor.
   
   :param array: Numpy array
   :type array: numpy.ndarray
   :param requires_grad: Whether the tensor requires gradients
   :type requires_grad: bool, optional
   :return: PyTorch tensor
   :rtype: torch.Tensor

.. py:function:: torch_to_numpy(tensor)

   Converts a PyTorch tensor to a numpy array.
   
   :param tensor: PyTorch tensor
   :type tensor: torch.Tensor
   :return: Numpy array
   :rtype: numpy.ndarray
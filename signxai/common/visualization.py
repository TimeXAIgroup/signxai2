"""Visualization utilities for displaying explanations."""
import numpy as np


def normalize_relevance_map(relevance_map, percentile=99):
    """Normalize relevance map by clipping at given percentile.
    
    Args:
        relevance_map: Numpy array of relevance values
        percentile: Percentile for clipping
        
    Returns:
        Normalized relevance map
    """
    abs_map = np.abs(relevance_map)
    vmax = np.percentile(abs_map, percentile)
    if vmax > 0:
        relevance_map = np.clip(relevance_map, -vmax, vmax) / vmax
    return relevance_map


def relevance_to_heatmap(relevance_map, cmap="seismic", symmetric=True):
    """Convert relevance map to RGB heatmap.
    
    Args:
        relevance_map: Normalized relevance map
        cmap: Matplotlib colormap name
        symmetric: If True, ensure colormap is centered at zero
        
    Returns:
        RGB heatmap (H, W, 3) with values in [0, 1]
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if symmetric:
        vmin, vmax = -1, 1
    else:
        vmin, vmax = 0, 1
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    
    return cmap(norm(relevance_map))[..., :3]  # Drop alpha channel


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap on image.
    
    Args:
        image: RGB image (H, W, 3) with values in [0, 1]
        heatmap: RGB heatmap (H, W, 3) with values in [0, 1]
        alpha: Transparency value for overlay
        
    Returns:
        Overlaid image (H, W, 3) with values in [0, 1]
    """
    return (1 - alpha) * image + alpha * heatmap
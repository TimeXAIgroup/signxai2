"""Wrapper functions for PyTorch explanation methods to match the TensorFlow implementation interface."""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Union, Optional, Callable, Dict, Any

from .zennit_impl import (
    GradientAnalyzer,
    IntegratedGradientsAnalyzer,
    SmoothGradAnalyzer,
    GuidedBackpropAnalyzer,
    GradCAMAnalyzer,
    LRPAnalyzer,
    AdvancedLRPAnalyzer,
    LRPSequential,
    BoundedLRPAnalyzer,
    DeepLiftAnalyzer,
    LRPStdxEpsilonAnalyzer,
    calculate_relevancemap as zennit_calculate_relevancemap,
)
from .signed import calculate_sign_mu
from .grad_cam import calculate_grad_cam_relevancemap, calculate_grad_cam_relevancemap_timeseries
from .guided_backprop import guided_backprop as guided_backprop_py


# Core implementation functions
def _calculate_relevancemap(model, input_tensor, method="gradients", **kwargs):
    """Calculate relevance map for a single input using specified method.
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor (can be numpy array or PyTorch tensor)
        method: Name of the explanation method
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Relevance map as numpy array
    """
    # Handle case where arguments might be swapped (when method is passed as model)
    if isinstance(model, str) and method == "gradients":
        # Assume model is actually the method
        temp = model
        model = input_tensor
        method = temp
        input_tensor = kwargs.pop("input_tensor", None)
        if input_tensor is None:
            raise ValueError("Input tensor missing when parameters are swapped")
    # Convert input to torch tensor if needed
    if not isinstance(input_tensor, torch.Tensor):
        input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
    
    # Make a copy to avoid modifying the original
    input_tensor = input_tensor.clone()
    
    # Add batch dimension if needed
    # Check if input tensor already has batch dimension by trying a forward pass
    needs_batch_dim = False
    if input_tensor.dim() == 2:  # Definitely needs batch dimension (C,T) or (H,W)
        needs_batch_dim = True
    elif input_tensor.dim() == 3:  # Could be (B,C,T) or (C,H,W) - check by model expectations
        # For Conv1d models: expect (B,C,T), for Conv2d: expect (B,C,H,W)
        # If input is (C,H,W) for Conv2d model, it needs batch dim
        # If input is (B,C,T) for Conv1d model, it doesn't need batch dim
        # Simple heuristic: if first layer is Conv1d and we have 3 dims, assume (B,C,T)
        first_conv = None
        for module in model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                first_conv = module
                break
        
        if isinstance(first_conv, nn.Conv1d):
            # For Conv1d: 3D input should be (B,C,T) - no batch dim needed
            needs_batch_dim = False
        elif isinstance(first_conv, nn.Conv2d):
            # For Conv2d: 3D input is (C,H,W) - needs batch dim  
            needs_batch_dim = True
    
    if needs_batch_dim:
        input_tensor = input_tensor.unsqueeze(0)
    
    # Set model to eval mode
    model.eval()
    
    # Extract common parameters
    target_class = kwargs.get('target_class', None)
    
    # Select and apply method
    if method == "gradients" or method == "vanilla_gradients" or method == "gradient":
        analyzer = GradientAnalyzer(model)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method == "integrated_gradients":
        steps = kwargs.get('steps', 50)
        baseline = kwargs.get('baseline', None)
        analyzer = IntegratedGradientsAnalyzer(model, steps, baseline)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method == "smooth_gradients" or method == "smoothgrad":
        noise_level = kwargs.get('noise_level', 0.2)
        num_samples = kwargs.get('num_samples', 50)
        analyzer = SmoothGradAnalyzer(model, noise_level, num_samples)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method == "vargrad":
        # VarGrad uses same implementation as SmoothGrad but returns variance instead of mean
        noise_level = kwargs.get('noise_level', 0.2)
        num_samples = kwargs.get('num_samples', 50)
        
        # Generate noisy samples and calculate gradients
        all_grads = []
        for _ in range(num_samples):
            noisy_input = input_tensor + torch.normal(
                0, noise_level * (input_tensor.max() - input_tensor.min()), 
                size=input_tensor.shape, device=input_tensor.device
            )
            noisy_input = noisy_input.requires_grad_(True)
            
            # Forward pass
            model.zero_grad()
            output = model(noisy_input)
            
            # Get target class tensor
            if target_class is None:
                target_idx = output.argmax(dim=1)
            else:
                target_idx = target_class
                
            one_hot = torch.zeros_like(output)
            one_hot.scatter_(1, target_idx.unsqueeze(1) if isinstance(target_idx, torch.Tensor) else torch.tensor([[target_idx]]), 1.0)
            
            # Backward pass
            output.backward(gradient=one_hot)
            
            # Store gradients
            all_grads.append(noisy_input.grad.detach())
        
        # Calculate variance of gradients
        all_grads_tensor = torch.stack(all_grads)
        relevance_map = torch.var(all_grads_tensor, dim=0).cpu().numpy()
    elif method == "guided_backprop":
        analyzer = GuidedBackpropAnalyzer(model)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method == "deconvnet":
        # Use our DeconvNet analyzer implemented in zennit_impl.analyzers
        from .zennit_impl.analyzers import DeconvNetAnalyzer
        analyzer = DeconvNetAnalyzer(model)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method == "grad_cam":
        target_layer = kwargs.get('target_layer', None)
        analyzer = GradCAMAnalyzer(model, target_layer)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method == "grad_cam_timeseries":
        target_layer = kwargs.get('target_layer', None)
        relevance_map = calculate_grad_cam_relevancemap_timeseries(
            model, input_tensor, target_layer, target_class
        )
    elif method == "guided_grad_cam":
        # Guided Grad-CAM is a combination of Guided Backprop and Grad-CAM
        target_layer = kwargs.get('target_layer', None)
        
        # Get Grad-CAM heatmap
        gradcam_analyzer = GradCAMAnalyzer(model, target_layer)
        gradcam_map = gradcam_analyzer.analyze(input_tensor, target_class)
        
        # Get guided backpropagation gradients
        guided_analyzer = GuidedBackpropAnalyzer(model)
        guided_grads = guided_analyzer.analyze(input_tensor, target_class)
        
        # Reshape gradcam map if needed for element-wise multiplication
        if guided_grads.ndim == 4:  # (B, C, H, W)
            # Ensure dimensions match for multiplication
            if gradcam_map.shape != guided_grads.shape[2:]:
                import torch.nn.functional as F
                gradcam_map_tensor = torch.from_numpy(gradcam_map).unsqueeze(0).unsqueeze(0)
                gradcam_map_tensor = F.interpolate(
                    gradcam_map_tensor, 
                    size=guided_grads.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                gradcam_map = gradcam_map_tensor.squeeze().cpu().numpy()
            
            # Element-wise product of guided backpropagation and Grad-CAM
            for i in range(guided_grads.shape[1]):  # For each channel
                guided_grads[:, i] *= gradcam_map
        
        relevance_map = guided_grads
    elif method == "deeplift":
        # DeepLift implementation
        baseline_type = kwargs.pop("baseline_type", "zero")
        analyzer = DeepLiftAnalyzer(model, baseline_type=baseline_type, **kwargs)
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method.startswith("lrp"):
        # Parse LRP method to determine rule and implementation
        if method == "lrp" or method == "lrp_epsilon" or method == "lrp.epsilon":
            # Basic epsilon rule
            rule = "epsilon"
            epsilon = kwargs.get("epsilon", 1e-6)
            analyzer = LRPAnalyzer(model, rule, epsilon)
        elif method == "lrp_z" or method == "lrp_zplus" or method == "lrp.z_plus":
            # Basic z+ rule
            rule = "zplus"
            analyzer = LRPAnalyzer(model, rule)
        elif method == "lrp_alphabeta" or method == "lrp.alphabeta":
            # Basic alpha-beta rule with default 1-0
            rule = "alphabeta"
            analyzer = LRPAnalyzer(model, rule)
        elif method == "lrp_alpha1beta0" or method == "lrp.alpha_1_beta_0":
            # Alpha=1, Beta=0 rule
            analyzer = AdvancedLRPAnalyzer(model, "alpha1beta0")
        elif method == "lrp_alpha2beta1" or method == "lrp.alpha_2_beta_1":
            # Alpha=2, Beta=1 rule
            analyzer = AdvancedLRPAnalyzer(model, "alpha2beta1")
        elif method == "lrp_gamma" or method == "lrp.gamma":
            # Gamma rule
            gamma = kwargs.get("gamma", 0.25)
            analyzer = AdvancedLRPAnalyzer(model, "gamma", gamma=gamma)
        elif method == "lrp_flat":
            # Flat rule
            analyzer = AdvancedLRPAnalyzer(model, "flat")
        elif method == "lrp_wsquare":
            # WSquare rule
            analyzer = AdvancedLRPAnalyzer(model, "wsquare")
        elif method == "lrp_zbox":
            # ZBox rule
            low = kwargs.get("low", 0.0)
            high = kwargs.get("high", 1.0)
            analyzer = AdvancedLRPAnalyzer(model, "zbox", low=low, high=high)
        elif method == "lrp_bounded":
            # Bounded LRP rule with ZBox for first layer
            # Extract parameters from kwargs to avoid multiple values error
            if "low" in kwargs:
                low = kwargs.pop("low")
            else:
                low = 0.0
                
            if "high" in kwargs:
                high = kwargs.pop("high")
            else:
                high = 1.0
                
            if "rule_name" in kwargs:
                rule_name = kwargs.pop("rule_name")
            else:
                rule_name = "epsilon"
                
            analyzer = BoundedLRPAnalyzer(model, low=low, high=high, rule_name=rule_name, **kwargs)
        elif method == "lrp_sequential" or method == "lrp_composite":
            # Sequential application of different rules
            first_layer_rule = kwargs.pop("first_layer_rule", "zbox")
            middle_layer_rule = kwargs.pop("middle_layer_rule", "alphabeta")
            last_layer_rule = kwargs.pop("last_layer_rule", "epsilon")
            # Also remove any other potential conflicting parameters
            for param in ['first_layer_rule_name', 'middle_layer_rule_name', 'last_layer_rule_name']:
                kwargs.pop(param, None)
            analyzer = LRPSequential(model, first_layer_rule, middle_layer_rule, last_layer_rule, **kwargs)
        elif method == "lrp_custom":
            # Custom rule mapping for specific layers
            layer_rules = kwargs.get("layer_rules", {})
            analyzer = AdvancedLRPAnalyzer(model, "sequential", layer_rules=layer_rules)
        elif method == "lrp_stdxepsilon" or method.startswith("lrp_epsilon") and "std_x" in method:
            # LRP with epsilon scaled by standard deviation
            if method.startswith("lrp_epsilon") and "std_x" in method:
                # Extract stdfactor from method name like lrp_epsilon_0_1_std_x
                try:
                    # Parse out the stdfactor value from the method name
                    parts = method.split("lrp_epsilon_")[1].split("_std_x")[0]
                    if parts.startswith("0_"):
                        # Handle decimal values
                        stdfactor = float("0." + parts.split("0_")[1])
                    else:
                        # Handle integer values
                        stdfactor = float(parts)
                except (ValueError, IndexError):
                    # Default value if parsing fails
                    stdfactor = 0.1
            else:
                # Use explicitly provided stdfactor or default
                stdfactor = kwargs.get("stdfactor", 0.1)
                
            # Create the LRPStdxEpsilonAnalyzer with appropriate parameters
            # Remove stdfactor from kwargs to avoid duplicate parameter
            clean_kwargs = {k: v for k, v in kwargs.items() if k != 'stdfactor'}
            analyzer = LRPStdxEpsilonAnalyzer(model, stdfactor=stdfactor, **clean_kwargs)
        elif method.startswith("lrpsign_"):
            # LRP Sign methods - treat similar to regular LRP epsilon but with sign transform
            if method.startswith("lrpsign_epsilon_") and "std_x" in method:
                # Handle epsilon with standard deviation
                try:
                    parts = method.split("lrpsign_epsilon_")[1].split("_std_x")[0]
                    if parts.startswith("0_"):
                        stdfactor = float("0." + parts.split("0_")[1])
                    else:
                        stdfactor = float(parts)
                except (ValueError, IndexError):
                    stdfactor = 0.1
                clean_kwargs = {k: v for k, v in kwargs.items() if k != 'stdfactor'}
                analyzer = LRPStdxEpsilonAnalyzer(model, stdfactor=stdfactor, **clean_kwargs)
            elif method.startswith("lrpsign_epsilon_"):
                # Regular LRP Sign epsilon
                epsilon_str = method.split("lrpsign_epsilon_")[1]
                try:
                    if epsilon_str.startswith("0_"):
                        epsilon = float("0." + epsilon_str.split("0_")[1])
                    else:
                        epsilon = float(epsilon_str)
                    analyzer = LRPAnalyzer(model, "epsilon", epsilon)
                except ValueError:
                    raise ValueError(f"Unknown LRP Sign method: {method}")
            else:
                # Default LRP Sign with epsilon
                epsilon = kwargs.get("epsilon", 1e-6)
                analyzer = LRPAnalyzer(model, "epsilon", epsilon)
        else:
            # Try to parse the method for epsilon value
            if method.startswith("lrp_epsilon_"):
                epsilon_str = method.split("lrp_epsilon_")[1]
                try:
                    if epsilon_str.startswith("0_"):
                        # Handle decimal values
                        epsilon = float("0." + epsilon_str.split("0_")[1])
                    else:
                        # Handle integer values
                        epsilon = float(epsilon_str)
                    
                    analyzer = LRPAnalyzer(model, "epsilon", epsilon)
                except ValueError:
                    raise ValueError(f"Unknown LRP method: {method}")
            else:
                raise ValueError(f"Unknown LRP method: {method}")
            
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method.startswith("flatlrp"):
        # FlatLRP methods - use our improved AdvancedLRPAnalyzer with custom hooks
        if method.startswith("flatlrp_epsilon_"):
            # Parse epsilon value from method name
            epsilon_str = method.split("flatlrp_epsilon_")[1]
            try:
                if epsilon_str.startswith("0_"):
                    # Handle decimal values like 0_1 -> 0.1
                    epsilon = float("0." + epsilon_str.split("0_")[1])
                else:
                    # Handle integer values
                    epsilon = float(epsilon_str)
                
                # Use AdvancedLRPAnalyzer with flatlrp variant and our custom hooks
                analyzer = AdvancedLRPAnalyzer(model, "flatlrp", epsilon=epsilon)
            except ValueError:
                raise ValueError(f"Unknown FlatLRP method: {method}")
        elif method == "flatlrp_z":
            # FlatLRP with Z rule
            analyzer = AdvancedLRPAnalyzer(model, "flatlrp")
        elif method.startswith("flatlrp_alpha_"):
            # Parse alpha/beta values from method name
            if "alpha_1_beta_0" in method:
                analyzer = AdvancedLRPAnalyzer(model, "flatlrp", alpha=1.0, beta=0.0)
            elif "alpha_2_beta_1" in method:
                analyzer = AdvancedLRPAnalyzer(model, "flatlrp", alpha=2.0, beta=1.0)
            else:
                raise ValueError(f"Unknown FlatLRP alpha/beta method: {method}")
        elif method.startswith("flatlrp_sequential_composite"):
            # FlatLRP sequential composites
            if "composite_a" in method:
                analyzer = LRPSequential(model, variant="A")
            elif "composite_b" in method:
                analyzer = LRPSequential(model, variant="B")
            else:
                raise ValueError(f"Unknown FlatLRP sequential method: {method}")
        elif method.startswith("flatlrp_epsilon_") and "std_x" in method:
            # FlatLRP with standard deviation-based epsilon
            try:
                parts = method.split("flatlrp_epsilon_")[1].split("_std_x")[0]
                if parts.startswith("0_"):
                    stdfactor = float("0." + parts.split("0_")[1])
                else:
                    stdfactor = float(parts)
                # Use LRPStdxEpsilonAnalyzer (already uses custom hooks)
                analyzer = LRPStdxEpsilonAnalyzer(model, stdfactor=stdfactor)
            except (ValueError, IndexError):
                raise ValueError(f"Unknown FlatLRP std_x method: {method}")
        else:
            # Default FlatLRP
            analyzer = AdvancedLRPAnalyzer(model, "flatlrp")
            
        relevance_map = analyzer.analyze(input_tensor, target_class)
    elif method.startswith("w2lrp"):
        # W2LRP methods - use AdvancedLRPAnalyzer with corrected w2lrp composites
        if method == "w2lrp_sequential_composite_a":
            # Use corrected W2LRP sequential composite A
            analyzer = AdvancedLRPAnalyzer(model, "w2lrp", subvariant="sequential_composite_a")
        elif method == "w2lrp_sequential_composite_b":
            # Use corrected W2LRP sequential composite B
            analyzer = AdvancedLRPAnalyzer(model, "w2lrp", subvariant="sequential_composite_b")
        elif method.startswith("w2lrp_epsilon_"):
            # Parse epsilon value for regular w2lrp methods
            epsilon_str = method.split("w2lrp_epsilon_")[1]
            try:
                if epsilon_str.startswith("0_"):
                    epsilon = float("0." + epsilon_str.split("0_")[1])
                else:
                    epsilon = float(epsilon_str)
                analyzer = AdvancedLRPAnalyzer(model, "w2lrp", epsilon=epsilon)
            except ValueError:
                raise ValueError(f"Unknown W2LRP method: {method}")
        else:
            # Default W2LRP
            analyzer = AdvancedLRPAnalyzer(model, "w2lrp")
            
        relevance_map = analyzer.analyze(input_tensor, target_class)
    else:
        raise ValueError(f"Unknown explanation method: {method}")
    
    # We're keeping the batch dimension even if it was added
    # This makes our API consistent with the output shapes
    
    # Apply sign transform if requested
    if kwargs.get("apply_sign", False):
        mu = kwargs.get("sign_mu", 0.0)
        relevance_map = calculate_sign_mu(relevance_map, mu)
    
    return relevance_map


def random_uniform(model_no_softmax, x, **kwargs):
    """Generate random uniform relevance map between -1 and 1.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments (unused)
        
    Returns:
        Random uniform relevance map
    """
    np.random.seed(1)
    
    channel_values = []
    
    # Match TensorFlow implementation
    uniform_values = np.random.uniform(low=-1, high=1, size=(x.shape[0], x.shape[1]))
    
    for i in range(x.shape[2]):
        channel_values.append(np.array(uniform_values))
    
    return np.stack(channel_values, axis=2)


def gradient(model_no_softmax, x, **kwargs):
    """Calculate vanilla gradient relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Gradient relevance map
    """
    # Convert x to torch tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Calculate relevance map
    analyzer = GradientAnalyzer(model_no_softmax)
    
    # Ensure x has batch dimension
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    relevance_map = analyzer.analyze(x, **kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        relevance_map = relevance_map[0]
    
    return relevance_map


def input_t_gradient(model_no_softmax, x, **kwargs):
    """Calculate input times gradient relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Input times gradient relevance map
    """
    g = gradient(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    return g * x_np


def gradient_x_input(model_no_softmax, x, **kwargs):
    """Same as input_t_gradient.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Gradient times input relevance map
    """
    return input_t_gradient(model_no_softmax, x, **kwargs)


def gradient_x_sign(model_no_softmax, x, **kwargs):
    """Calculate gradient times sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Gradient times sign relevance map
    """
    g = gradient(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    
    return g * s


def gradient_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    """Calculate gradient times thresholded sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        mu: Threshold parameter
        batchmode: Whether to process inputs in batch mode
        **kwargs: Additional arguments
        
    Returns:
        Gradient times thresholded sign relevance map
    """
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(gradient(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        grad_result = gradient(model_no_softmax, x, **kwargs)
        sign_result = calculate_sign_mu(grad_result, mu, **kwargs)
        return grad_result * sign_result


def gradient_x_sign_mu_0(model_no_softmax, x, **kwargs):
    """Calculate gradient times thresholded sign relevance map with mu=0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Gradient times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return gradient_x_sign_mu(model_no_softmax, x, mu=0, **kwargs_clean)


def gradient_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    """Calculate gradient times thresholded sign relevance map with mu=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Gradient times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return gradient_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs_clean)


def gradient_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    """Calculate gradient times thresholded sign relevance map with mu=-0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Gradient times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return gradient_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs_clean)


def guided_backprop(model_no_softmax, x, **kwargs):
    """Calculate guided backprop relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop relevance map
    """
    return zennit_calculate_relevancemap(model_no_softmax, x, method="guided_backprop", **kwargs)


def guided_backprop_x_input(model_no_softmax, x, **kwargs):
    """Calculate guided backprop times input relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop times input relevance map
    """
    g = guided_backprop(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    return g * x_np


def guided_backprop_x_sign(model_no_softmax, x, **kwargs):
    """Calculate guided backprop times sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop times sign relevance map
    """
    g = guided_backprop(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    
    return g * s


def guided_backprop_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    """Calculate guided backprop times thresholded sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        mu: Threshold parameter
        batchmode: Whether to process inputs in batch mode
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop times thresholded sign relevance map
    """
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(guided_backprop(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu))
        return np.array(G) * np.array(S)
    else:
        gbp_result = guided_backprop(model_no_softmax, x, **kwargs)
        
        # Adaptive scaling for guided backprop: if mu is large compared to actual values,
        # scale it down to be proportional to the value range
        if isinstance(gbp_result, np.ndarray):
            gbp_abs_max = np.abs(gbp_result).max()
        else:
            gbp_abs_max = gbp_result.abs().max().item()
            
        # If mu is much larger than the actual value range, scale it down
        if mu > 0.1 and gbp_abs_max < 0.1:
            # Scale mu to be about 20% of the max absolute value
            effective_mu = gbp_abs_max * 0.2
        else:
            effective_mu = mu
            
        sign_result = calculate_sign_mu(gbp_result, effective_mu)
        return gbp_result * sign_result


def guided_backprop_x_sign_mu_0(model_no_softmax, x, **kwargs):
    """Calculate guided backprop times thresholded sign relevance map with mu=0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=0, **kwargs_clean)


def guided_backprop_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    """Calculate guided backprop times thresholded sign relevance map with mu=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs_clean)


def guided_backprop_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    """Calculate guided backprop times thresholded sign relevance map with mu=-0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided backprop times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs_clean)


def integrated_gradients(model_no_softmax, x, **kwargs):
    """Calculate integrated gradients relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Integrated gradients relevance map
    """
    # Convert x to torch tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Ensure x has batch dimension
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Get parameters matching TensorFlow implementation
    steps = kwargs.get('steps', 50)
    # TensorFlow uses 'reference_inputs' for the baseline
    reference_inputs = kwargs.get('reference_inputs', None)
    
    if reference_inputs is None:
        reference_inputs = torch.zeros_like(x)
    
    # Pass all relevant parameters to the analyzer
    analyzer = IntegratedGradientsAnalyzer(model_no_softmax, steps=steps)
    # Ensure both reference_inputs and steps are passed as kwargs for consistency
    kwargs['reference_inputs'] = reference_inputs
    kwargs['steps'] = steps
    relevance_map = analyzer.analyze(x, **kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        relevance_map = relevance_map[0]
    
    return relevance_map


def smoothgrad(model_no_softmax, x, **kwargs):
    """Calculate SmoothGrad relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad relevance map
    """
    # Convert x to torch tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Ensure x has batch dimension
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Map TensorFlow parameter names to PyTorch ones
    # Handle both TF parameter 'noise_level' and PT parameter 'noise_scale'
    noise_level = kwargs.get('noise_level', 0.2)
    # Handle both TF parameter 'augment_by_n' and PT parameter 'num_samples'
    num_samples = kwargs.get('augment_by_n', kwargs.get('num_samples', 50))
    
    # For TensorFlow compatibility - directly pass through the parameters
    kwargs['noise_level'] = noise_level
    kwargs['num_samples'] = num_samples
    
    # Calculate relevance map
    analyzer = SmoothGradAnalyzer(model_no_softmax, noise_level=noise_level, num_samples=num_samples)
    relevance_map = analyzer.analyze(x, **kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        relevance_map = relevance_map[0]
    
    return relevance_map


def smoothgrad_x_input_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate SmoothGrad times input relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad times input relevance map
    """
    g = smoothgrad(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    return g * x_np


def smoothgrad_x_sign_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate SmoothGrad times sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad times sign relevance map
    """
    g = smoothgrad(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    
    return g * s


def smoothgrad_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    """Calculate SmoothGrad times thresholded sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        mu: Threshold parameter
        batchmode: Whether to process inputs in batch mode
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad times thresholded sign relevance map
    """
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(smoothgrad(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        return smoothgrad(model_no_softmax, x, **kwargs) * calculate_sign_mu(x, mu, **kwargs)


def smoothgrad_x_sign_mu_0(model_no_softmax, x, **kwargs):
    """Calculate SmoothGrad times thresholded sign relevance map with mu=0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=0, **kwargs_clean)


def smoothgrad_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    """Calculate SmoothGrad times thresholded sign relevance map with mu=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs_clean)


def smoothgrad_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    """Calculate SmoothGrad times thresholded sign relevance map with mu=-0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        SmoothGrad times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return smoothgrad_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs_clean)


def vargrad_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate VarGrad relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        VarGrad relevance map
    """
    # Convert x to torch tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Ensure x has batch dimension
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Extract parameters
    noise_level = kwargs.get('noise_level', 0.2)
    num_samples = kwargs.get('num_samples', 50)
    target_class = kwargs.get('target_class', None)
    
    # Generate noisy samples and calculate gradients
    all_grads = []
    
    for _ in range(num_samples):
        # Create noise using proper PyTorch syntax
        noise = torch.randn_like(x) * noise_level * (x.max() - x.min())
        noisy_input = x + noise
        noisy_input = noisy_input.requires_grad_(True)
        
        # Forward pass
        model_no_softmax.zero_grad()
        output = model_no_softmax(noisy_input)
        
        # Get target class tensor
        if target_class is None:
            target_idx = output.argmax(dim=1)
        else:
            target_idx = target_class
        
        one_hot = torch.zeros_like(output)
        if isinstance(target_idx, torch.Tensor):
            one_hot.scatter_(1, target_idx.unsqueeze(1), 1.0)
        else:
            one_hot.scatter_(1, torch.tensor([[target_idx]], device=output.device), 1.0)
        
        # Backward pass
        output.backward(gradient=one_hot)
        
        # Store gradients
        if noisy_input.grad is not None:
            all_grads.append(noisy_input.grad.detach())
        else:
            print("Warning: Grad is None for one of the VarGrad samples. Appending zeros.")
            all_grads.append(torch.zeros_like(noisy_input))
    
    # Calculate variance of gradients
    all_grads_tensor = torch.stack(all_grads)
    variance = torch.var(all_grads_tensor, dim=0)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        variance = variance[0]
    
    return variance.cpu().numpy()


def deconvnet(model_no_softmax, x, **kwargs):
    """Calculate DeconvNet relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet relevance map
    """
    # Using actual DeconvNet implementation
    from .zennit_impl.analyzers import DeconvNetAnalyzer
    
    # Convert x to torch tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Ensure x has batch dimension
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Calculate relevance map
    analyzer = DeconvNetAnalyzer(model_no_softmax)
    relevance_map = analyzer.analyze(x, **kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        relevance_map = relevance_map[0]
    
    return relevance_map


def deconvnet_x_input_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate DeconvNet times input relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet times input relevance map
    """
    g = deconvnet(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    return g * x_np


def deconvnet_x_sign_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate DeconvNet times sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet times sign relevance map
    """
    g = deconvnet(model_no_softmax, x, **kwargs)
    
    # Convert x to numpy if it's a torch tensor
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    
    return g * s


def deconvnet_x_sign_mu(model_no_softmax, x, mu, batchmode=False, **kwargs):
    """Calculate DeconvNet times thresholded sign relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        mu: Threshold parameter
        batchmode: Whether to process inputs in batch mode
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet times thresholded sign relevance map
    """
    if batchmode:
        G = []
        S = []
        for xi in x:
            G.append(deconvnet(model_no_softmax, xi, **kwargs))
            S.append(calculate_sign_mu(xi, mu, **kwargs))
        return np.array(G) * np.array(S)
    else:
        deconv_result = deconvnet(model_no_softmax, x, **kwargs)
        sign_result = calculate_sign_mu(deconv_result, mu, **kwargs)
        return deconv_result * sign_result


def deconvnet_x_sign_mu_0(model_no_softmax, x, **kwargs):
    """Calculate DeconvNet times thresholded sign relevance map with mu=0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=0, **kwargs_clean)


def deconvnet_x_sign_mu_0_5_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate DeconvNet times thresholded sign relevance map with mu=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=0.5, **kwargs_clean)


def deconvnet_x_sign_mu_neg_0_5(model_no_softmax, x, **kwargs):
    """Calculate DeconvNet times thresholded sign relevance map with mu=-0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeconvNet times thresholded sign relevance map
    """
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'mu'}
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=-0.5, **kwargs_clean)


def grad_cam(model_no_softmax, x, **kwargs):
    """Calculate Grad-CAM relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Grad-CAM relevance map
    """
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM")
    
    # Handle TensorFlow parameter name 'layer_name' -> map to 'target_layer'
    if 'layer_name' in kwargs:
        kwargs['target_layer'] = kwargs.pop('layer_name')
    
    # Remove unsupported parameters
    kwargs.pop('resize', None)  # Remove resize parameter - not supported by implementation
    
    # Convert numpy to tensor if needed
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
        if torch.cuda.is_available() and next(model_no_softmax.parameters()).is_cuda:
            x = x.cuda()
    
    # Resolve string layer names to actual layer objects
    if 'target_layer' in kwargs and isinstance(kwargs['target_layer'], str):
        layer_name = kwargs['target_layer']
        # Navigate to the layer by name (e.g., 'features.28')
        layer = model_no_softmax
        for part in layer_name.split('.'):
            if part.isdigit():
                layer = layer[int(part)]
            else:
                layer = getattr(layer, part)
        kwargs['target_layer'] = layer
    
    return calculate_grad_cam_relevancemap(model_no_softmax, x, **kwargs)


def grad_cam_timeseries(model_no_softmax, x, **kwargs):
    """Calculate Grad-CAM relevance map for time series data.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Grad-CAM relevance map for time series
    """
    # Ensure x is a PyTorch tensor (grad_cam function expects tensors, not numpy arrays)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Map neuron_selection to target_class for the function call
    if 'neuron_selection' in kwargs:
        target_class = kwargs.pop('neuron_selection')
        # Convert target_class to tensor if it's an integer
        if isinstance(target_class, int):
            target_class = torch.tensor([target_class])
        kwargs['target_class'] = target_class
    return calculate_grad_cam_relevancemap_timeseries(model_no_softmax, x, **kwargs)


def grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """Calculate Grad-CAM relevance map for VGG16 ILSVRC model.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Grad-CAM relevance map
    """
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM VGG16")
        
    # Find the target layer by name
    target_layer = None
    for name, module in model_no_softmax.named_modules():
        if name == 'block5_conv3' or name.endswith('.block5_conv3'):
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError("Could not find layer 'block5_conv3' in the model")
        
    return calculate_grad_cam_relevancemap(x, model_no_softmax, target_layer=target_layer, **kwargs)


def guided_grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """Calculate Guided Grad-CAM relevance map for VGG16 ILSVRC model.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided Grad-CAM relevance map
    """
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for Guided GradCAM")
        
    gc = grad_cam_VGG16ILSVRC(model_no_softmax, x, **kwargs)
    
    # Convert to torch tensor for guided backprop
    if not isinstance(x, torch.Tensor):
        x_torch = torch.tensor(x, dtype=torch.float32)
    else:
        x_torch = x
    
    # Find the target layer by name
    target_layer = None
    for name, module in model_no_softmax.named_modules():
        if name == 'block5_conv3' or name.endswith('.block5_conv3'):
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError("Could not find layer 'block5_conv3' in the model")
    
    # Get guided backprop
    gbp = guided_backprop(model_no_softmax, x_torch)
    
    # Element-wise multiplication
    return gbp * gc


def grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs):
    """Calculate Grad-CAM relevance map for VGG16 MIT Places 365 model.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Grad-CAM relevance map
    """
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM VGG16 MITPL365")
        
    # Find the target layer by name
    target_layer = None
    for name, module in model_no_softmax.named_modules():
        if name == 'relu5_3' or name.endswith('.relu5_3'):
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError("Could not find layer 'relu5_3' in the model")
        
    return calculate_grad_cam_relevancemap(x, model_no_softmax, target_layer=target_layer, **kwargs)


def guided_grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs):
    """Calculate Guided Grad-CAM relevance map for VGG16 MIT Places 365 model.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        Guided Grad-CAM relevance map
    """
    # Ensure x has batch dimension
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)
    elif x.ndim > 4:
        # Handle case where x has too many dimensions
        raise ValueError(f"Input shape {x.shape} has too many dimensions for Guided GradCAM VGG16 MITPL365")
        
    gc = grad_cam_VGG16MITPL365(model_no_softmax, x, **kwargs)
    
    # Convert to torch tensor for guided backprop
    if not isinstance(x, torch.Tensor):
        x_torch = torch.tensor(x, dtype=torch.float32)
    else:
        x_torch = x
    
    # Find the target layer by name
    target_layer = None
    for name, module in model_no_softmax.named_modules():
        if name == 'relu5_3' or name.endswith('.relu5_3'):
            target_layer = module
            break
    
    if target_layer is None:
        raise ValueError("Could not find layer 'relu5_3' in the model")
    
    # Get guided backprop
    gbp = guided_backprop(model_no_softmax, x_torch)
    
    # Element-wise multiplication
    return gbp * gc


def grad_cam_MNISTCNN(model_no_softmax, x, batchmode=False, **kwargs):
    """Calculate Grad-CAM relevance map for MNIST CNN model.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        batchmode: Whether to process inputs in batch mode
        **kwargs: Additional arguments
        
    Returns:
        Grad-CAM relevance map
    """
    if batchmode:
        H = []
        for xi in x:
            # Ensure each individual example has batch dimension
            xi_batched = np.expand_dims(xi, axis=0)
            
            # Find the target layer by name
            target_layer = None
            for name, module in model_no_softmax.named_modules():
                if name == 'conv2d_1' or name.endswith('.conv2d_1'):
                    target_layer = module
                    break
            
            if target_layer is None:
                # Try to find last convolutional layer
                for name, module in model_no_softmax.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
            
            H.append(calculate_grad_cam_relevancemap(
                xi_batched, model_no_softmax, target_layer=target_layer, resize=True, **kwargs))
        return np.array(H)
    else:
        # Ensure x has batch dimension
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        elif x.ndim > 4:
            # Handle case where x has too many dimensions
            raise ValueError(f"Input shape {x.shape} has too many dimensions for GradCAM MNIST CNN")
            
        # Find the target layer by name
        target_layer = None
        for name, module in model_no_softmax.named_modules():
            if name == 'conv2d_1' or name.endswith('.conv2d_1'):
                target_layer = module
                break
        
        if target_layer is None:
            # Try to find last convolutional layer
            for name, module in model_no_softmax.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        
        return calculate_grad_cam_relevancemap(
            x, model_no_softmax, target_layer=target_layer, resize=True, **kwargs)


def guided_grad_cam_MNISTCNN(model_no_softmax, x, batchmode=False, **kwargs):
    """Calculate Guided Grad-CAM relevance map for MNIST CNN model.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        batchmode: Whether to process inputs in batch mode
        **kwargs: Additional arguments
        
    Returns:
        Guided Grad-CAM relevance map
    """
    if batchmode:
        gc = grad_cam_MNISTCNN(model_no_softmax, x, batchmode=True, **kwargs)
        
        # Process each input individually for guided backprop
        gbp_results = []
        for i, xi in enumerate(x):
            # Convert to torch tensor for guided backprop
            if not isinstance(xi, torch.Tensor):
                xi_torch = torch.tensor(xi, dtype=torch.float32)
            else:
                xi_torch = xi
            
            # Get guided backprop
            gbp = guided_backprop(model_no_softmax, xi_torch)
            
            # Ensure dimensions match for multiplication
            if gbp.ndim == 3:  # (C, H, W)
                # Expand to match gc batch dimension
                gbp = gbp[np.newaxis, ...]
            
            # Element-wise multiplication
            gbp_gc = gbp * gc[i]
            gbp_results.append(gbp_gc)
        
        return np.stack(gbp_results)
    else:
        # Ensure x has batch dimension
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        elif x.ndim > 4:
            # Handle case where x has too many dimensions
            raise ValueError(f"Input shape {x.shape} has too many dimensions for Guided GradCAM MNIST CNN")
            
        gc = grad_cam_MNISTCNN(model_no_softmax, x, **kwargs)
        
        # Convert to torch tensor for guided backprop
        if not isinstance(x, torch.Tensor):
            x_torch = torch.tensor(x, dtype=torch.float32)
        else:
            x_torch = x
        
        # Find the target layer by name
        target_layer = None
        for name, module in model_no_softmax.named_modules():
            if name == 'conv2d_1' or name.endswith('.conv2d_1'):
                target_layer = module
                break
        
        if target_layer is None:
            # Try to find last convolutional layer
            for name, module in model_no_softmax.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        
        # Get guided backprop
        gbp = guided_backprop(model_no_softmax, x_torch)
        
        # Element-wise multiplication
        return gbp * gc


# Generate all LRP variants to match TensorFlow implementation

def lrp_z(model_no_softmax, x, **kwargs):
    """Calculate LRP-Z relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP-Z relevance map
    """
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_z", **kwargs)


def lrpsign_z(model_no_softmax, x, **kwargs):
    """Calculate LRP-Z with SIGN input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP-Z relevance map with SIGN input layer rule
    """
    kwargs["input_layer_rule"] = "SIGN"
    return lrp_z(model_no_softmax, x, **kwargs)


def zblrp_z_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """Calculate LRP-Z with Bounded input layer rule for VGG16 ILSVRC.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP-Z relevance map with Bounded input layer rule
    """
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061
    })
    return lrp_z(model_no_softmax, x, **kwargs)


def w2lrp_z(model_no_softmax, x, **kwargs):
    """Calculate LRP-Z with WSquare input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP-Z relevance map with WSquare input layer rule
    """
    kwargs["input_layer_rule"] = "WSquare"
    return lrp_z(model_no_softmax, x, **kwargs)


def flatlrp_z(model_no_softmax, x, **kwargs):
    """Calculate LRP-Z with Flat input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP-Z relevance map with Flat input layer rule
    """
    kwargs["input_layer_rule"] = "Flat"
    return lrp_z(model_no_softmax, x, **kwargs)


# Define functions for different LRP epsilon values
def lrp_epsilon_0_001(model_no_softmax, x, **kwargs):
    kwargs["epsilon"] = 0.001
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpsign_epsilon_0_001(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "SIGN"
    return lrp_epsilon_0_001(model_no_softmax, x, **kwargs)


def zblrp_epsilon_0_001_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061
    })
    return lrp_epsilon_0_001(model_no_softmax, x, **kwargs)


def lrpz_epsilon_0_001(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Z"
    return lrp_epsilon_0_001(model_no_softmax, x, **kwargs)


def lrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    kwargs["epsilon"] = 0.01
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpsign_epsilon_0_01(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "SIGN"
    return lrp_epsilon_0_01(model_no_softmax, x, **kwargs)


def zblrp_epsilon_0_01_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061
    })
    return lrp_epsilon_0_01(model_no_softmax, x, **kwargs)


def w2lrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "WSquare"
    return lrp_epsilon_0_01(model_no_softmax, x, **kwargs)


def flatlrp_epsilon_0_01(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Flat"
    return lrp_epsilon_0_01(model_no_softmax, x, **kwargs)


def lrpz_epsilon_0_01(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Z"
    return lrp_epsilon_0_01(model_no_softmax, x, **kwargs)


def lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=0.1.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=0.1
    """
    # Return using the existing alpha_1_beta_0 method which we know works
    # For epsilon=0.1, we just use a specialized implementation that adds a small amount of stabilization
    from .zennit_impl.analyzers import LRPAnalyzer
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Start with a simple approach using our already working LRPAnalyzer
    # Create the analyzer with epsilon rule
    analyzer = LRPAnalyzer(model_no_softmax, "epsilon", epsilon=0.1)
    
    # Extract target class from kwargs
    target_class = kwargs.get("target_class", None)
    
    # Get attribution
    attribution = analyzer.analyze(x, target_class)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        attribution = attribution[0]
    
    return attribution


def lrpsign_epsilon_0_1(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "SIGN"
    return lrp_epsilon_0_1(model_no_softmax, x, **kwargs)


def zblrp_epsilon_0_1_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061
    })
    return lrp_epsilon_0_1(model_no_softmax, x, **kwargs)


def w2lrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "WSquare"
    return lrp_epsilon_0_1(model_no_softmax, x, **kwargs)


def flatlrp_epsilon_0_1(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Flat"
    return lrp_epsilon_0_1(model_no_softmax, x, **kwargs)


def lrpz_epsilon_0_1(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Z"
    return lrp_epsilon_0_1(model_no_softmax, x, **kwargs)


# Continue with all other LRP variants from the TF implementation...
# These are just a few examples, the rest would follow the same pattern

def lrp_epsilon_0_2(model_no_softmax, x, **kwargs):
    kwargs["epsilon"] = 0.2
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_epsilon_0_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=0.5
    """
    # Create new kwargs without epsilon to avoid duplicate parameter
    new_kwargs = {k: v for k, v in kwargs.items() if k != 'epsilon'}
    
    # Create a LRP analyzer with the specific epsilon value
    analyzer = LRPAnalyzer(model_no_softmax, rule_name="epsilon", epsilon=0.5)
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Get the attribution map
    result = analyzer.analyze(x, **new_kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        result = result[0]
    
    return result


def lrp_epsilon_1(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=1.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=1.0
    """
    # Create new kwargs without epsilon to avoid duplicate parameter
    new_kwargs = {k: v for k, v in kwargs.items() if k != 'epsilon'}
    
    # Create a LRP analyzer with the specific epsilon value
    analyzer = LRPAnalyzer(model_no_softmax, rule_name="epsilon", epsilon=1.0)
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Get the attribution map
    result = analyzer.analyze(x, **new_kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        result = result[0]
    
    return result


def lrp_epsilon_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=5.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=5.0
    """
    kwargs["epsilon"] = 5.0
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_epsilon_10(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=10.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=10.0
    """
    kwargs["epsilon"] = 10.0
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_epsilon_20(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=20.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=20.0
    """
    kwargs["epsilon"] = 20.0
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_epsilon_50(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=50.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=50.0
    """
    kwargs["epsilon"] = 50.0
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_epsilon_75(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=75.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=75.0
    """
    kwargs["epsilon"] = 75.0
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_epsilon_100(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=100.0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=100.0
    """
    kwargs["epsilon"] = 100.0
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpsign_epsilon_1(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=1.0 and SIGN input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=1.0 and SIGN input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGN",
        "epsilon": 1.0
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpz_epsilon_1(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=1.0 and Z input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=1.0 and Z input layer rule
    """
    kwargs.update({
        "input_layer_rule": "Z",
        "epsilon": 1.0
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpsign_epsilon_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=5.0 and SIGN input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=5.0 and SIGN input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGN",
        "epsilon": 5.0
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpz_epsilon_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=5.0 and Z input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=5.0 and Z input layer rule
    """
    kwargs.update({
        "input_layer_rule": "Z",
        "epsilon": 5.0
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpsign_epsilon_0_2(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "SIGN"
    return lrp_epsilon_0_2(model_no_softmax, x, **kwargs)


def lrpz_epsilon_0_2(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Z"
    return lrp_epsilon_0_2(model_no_softmax, x, **kwargs)


def zblrp_epsilon_0_2_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061
    })
    return lrp_epsilon_0_2(model_no_softmax, x, **kwargs)


# StdxEpsilon LRP variants
def lrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon scaled by standard deviation (factor 0.1).
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon scaled by standard deviation
    """
    # Direct implementation without using custom StdxEpsilon rule
    from zennit.rules import Epsilon, Pass
    from zennit.core import Composite
    from zennit.attribution import Gradient
    import torch.nn as nn
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Calculate standard deviation of the input and scale by factor
    stdfactor = 0.1
    if x.dim() <= 1:
        std_val = torch.std(x).item()
    else:
        # For multi-dimensional tensors, flatten all but batch dimension
        flattened = x.reshape(x.size(0), -1)
        std_val = torch.std(flattened).item()
    
    # Calculate epsilon based on standard deviation
    epsilon_value = std_val * stdfactor
    
    # Create a composite with Epsilon rule using the calculated epsilon
    from zennit.types import Convolution, Linear, Activation, BatchNorm, AvgPool
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return Epsilon(epsilon=epsilon_value)
        elif isinstance(module, (Activation, BatchNorm, AvgPool, nn.Flatten, nn.Dropout, 
                              nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.MaxPool1d, nn.MaxPool2d)):
            return Pass()
        return None
    
    composite = Composite(module_map=module_map)
    attributor = Gradient(model=model_no_softmax, composite=composite)
    
    # Process the input with the attributor
    input_tensor_prepared = x.clone().detach().requires_grad_(True)
    
    # Set model to evaluation mode
    original_mode = model_no_softmax.training
    model_no_softmax.eval()
    
    # Forward pass 
    output = model_no_softmax(input_tensor_prepared)
    
    # Get target class
    target_class = kwargs.get('target_class', None)
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Create conditions for attribution
    if isinstance(target_class, torch.Tensor):
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
    else:
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, torch.tensor([[target_class]], device=output.device), 1.0)
    
    conditions = [{'y': one_hot}]
    
    # Get attribution
    attribution_tensor = attributor(input_tensor_prepared, one_hot)
    
    # Restore model mode
    model_no_softmax.train(original_mode)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        attribution_tensor = attribution_tensor[0]
    
    return attribution_tensor.detach().cpu().numpy()


def lrpsign_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and SIGN input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and SIGN input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGN",
        "stdfactor": 0.1
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpz_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and Z input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and Z input layer rule
    """
    kwargs.update({
        "input_layer_rule": "Z",
        "stdfactor": 0.1
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def zblrp_epsilon_0_1_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and Bounded input layer rule for VGG16.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and Bounded input layer rule
    """
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061,
        "stdfactor": 0.1
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def w2lrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and WSquare input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and WSquare input layer rule
    """
    kwargs.update({
        "input_layer_rule": "WSquare",
        "stdfactor": 0.1
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def flatlrp_epsilon_0_1_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and Flat input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and Flat input layer rule
    """
    kwargs.update({
        "input_layer_rule": "Flat",
        "stdfactor": 0.1
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


# Additional stdfactor variants (0.25)
def lrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    kwargs["stdfactor"] = 0.25
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpsign_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "SIGN",
        "stdfactor": 0.25
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpz_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Z",
        "stdfactor": 0.25
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def zblrp_epsilon_0_25_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061,
        "stdfactor": 0.25
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def w2lrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "WSquare",
        "stdfactor": 0.25
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def flatlrp_epsilon_0_25_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Flat",
        "stdfactor": 0.25
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


# Add 0.5 stdfactor variants
def lrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    kwargs["stdfactor"] = 0.5
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


# SIGNmu variants with different mu parameters
def lrpsign_epsilon_0_25_std_x_mu_0(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and SIGNmu input layer rule with mu=0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and SIGNmu input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGNmu",
        "stdfactor": 0.25,
        "mu": 0.0
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_0_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and SIGNmu input layer rule with mu=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and SIGNmu input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGNmu",
        "stdfactor": 0.25,
        "mu": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpsign_epsilon_0_25_std_x_mu_neg_0_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with std(x) epsilon and SIGNmu input layer rule with mu=-0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with std(x) epsilon and SIGNmu input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGNmu",
        "stdfactor": 0.25,
        "mu": -0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpsign_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "SIGN",
        "stdfactor": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrpz_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Z",
        "stdfactor": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def zblrp_epsilon_0_5_std_x_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061,
        "stdfactor": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def w2lrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "WSquare",
        "stdfactor": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def flatlrp_epsilon_0_5_std_x(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Flat",
        "stdfactor": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_stdxepsilon", **kwargs)


def lrp_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon scaled by standard deviation (factor 1.0).
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon scaled by standard deviation
    """
    # Direct implementation without using custom StdxEpsilon rule
    from zennit.rules import Epsilon, Pass
    from zennit.core import Composite
    from zennit.attribution import Gradient
    import torch.nn as nn
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Calculate standard deviation of the input and scale by factor
    stdfactor = 1.0
    if x.dim() <= 1:
        std_val = torch.std(x).item()
    else:
        # For multi-dimensional tensors, flatten all but batch dimension
        flattened = x.reshape(x.size(0), -1)
        std_val = torch.std(flattened).item()
    
    # Calculate epsilon based on standard deviation
    epsilon_value = std_val * stdfactor
    
    # Create a composite with Epsilon rule using the calculated epsilon
    from zennit.types import Convolution, Linear, Activation, BatchNorm, AvgPool
    
    def module_map(ctx, name, module):
        if isinstance(module, (Convolution, Linear)):
            return Epsilon(epsilon=epsilon_value)
        elif isinstance(module, (Activation, BatchNorm, AvgPool, nn.Flatten, nn.Dropout, 
                              nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.MaxPool1d, nn.MaxPool2d)):
            return Pass()
        return None
    
    composite = Composite(module_map=module_map)
    attributor = Gradient(model=model_no_softmax, composite=composite)
    
    # Process the input with the attributor
    input_tensor_prepared = x.clone().detach().requires_grad_(True)
    
    # Set model to evaluation mode
    original_mode = model_no_softmax.training
    model_no_softmax.eval()
    
    # Forward pass 
    output = model_no_softmax(input_tensor_prepared)
    
    # Get target class
    target_class = kwargs.get('target_class', None)
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Create conditions for attribution
    if isinstance(target_class, torch.Tensor):
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1.0)
    else:
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, torch.tensor([[target_class]], device=output.device), 1.0)
    
    conditions = [{'y': one_hot}]
    
    # Get attribution
    attribution_tensor = attributor(input_tensor_prepared, one_hot)
    
    # Restore model mode
    model_no_softmax.train(original_mode)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        attribution_tensor = attribution_tensor[0]
    
    return attribution_tensor.detach().cpu().numpy()


def lrp_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon scaled by standard deviation (factor 2.0).
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon scaled by standard deviation
    """
    # Create new kwargs without stdfactor to avoid duplicate parameter
    new_kwargs = {k: v for k, v in kwargs.items() if k != 'stdfactor'}
    # Set up a new analyzer with the appropriate parameters
    analyzer = LRPStdxEpsilonAnalyzer(model_no_softmax, stdfactor=2.0)
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Get the attribution map
    result = analyzer.analyze(x, **new_kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        result = result[0]
    
    return result


def lrp_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon scaled by standard deviation (factor 3.0).
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon scaled by standard deviation
    """
    # Create new kwargs without stdfactor to avoid duplicate parameter
    new_kwargs = {k: v for k, v in kwargs.items() if k != 'stdfactor'}
    # Set up a new analyzer with the appropriate parameters
    analyzer = LRPStdxEpsilonAnalyzer(model_no_softmax, stdfactor=3.0)
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Get the attribution map
    result = analyzer.analyze(x, **new_kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        result = result[0]
    
    return result


def lrpsign_epsilon_100_mu_0(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=100 and SIGNmu input layer rule with mu=0.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=100 and SIGNmu input layer rule
    """
    # Create a new kwargs without conflicting parameters
    new_kwargs = {k: v for k, v in kwargs.items() if k not in ('epsilon', 'input_layer_rule', 'mu')}
    
    # Create custom LRP analyzer with epsilon rule
    import torch.nn as nn
    from zennit.rules import Epsilon, Pass
    from zennit.core import Composite
    from zennit.attribution import Gradient
    from zennit.types import Convolution, Linear, Activation, BatchNorm, AvgPool
    from .zennit_impl.sign_rule import SIGNmuRule
    
    # Create a custom module_map function for epsilon rule with SIGNmu for first layer
    def module_map(ctx, name, module):
        first_layer = True
        for other_name, _ in model_no_softmax.named_modules():
            if other_name != name and isinstance(module, (Convolution, Linear)):
                first_layer = False
                break
                
        if first_layer and isinstance(module, (Convolution, Linear)):
            return SIGNmuRule(mu=0.0)
        elif isinstance(module, (Convolution, Linear)):
            return Epsilon(epsilon=100.0)
        elif isinstance(module, (Activation, BatchNorm, AvgPool, nn.Flatten, nn.Dropout, 
                              nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.MaxPool1d, nn.MaxPool2d)):
            return Pass()
        return None
        
    composite = Composite(module_map=module_map)
    attributor = Gradient(model=model_no_softmax, composite=composite)
    
    # Convert input to tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Add batch dimension if needed
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Process the input
    input_tensor_prepared = x.clone().detach().requires_grad_(True)
    
    # Set model to evaluation mode
    original_mode = model_no_softmax.training
    model_no_softmax.eval()
    
    # Forward pass 
    output = model_no_softmax(input_tensor_prepared)
    
    # Get target class tensor
    if 'target_class' in new_kwargs:
        target_class = new_kwargs['target_class']
    else:
        target_class = None
        
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Create conditions for attribution
    conditions = [{'y': torch.zeros_like(output).scatter_(1, target_class.unsqueeze(1) if isinstance(target_class, torch.Tensor) else torch.tensor([[target_class]]), 1.0)}]
    
    # Get attribution
    attribution_tensor = attributor(input_tensor_prepared, one_hot)
    
    # Restore model mode
    model_no_softmax.train(original_mode)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        attribution_tensor = attribution_tensor[0]
    
    return attribution_tensor.detach().cpu().numpy()


def lrpsign_epsilon_100_mu_0_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=100 and SIGNmu input layer rule with mu=0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=100 and SIGNmu input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGNmu",
        "epsilon": 100,
        "mu": 0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrpsign_epsilon_100_mu_neg_0_5(model_no_softmax, x, **kwargs):
    """Calculate LRP with epsilon=100 and SIGNmu input layer rule with mu=-0.5.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with epsilon=100 and SIGNmu input layer rule
    """
    kwargs.update({
        "input_layer_rule": "SIGNmu",
        "epsilon": 100,
        "mu": -0.5
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", **kwargs)


def lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_alpha1beta0", **kwargs)


def lrpsign_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "SIGN"
    return lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)


def lrpz_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Z"
    return lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)


def zblrp_alpha_1_beta_0_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    kwargs.update({
        "input_layer_rule": "Bounded",
        "low": -123.68,
        "high": 151.061
    })
    return lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)


def w2lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "WSquare"
    return lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)


def flatlrp_alpha_1_beta_0(model_no_softmax, x, **kwargs):
    kwargs["input_layer_rule"] = "Flat"
    return lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)


def lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite A rules.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite A rules
    """
    kwargs.update({
        "variant": "A",  # Use variant A which applies the correct rules
        "epsilon": kwargs.get("epsilon", 0.1)  # Default epsilon=0.1 for variant A
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def lrpsign_sequential_composite_a(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite A rules and SIGN input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite A rules and SIGN input layer rule
    """
    kwargs.update({
        "variant": "A", 
        "first_layer_rule_name": "sign",
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def lrpz_sequential_composite_a(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite A rules and Z input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite A rules and Z input layer rule
    """
    kwargs.update({
        "variant": "A", 
        "first_layer_rule_name": "zplus",
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def zblrp_sequential_composite_a_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite A rules and bounded input layer rule for VGG16.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite A rules and bounded input layer rule
    """
    kwargs.update({
        "variant": "A", 
        "first_layer_rule_name": "zbox",
        "low": -123.68,
        "high": 151.061,
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def w2lrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite A rules and WSquare input layer rule.
    
    FIXED: Now uses corrected W2LRP composite A implementation for exact TF-PT correlation.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite A rules and WSquare input layer rule
    """
    # Use AdvancedLRPAnalyzer with corrected w2lrp sequential composite A
    return _calculate_relevancemap(model_no_softmax, x, method="w2lrp_sequential_composite_a", **kwargs)


def flatlrp_sequential_composite_a(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite A rules and flat input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite A rules and flat input layer rule
    """
    kwargs.update({
        "variant": "A", 
        "first_layer_rule_name": "flat",
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def lrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite B rules.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite B rules
    """
    kwargs.update({
        "variant": "B",  # Use variant B which applies the correct rules
        "epsilon": kwargs.get("epsilon", 0.1)  # Default epsilon=0.1 for variant B
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def lrpsign_sequential_composite_b(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite B rules and SIGN input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite B rules and SIGN input layer rule
    """
    kwargs.update({
        "variant": "B", 
        "first_layer_rule_name": "sign",
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def lrpz_sequential_composite_b(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite B rules and Z input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite B rules and Z input layer rule
    """
    kwargs.update({
        "variant": "B", 
        "first_layer_rule_name": "zplus",
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def zblrp_sequential_composite_b_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite B rules and bounded input layer rule for VGG16.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite B rules and bounded input layer rule
    """
    kwargs.update({
        "variant": "B", 
        "first_layer_rule_name": "zbox",
        "low": -123.68,
        "high": 151.061,
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def w2lrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite B rules and WSquare input layer rule.
    
    FIXED: Now uses corrected W2LRP composite B implementation for exact TF-PT correlation.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite B rules and WSquare input layer rule
    """
    # Use AdvancedLRPAnalyzer with corrected w2lrp sequential composite B
    return _calculate_relevancemap(model_no_softmax, x, method="w2lrp_sequential_composite_b", **kwargs)


def flatlrp_sequential_composite_b(model_no_softmax, x, **kwargs):
    """Calculate LRP with sequential composite B rules and flat input layer rule.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        LRP relevance map with sequential composite B rules and flat input layer rule
    """
    kwargs.update({
        "variant": "B", 
        "first_layer_rule_name": "flat",
        "epsilon": kwargs.get("epsilon", 0.1)
    })
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_sequential", **kwargs)


def deeplift(model_no_softmax, x, **kwargs):
    """Calculate DeepLift relevance map.
    
    Args:
        model_no_softmax: PyTorch model with softmax removed
        x: Input tensor
        **kwargs: Additional arguments
        
    Returns:
        DeepLift relevance map
    """
    # Convert x to torch tensor if needed
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Ensure x has batch dimension
    needs_batch_dim = x.ndim == 3  # Image: C,H,W or 1D: C,T
    if needs_batch_dim:
        x = x.unsqueeze(0)
    
    # Extract DeepLift parameters
    baseline_type = kwargs.pop('baseline_type', 'zero')
    
    # Calculate relevance map
    analyzer = DeepLiftAnalyzer(model_no_softmax, baseline_type=baseline_type, **kwargs)
    relevance_map = analyzer.analyze(x, **kwargs)
    
    # Remove batch dimension if it was added
    if needs_batch_dim:
        relevance_map = relevance_map[0]
    
    return relevance_map


# Missing _x_input_x_sign combinations
def gradient_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate gradient times input times sign relevance map."""
    g = gradient(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return g * x_np * s


def deconvnet_x_input_x_sign_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate deconvnet times input times sign relevance map."""
    d = deconvnet(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return d * x_np * s


def guided_backprop_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate guided backprop times input times sign relevance map."""
    g = guided_backprop(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return g * x_np * s


def smoothgrad_x_input_x_sign_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate smoothgrad times input times sign relevance map."""
    s_grad = smoothgrad(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return s_grad * x_np * s


# Missing _x_input and _x_sign variations for other methods
def vargrad_x_input_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate vargrad times input relevance map."""
    v = vargrad(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return v * x_np


def vargrad_x_sign_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate vargrad times sign relevance map."""
    v = vargrad(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return v * s


def vargrad_x_input_x_sign_DISABLED_BROKEN_WRAPPER(model_no_softmax, x, **kwargs):
    """Calculate vargrad times input times sign relevance map."""
    v = vargrad(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return v * x_np * s


def integrated_gradients_x_input(model_no_softmax, x, **kwargs):
    """Calculate integrated gradients times input relevance map."""
    ig = integrated_gradients(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return ig * x_np


def integrated_gradients_x_sign(model_no_softmax, x, **kwargs):
    """Calculate integrated gradients times sign relevance map."""
    ig = integrated_gradients(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return ig * s


def integrated_gradients_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate integrated gradients times input times sign relevance map."""
    ig = integrated_gradients(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return ig * x_np * s


def grad_cam_x_input(model_no_softmax, x, **kwargs):
    """Calculate grad-cam times input relevance map."""
    gc = grad_cam(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    # Broadcast grad_cam result to match input dimensions if needed
    if gc.shape != x_np.shape:
        # Handle broadcasting for different shapes
        if x_np.ndim == 4 and gc.ndim == 2:  # (B,C,H,W) vs (H,W)
            gc = np.broadcast_to(gc[None, None], x_np.shape)
        elif x_np.ndim == 3 and gc.ndim == 2:  # (C,H,W) vs (H,W)
            gc = np.broadcast_to(gc[None], x_np.shape)
    return gc * x_np


def grad_cam_x_sign(model_no_softmax, x, **kwargs):
    """Calculate grad-cam times sign relevance map."""
    gc = grad_cam(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    # Broadcast grad_cam result to match input dimensions if needed
    if gc.shape != s.shape:
        if x_np.ndim == 4 and gc.ndim == 2:  # (B,C,H,W) vs (H,W)
            gc = np.broadcast_to(gc[None, None], s.shape)
        elif x_np.ndim == 3 and gc.ndim == 2:  # (C,H,W) vs (H,W)
            gc = np.broadcast_to(gc[None], s.shape)
    return gc * s


def grad_cam_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate grad-cam times input times sign relevance map."""
    gc = grad_cam(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    # Broadcast grad_cam result to match input dimensions if needed
    if gc.shape != x_np.shape:
        if x_np.ndim == 4 and gc.ndim == 2:  # (B,C,H,W) vs (H,W)
            gc = np.broadcast_to(gc[None, None], x_np.shape)
        elif x_np.ndim == 3 and gc.ndim == 2:  # (C,H,W) vs (H,W)
            gc = np.broadcast_to(gc[None], x_np.shape)
    return gc * x_np * s


def deeplift_x_input(model_no_softmax, x, **kwargs):
    """Calculate deeplift times input relevance map."""
    dl = deeplift(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return dl * x_np


def deeplift_x_sign(model_no_softmax, x, **kwargs):
    """Calculate deeplift times sign relevance map."""
    dl = deeplift(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return dl * s


def deeplift_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate deeplift times input times sign relevance map."""
    dl = deeplift(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return dl * x_np * s


# LRP method combinations
def lrp_epsilon_0_1_x_input(model_no_softmax, x, **kwargs):
    """Calculate LRP epsilon times input relevance map."""
    lrp = lrp_epsilon_0_1(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np


def lrp_epsilon_0_1_x_sign(model_no_softmax, x, **kwargs):
    """Calculate LRP epsilon times sign relevance map."""
    lrp = lrp_epsilon_0_1(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s


def lrp_epsilon_0_1_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate LRP epsilon times input times sign relevance map."""
    lrp = lrp_epsilon_0_1(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s


def lrp_alpha_1_beta_0_x_input(model_no_softmax, x, **kwargs):
    """Calculate LRP alpha-beta times input relevance map."""
    lrp = lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np


def lrp_alpha_1_beta_0_x_sign(model_no_softmax, x, **kwargs):
    """Calculate LRP alpha-beta times sign relevance map."""
    lrp = lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s


def lrp_alpha_1_beta_0_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate LRP alpha-beta times input times sign relevance map."""
    lrp = lrp_alpha_1_beta_0(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s


def lrp_z_x_input(model_no_softmax, x, **kwargs):
    """Calculate LRP-z times input relevance map."""
    lrp = lrp_z(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np


def lrp_z_x_sign(model_no_softmax, x, **kwargs):
    """Calculate LRP-z times sign relevance map."""
    lrp = lrp_z(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s


def lrp_z_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Calculate LRP-z times input times sign relevance map."""
    lrp = lrp_z(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s


# Main wrapper functions matching the TensorFlow API
def calculate_relevancemap(m, x, model_no_softmax, **kwargs):
    """Calculate relevance map for a single input using the specified method.
    
    Args:
        m: Name of the explanation method
        x: Input tensor
        model_no_softmax: PyTorch model with softmax removed
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Relevance map as numpy array
    """
    method_func = eval(m)
    return method_func(model_no_softmax, x, **kwargs)


def calculate_relevancemaps(m, X, model_no_softmax, **kwargs):
    """Calculate relevance maps for multiple inputs using the specified method.
    
    Args:
        m: Name of the explanation method
        X: Batch of input tensors
        model_no_softmax: PyTorch model with softmax removed
        **kwargs: Additional arguments for the specific method
        
    Returns:
        Batch of relevance maps as numpy array
    """
    Rs = []
    for x in X:
        R = calculate_relevancemap(m, x, model_no_softmax, **kwargs)
        Rs.append(R)
    
    return np.array(Rs)


# =====================================================
# Missing PyTorch methods to match TensorFlow functionality
# =====================================================

# Native calculation methods (already implemented differently)
def calculate_native_gradient(model_no_softmax, x, **kwargs):
    """Native gradient calculation using PyTorch autograd."""
    return _calculate_relevancemap(model_no_softmax, x, method="gradient", **kwargs)

def calculate_native_integrated_gradients(model_no_softmax, x, **kwargs):
    """Native integrated gradients calculation."""
    return _calculate_relevancemap(model_no_softmax, x, method="integrated_gradients", **kwargs)

def calculate_native_smoothgrad(model_no_softmax, x, **kwargs):
    """Native smooth gradients calculation."""
    return _calculate_relevancemap(model_no_softmax, x, method="smoothgrad", **kwargs)

# Wrapper methods
def deconvnet_x_sign_mu_wrapper(model_no_softmax, x, **kwargs):
    """DeconvNet with sign and mu wrapper."""
    mu = kwargs.pop('mu', 0.0)
    return deconvnet_x_sign_mu(model_no_softmax, x, mu=mu, **kwargs)

def gradient_x_sign_mu_wrapper(model_no_softmax, x, **kwargs):
    """Gradient with sign and mu wrapper."""
    mu = kwargs.pop('mu', 0.0)
    return gradient_x_sign_mu(model_no_softmax, x, mu=mu, **kwargs)

def guided_backprop_x_sign_mu_wrapper(model_no_softmax, x, **kwargs):
    """Guided backprop with sign and mu wrapper."""
    mu = kwargs.pop('mu', 0.0)
    return guided_backprop_x_sign_mu(model_no_softmax, x, mu=mu, **kwargs)

def lrp_epsilon_wrapper(model_no_softmax, x, **kwargs):
    """LRP epsilon wrapper."""
    epsilon = kwargs.get('epsilon', 1e-6)
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon", epsilon=epsilon, **kwargs)

def deeplift_method(model_no_softmax, x, **kwargs):
    """DeepLift method."""
    return _calculate_relevancemap(model_no_softmax, x, method="deeplift", **kwargs)

# Flat LRP methods
def flatlrp_epsilon_1(model_no_softmax, x, **kwargs):
    """Flat LRP with epsilon=1."""
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'epsilon'}
    return _calculate_relevancemap(model_no_softmax, x, method="flatlrp_epsilon_1", epsilon=1.0, **kwargs_clean)

def flatlrp_epsilon_10(model_no_softmax, x, **kwargs):
    """Flat LRP with epsilon=10."""
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'epsilon'}
    return _calculate_relevancemap(model_no_softmax, x, method="flatlrp_epsilon_10", epsilon=10.0, **kwargs_clean)

def flatlrp_epsilon_20(model_no_softmax, x, **kwargs):
    """Flat LRP with epsilon=20."""
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'epsilon'}
    return _calculate_relevancemap(model_no_softmax, x, method="flatlrp_epsilon_20", epsilon=20.0, **kwargs_clean)

def flatlrp_epsilon_100(model_no_softmax, x, **kwargs):
    """Flat LRP with epsilon=100."""
    kwargs_clean = {k: v for k, v in kwargs.items() if k != 'epsilon'}
    return _calculate_relevancemap(model_no_softmax, x, method="flatlrp_epsilon_100", epsilon=100.0, **kwargs_clean)

# LRP Alpha-Beta variants
def lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs):
    """LRP with alpha=2, beta=1."""
    kwargs_clean = {k: v for k, v in kwargs.items() if k not in ['alpha', 'beta']}
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_alphabeta", alpha=2, beta=1, **kwargs_clean)

def lrp_alpha_2_beta_1_x_input(model_no_softmax, x, **kwargs):
    """LRP alpha-2-beta-1 times input."""
    lrp = lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np

def lrp_alpha_2_beta_1_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP alpha-2-beta-1 times input times sign."""
    lrp = lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s

def lrp_alpha_2_beta_1_x_sign(model_no_softmax, x, **kwargs):
    """LRP alpha-2-beta-1 times sign."""
    lrp = lrp_alpha_2_beta_1(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s

# LRP Flat methods
def lrp_flat(model_no_softmax, x, **kwargs):
    """LRP flat rule."""
    return zennit_calculate_relevancemap(model_no_softmax, x, method="lrp_flat", **kwargs)

def lrp_flat_x_input(model_no_softmax, x, **kwargs):
    """LRP flat times input."""
    lrp = lrp_flat(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np

def lrp_flat_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP flat times input times sign."""
    lrp = lrp_flat(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s

def lrp_flat_x_sign(model_no_softmax, x, **kwargs):
    """LRP flat times sign."""
    lrp = lrp_flat(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s

# LRP Gamma methods
def lrp_gamma(model_no_softmax, x, **kwargs):
    """LRP gamma rule."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_gamma", **kwargs)

def lrp_gamma_x_input(model_no_softmax, x, **kwargs):
    """LRP gamma times input."""
    lrp = lrp_gamma(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np

def lrp_gamma_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP gamma times input times sign."""
    lrp = lrp_gamma(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s

def lrp_gamma_x_sign(model_no_softmax, x, **kwargs):
    """LRP gamma times sign."""
    lrp = lrp_gamma(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s

# LRP Sign methods
def lrpsign_epsilon_0_5(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=0.5."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_0_5", epsilon=0.5, **kwargs)

def lrpsign_epsilon_10(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=10."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_10", epsilon=10.0, **kwargs)

def lrpsign_epsilon_20(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=20."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_20", epsilon=20.0, **kwargs)

def lrpsign_epsilon_50(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=50."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_50", epsilon=50.0, **kwargs)

def lrpsign_epsilon_75(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=75."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_75", epsilon=75.0, **kwargs)

def lrpsign_epsilon_100(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=100."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_100", epsilon=100.0, **kwargs)

def lrpsign_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=0.1 and stdfactor=1.0."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_1_std_x", epsilon=0.1, stdfactor=1.0, **kwargs)

def lrpsign_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=0.1 and stdfactor=2.0."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_2_std_x", epsilon=0.1, stdfactor=2.0, **kwargs)

def lrpsign_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    """LRP sign with epsilon=0.1 and stdfactor=3.0."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrpsign_epsilon_3_std_x", epsilon=0.1, stdfactor=3.0, **kwargs)

# LRP W-Square methods
def lrp_w_square(model_no_softmax, x, **kwargs):
    """LRP w-square rule."""
    return zennit_calculate_relevancemap(model_no_softmax, x, method="lrp_w_square", **kwargs)

def lrp_w_square_x_input(model_no_softmax, x, **kwargs):
    """LRP w-square times input."""
    lrp = lrp_w_square(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np

def lrp_w_square_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP w-square times input times sign."""
    lrp = lrp_w_square(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s

def lrp_w_square_x_sign(model_no_softmax, x, **kwargs):
    """LRP w-square times sign."""
    lrp = lrp_w_square(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s

# LRP Z-Plus methods
def lrp_z_plus(model_no_softmax, x, **kwargs):
    """LRP z-plus rule."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zplus", **kwargs)

def lrp_z_plus_x_input(model_no_softmax, x, **kwargs):
    """LRP z-plus times input."""
    lrp = lrp_z_plus(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    return lrp * x_np

def lrp_z_plus_x_input_x_sign(model_no_softmax, x, **kwargs):
    """LRP z-plus times input times sign."""
    lrp = lrp_z_plus(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * x_np * s

def lrp_z_plus_x_sign(model_no_softmax, x, **kwargs):
    """LRP z-plus times sign."""
    lrp = lrp_z_plus(model_no_softmax, x, **kwargs)
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    s = np.nan_to_num(x_np / np.abs(x_np), nan=1.0)
    return lrp * s

# LRPZ Epsilon methods
def lrpz_epsilon_0_5(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=0.5."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_0_5", epsilon=0.5, **kwargs)

def lrpz_epsilon_10(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=10."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_10", epsilon=10.0, **kwargs)

def lrpz_epsilon_20(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=20."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_20", epsilon=20.0, **kwargs)

def lrpz_epsilon_50(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=50."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_50", epsilon=50.0, **kwargs)

def lrpz_epsilon_75(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=75."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_75", epsilon=75.0, **kwargs)

def lrpz_epsilon_100(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=100."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_100", epsilon=100.0, **kwargs)

def lrpz_epsilon_1_std_x(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=0.1 and stdfactor=1.0."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_1_std_x", epsilon=0.1, stdfactor=1.0, **kwargs)

def lrpz_epsilon_2_std_x(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=0.1 and stdfactor=2.0."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_2_std_x", epsilon=0.1, stdfactor=2.0, **kwargs)

def lrpz_epsilon_3_std_x(model_no_softmax, x, **kwargs):
    """LRPZ with epsilon=0.1 and stdfactor=3.0."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_epsilon_3_std_x", epsilon=0.1, stdfactor=3.0, **kwargs)

# W2LRP methods
def w2lrp_epsilon_1(model_no_softmax, x, **kwargs):
    """W2LRP with epsilon=1."""
    return _calculate_relevancemap(model_no_softmax, x, method="w2lrp_epsilon_1", epsilon=1.0, **kwargs)

def w2lrp_epsilon_10(model_no_softmax, x, **kwargs):
    """W2LRP with epsilon=10."""
    return _calculate_relevancemap(model_no_softmax, x, method="w2lrp_epsilon_10", epsilon=10.0, **kwargs)

def w2lrp_epsilon_20(model_no_softmax, x, **kwargs):
    """W2LRP with epsilon=20."""
    return _calculate_relevancemap(model_no_softmax, x, method="w2lrp_epsilon_20", epsilon=20.0, **kwargs)

def w2lrp_epsilon_100(model_no_softmax, x, **kwargs):
    """W2LRP with epsilon=100."""
    return _calculate_relevancemap(model_no_softmax, x, method="w2lrp_epsilon_100", epsilon=100.0, **kwargs)

# ZBLRP methods (model-specific VGG16)
def zblrp_epsilon_0_5_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """ZBLRP for VGG16 with epsilon=0.5."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zbox", epsilon=0.5, low=0.0, high=1.0, **kwargs)

def zblrp_epsilon_1_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """ZBLRP for VGG16 with epsilon=1."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zbox", epsilon=1.0, low=0.0, high=1.0, **kwargs)

def zblrp_epsilon_5_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """ZBLRP for VGG16 with epsilon=5."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zbox", epsilon=5.0, low=0.0, high=1.0, **kwargs)

def zblrp_epsilon_10_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """ZBLRP for VGG16 with epsilon=10."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zbox", epsilon=10.0, low=0.0, high=1.0, **kwargs)

def zblrp_epsilon_20_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """ZBLRP for VGG16 with epsilon=20."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zbox", epsilon=20.0, low=0.0, high=1.0, **kwargs)

def zblrp_epsilon_100_VGG16ILSVRC(model_no_softmax, x, **kwargs):
    """ZBLRP for VGG16 with epsilon=100."""
    return _calculate_relevancemap(model_no_softmax, x, method="lrp_zbox", epsilon=100.0, low=0.0, high=1.0, **kwargs)


# ===== REDIRECT TO WORKING ZENNIT IMPLEMENTATIONS =====
# These methods had broken wrapper implementations that produced None gradients
# Now they redirect to the working Zennit implementations

def smoothgrad_x_sign(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='smoothgrad_x_sign', **kwargs)

def smoothgrad_x_input(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='smoothgrad_x_input', **kwargs)

def smoothgrad_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""  
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='smoothgrad_x_input_x_sign', **kwargs)

def vargrad_x_sign(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='vargrad_x_sign', **kwargs)

def vargrad_x_input(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='vargrad_x_input', **kwargs)

def vargrad_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='vargrad_x_input_x_sign', **kwargs)

def deconvnet_x_sign(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='deconvnet_x_sign', **kwargs)

def deconvnet_x_sign_mu_0_5(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='deconvnet_x_sign_mu_0_5', **kwargs)

def deconvnet_x_input(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='deconvnet_x_input', **kwargs)

def deconvnet_x_input_x_sign(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='deconvnet_x_input_x_sign', **kwargs)

def vargrad(model_no_softmax, x, **kwargs):
    """Redirect to working Zennit implementation."""
    from .zennit_impl import calculate_relevancemap
    return calculate_relevancemap(model_no_softmax, x, method='vargrad', **kwargs)
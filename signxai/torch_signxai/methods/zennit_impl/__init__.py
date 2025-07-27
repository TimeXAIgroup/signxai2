"""
Zennit-based implementation details for PyTorch XAI methods.
This subpackage relies on the Zennit library.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, Optional, Dict, Any, Type

# Import specific analyzers that are defined in analyzers.py within this package
from .analyzers import (
    AnalyzerBase, # Import the base class for type hinting
    GradientAnalyzer,
    SmoothGradAnalyzer,
    IntegratedGradientsAnalyzer,
    LRPAnalyzer, # This is the generic LRPAnalyzer from analyzers.py
    GuidedBackpropAnalyzer,
    DeconvNetAnalyzer,
    GradCAMAnalyzer,
    GradientXSignAnalyzer,
    GradientXInputAnalyzer,
    VarGradAnalyzer,
    DeepTaylorAnalyzer,
)

# Import additional Zennit composites
import zennit.composites as zcomposites

# Import the custom rules
from .stdx_rule import StdxEpsilon
from .sign_rule import SIGNRule, SIGNmuRule

# Import the advanced LRP analyzers from lrp_variants.py
from .lrp_variants import AdvancedLRPAnalyzer, LRPSequential, BoundedLRPAnalyzer, DeepLiftAnalyzer, LRPStdxEpsilonAnalyzer

# SUPPORTED_ZENNIT_METHODS dictionary
SUPPORTED_ZENNIT_METHODS: Dict[str, Type[AnalyzerBase]] = { # Use AnalyzerBase for correct type hint
    "gradient": GradientAnalyzer,
    "integrated_gradients": IntegratedGradientsAnalyzer,
    "smoothgrad": SmoothGradAnalyzer,
    "guided_backprop": GuidedBackpropAnalyzer,
    "deconvnet": DeconvNetAnalyzer,
    "grad_cam": GradCAMAnalyzer,
    "gradient_x_sign": GradientXSignAnalyzer,
    "gradient_x_input": GradientXInputAnalyzer,
    "vargrad": VarGradAnalyzer,
    "deeptaylor": DeepTaylorAnalyzer,
    "lrp": LRPAnalyzer, # Basic LRP
    # The following specific LRP methods will now use AdvancedLRPAnalyzer
    # by passing the 'variant' parameter.
    "lrp.epsilon": LRPAnalyzer,
    "lrp_epsilon": AdvancedLRPAnalyzer,  # For epsilon methods called without dot
    # Specific LRP epsilon variants
    "lrp_epsilon_0_001": AdvancedLRPAnalyzer,
    "lrp_epsilon_0_01": AdvancedLRPAnalyzer,
    "lrp_epsilon_0_1": AdvancedLRPAnalyzer,
    "lrp_epsilon_0_2": AdvancedLRPAnalyzer,
    "lrp_epsilon_0_25": AdvancedLRPAnalyzer,
    "lrp_epsilon_0_5": AdvancedLRPAnalyzer,
    "lrp_epsilon_1": AdvancedLRPAnalyzer,
    "lrp_epsilon_2": AdvancedLRPAnalyzer,
    "lrp_epsilon_3": AdvancedLRPAnalyzer,
    "lrp_epsilon_5": AdvancedLRPAnalyzer,
    "lrp_epsilon_10": AdvancedLRPAnalyzer,
    "lrp_epsilon_20": AdvancedLRPAnalyzer,
    "lrp_epsilon_50": AdvancedLRPAnalyzer,
    "lrp_epsilon_75": AdvancedLRPAnalyzer,
    "lrp_epsilon_100": AdvancedLRPAnalyzer,
    "lrp.zplus": LRPAnalyzer,
    "lrp.alphabeta": LRPAnalyzer, # Use basic LRPAnalyzer for alpha-beta
    "lrp_alpha_1_beta_0": "WrapperDelegation", # Specific alpha=1, beta=0 variant
    "lrpsign_alpha_1_beta_0": "WrapperDelegation", # LRP Alpha1Beta0 with SIGN input layer rule
    "lrpsign_epsilon_0_1": "WrapperDelegation", # LRP Epsilon with SIGN input layer rule
    "lrp_alpha_2_beta_1": "WrapperDelegation", # Specific alpha=2, beta=1 variant
    "lrp_alpha_2_beta_1_x_input": "WrapperDelegation", # Alpha2Beta1 times input
    "lrp_alpha_2_beta_1_x_input_x_sign": "WrapperDelegation", # Alpha2Beta1 times input times sign
    "lrp_alpha_2_beta_1_x_sign": "WrapperDelegation", # Alpha2Beta1 times sign
    "lrp.gamma": AdvancedLRPAnalyzer,
    "lrp.flat": LRPAnalyzer,
    "lrp.wsquare": AdvancedLRPAnalyzer,
    "lrp.zbox": AdvancedLRPAnalyzer,
    "lrp.bounded": BoundedLRPAnalyzer, # New bounded LRP
    "lrp.stdxepsilon": LRPStdxEpsilonAnalyzer, # Standard deviation based epsilon LRP
    "deeplift": DeepLiftAnalyzer, # DeepLift implementation
    "advanced_lrp": AdvancedLRPAnalyzer, # A more generic key for AdvancedLRPAnalyzer
    "lrp_sequential": LRPSequential,   # For the LRPSequential analyzer
    
    # === MISSING METHODS FROM PYTORCH FAILURES ===
    # FlatLRP variants - missing methods causing negative correlations
    "flatlrp_epsilon_1": "WrapperDelegation",
    "flatlrp_epsilon_10": "WrapperDelegation",
    "flatlrp_epsilon_20": "WrapperDelegation",
    "flatlrp_epsilon_100": "WrapperDelegation",
    "flatlrp_alpha_1_beta_0": "WrapperDelegation",
    "w2lrp_alpha_1_beta_0": "WrapperDelegation",
    "flatlrp_epsilon_0_1": "WrapperDelegation",
    "flatlrp_epsilon_0_01": "WrapperDelegation",
    "flatlrp_epsilon_0_1_std_x": "WrapperDelegation",
    "flatlrp_sequential_composite_a": "WrapperDelegation",
    "flatlrp_sequential_composite_b": "WrapperDelegation",
    "flatlrp_z": "WrapperDelegation",
    
    # FlatLRP variants with std_x - REMOVED: conflicts with WrapperDelegation on line 73
    # "flatlrp_epsilon_0_1_std_x": LRPStdxEpsilonAnalyzer,
    "flatlrp_epsilon_0_25_std_x": "WrapperDelegation",
    "flatlrp_epsilon_0_5_std_x": "WrapperDelegation",
    
    # GradCAM variants
    "grad_cam_VGG16ILSVRC": GradCAMAnalyzer,
    "grad_cam_x_input": GradCAMAnalyzer,
    "grad_cam_x_input_x_sign": GradCAMAnalyzer,
    "grad_cam_x_sign": GradCAMAnalyzer,
    
    # Gradient variants with mu parameters
    "gradient_x_sign_mu_0": GradientXSignAnalyzer,
    "gradient_x_sign_mu_0_5": GradientXSignAnalyzer,
    "gradient_x_sign_mu_neg_0_5": GradientXSignAnalyzer,
    
    # Guided GradCAM variants
    "guided_grad_cam_VGG16ILSVRC": GradCAMAnalyzer,
    
    # LRP variants with std_x
    "lrp_epsilon_0_1_std_x": LRPStdxEpsilonAnalyzer,
    "lrp_epsilon_0_25_std_x": LRPStdxEpsilonAnalyzer,
    "lrp_epsilon_0_5_std_x": LRPStdxEpsilonAnalyzer,
    "lrp_epsilon_1_std_x": LRPStdxEpsilonAnalyzer,
    "lrp_epsilon_2_std_x": LRPStdxEpsilonAnalyzer,
    "lrp_epsilon_3_std_x": LRPStdxEpsilonAnalyzer,
    
    # LRPSign variants
    "lrpsign_epsilon_0_1_std_x": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_0_25_std_x": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_0_25_std_x_mu_0": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_0_25_std_x_mu_0_5": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_0_25_std_x_mu_neg_0_5": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_0_5": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_0_5_std_x": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_1": "WrapperDelegation",
    "lrpsign_epsilon_10": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_100": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_100_mu_0": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_1_std_x": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_20": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_2_std_x": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_3_std_x": LRPStdxEpsilonAnalyzer,
    "lrpsign_epsilon_5": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_50": AdvancedLRPAnalyzer,
    "lrpsign_epsilon_75": AdvancedLRPAnalyzer,
    
    # LRPSign Z variant
    "lrpsign_z": "WrapperDelegation",
    
    # LRPSign sequential composite variants
    "lrpsign_sequential_composite_a": "WrapperDelegation",
    
    # LRPZ variants
    "lrpz_alpha_1_beta_0": "WrapperDelegation",
    "lrpz_epsilon_0_1": AdvancedLRPAnalyzer,
    "lrpz_epsilon_0_1_std_x": LRPStdxEpsilonAnalyzer,
    "lrpz_epsilon_0_25_std_x": LRPStdxEpsilonAnalyzer,
    "lrpz_epsilon_0_5": AdvancedLRPAnalyzer,
    "lrpz_epsilon_0_5_std_x": LRPStdxEpsilonAnalyzer,
    "lrpz_epsilon_1": AdvancedLRPAnalyzer,
    "lrpz_epsilon_1_std_x": LRPStdxEpsilonAnalyzer,
    "lrpz_epsilon_2_std_x": LRPStdxEpsilonAnalyzer,
    "lrpz_epsilon_3_std_x": LRPStdxEpsilonAnalyzer,
    "lrpz_epsilon_10": AdvancedLRPAnalyzer,
    "lrpz_epsilon_20": AdvancedLRPAnalyzer,
    "lrpz_epsilon_75": AdvancedLRPAnalyzer,
    
    # SmoothGrad variants (missing x_sign base methods)
    "smoothgrad_x_sign": SmoothGradAnalyzer,
    "smoothgrad_x_input": SmoothGradAnalyzer,
    "smoothgrad_x_input_x_sign": SmoothGradAnalyzer,
    
    # SmoothGrad variants with mu parameters
    "smoothgrad_x_sign_mu_0": SmoothGradAnalyzer,
    "smoothgrad_x_sign_mu_0_5": SmoothGradAnalyzer,
    "smoothgrad_x_sign_mu_neg_0_5": SmoothGradAnalyzer,
    
    # VarGrad variants (missing x_sign base methods)
    "vargrad_x_sign": "tf_exact_vargrad_x_sign",
    "vargrad_x_input": "tf_exact_vargrad_x_input",
    "vargrad_x_input_x_sign": "tf_exact_vargrad_x_input_x_sign",
    
    # DeconvNet variants with mu parameters (missing)
    "deconvnet_x_sign": DeconvNetAnalyzer,
    "deconvnet_x_sign_mu_0_5": DeconvNetAnalyzer,
    "deconvnet_x_input": DeconvNetAnalyzer,
    "deconvnet_x_input_x_sign": DeconvNetAnalyzer,
    "guided_backprop_x_sign_mu_0_5": GuidedBackpropAnalyzer,
    "guided_backprop_x_input": GuidedBackpropAnalyzer,
    "guided_backprop_x_input_x_sign": GuidedBackpropAnalyzer,
    
    # LRP flat and w_square variants (underscore versions)
    "lrp_flat": AdvancedLRPAnalyzer,
    "lrp_flat_x_input": AdvancedLRPAnalyzer, 
    "lrp_flat_x_sign": AdvancedLRPAnalyzer,
    "lrp_flat_x_input_x_sign": AdvancedLRPAnalyzer,
    "lrp_w_square": AdvancedLRPAnalyzer,
    "lrp_w_square_x_input": AdvancedLRPAnalyzer,
    "lrp_w_square_x_sign": AdvancedLRPAnalyzer, 
    "lrp_w_square_x_input_x_sign": AdvancedLRPAnalyzer,
    
    # W2LRP variants
    "w2lrp_epsilon_0_1": AdvancedLRPAnalyzer,
    "w2lrp_epsilon_0_1_std_x": LRPStdxEpsilonAnalyzer,
    "w2lrp_epsilon_0_25_std_x": LRPStdxEpsilonAnalyzer,
    "w2lrp_epsilon_0_5_std_x": LRPStdxEpsilonAnalyzer,
    "w2lrp_epsilon_1": AdvancedLRPAnalyzer,
    "w2lrp_epsilon_10": AdvancedLRPAnalyzer,
    "w2lrp_epsilon_100": AdvancedLRPAnalyzer,
    "w2lrp_epsilon_20": AdvancedLRPAnalyzer,
    "w2lrp_sequential_composite_a": LRPSequential,
    "w2lrp_sequential_composite_b": LRPSequential,
    
    # ZBLRP variants (VGG16 specific)
    "zblrp_epsilon_0_1_std_x_VGG16ILSVRC": LRPStdxEpsilonAnalyzer,
    "zblrp_epsilon_0_25_std_x_VGG16ILSVRC": LRPStdxEpsilonAnalyzer,
    "zblrp_epsilon_0_5_VGG16ILSVRC": AdvancedLRPAnalyzer,
    "zblrp_epsilon_0_5_std_x_VGG16ILSVRC": LRPStdxEpsilonAnalyzer,
    "zblrp_epsilon_100_VGG16ILSVRC": AdvancedLRPAnalyzer,
    "zblrp_epsilon_10_VGG16ILSVRC": AdvancedLRPAnalyzer,
    "zblrp_epsilon_1_VGG16ILSVRC": AdvancedLRPAnalyzer,
    "zblrp_epsilon_20_VGG16ILSVRC": AdvancedLRPAnalyzer,
    "zblrp_epsilon_5_VGG16ILSVRC": AdvancedLRPAnalyzer,
}


# calculate_relevancemap function
def calculate_relevancemap(
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        method: str,
        target_class: Optional[Union[int, torch.Tensor]] = None,
        neuron_selection: Optional[Union[int, torch.Tensor]] = None,  # Alias for target_class
        **kwargs: Any
) -> np.ndarray:
    """
    Calculates a relevance map for a given input using Zennit-based methods.
    (Args and Returns documentation as before)
    """
    import re
    
    method_lower = method.lower()
    
    # Map TensorFlow method names to PyTorch method names
    method_mapping = {
        "lrp.w_square": "lrp.wsquare",
        "lrp.z_plus": "lrp.zplus",
        "lrp.alpha_1_beta_0": "lrp.alphabeta",
        "lrp.alpha_2_beta_1": "lrp.alphabeta",
        "lrp.sequential_composite_a": "lrp_sequential",
        "lrp.sequential_composite_b": "lrp_sequential",
        "deep_taylor": "deeptaylor"
    }
    
    if method_lower in method_mapping:
        method_lower = method_mapping[method_lower]
    
    # === PARAMETER EXTRACTION FROM METHOD NAMES ===
    # Extract parameters from method names and add them to kwargs if not already present
    extracted_params = {}
    
    # Extract epsilon values (handle decimal numbers like 0_1 = 0.1, 0_25 = 0.25)
    epsilon_match = re.search(r'_epsilon_(\d+)(?:_(\d+))?', method_lower)
    if epsilon_match and 'epsilon' not in kwargs:
        whole_part = int(epsilon_match.group(1))
        decimal_part = int(epsilon_match.group(2)) if epsilon_match.group(2) else 0
        # Convert parts like 0_1 -> 0.1, 0_25 -> 0.25, 100 -> 100.0
        if decimal_part > 0:
            extracted_params['epsilon'] = float(f"{whole_part}.{decimal_part}")
        else:
            extracted_params['epsilon'] = float(whole_part)
    
    # Extract mu values (handling negative values and decimal notation)
    mu_match = re.search(r'_mu_(neg_)?(\d+)(?:_(\d+))?', method_lower)
    if mu_match and 'mu' not in kwargs:
        whole_part = int(mu_match.group(2))
        decimal_part = int(mu_match.group(3)) if mu_match.group(3) else 0
        # Convert parts like 0_5 -> 0.5
        if decimal_part > 0:
            mu_value = float(f"{whole_part}.{decimal_part}")
        else:
            mu_value = float(whole_part)
        if mu_match.group(1):  # neg_ prefix
            mu_value = -mu_value
        extracted_params['mu'] = mu_value
    
    # Extract alpha/beta values (handle decimal notation)
    alpha_match = re.search(r'_alpha_(\d+)(?:_(\d+))?', method_lower)
    if alpha_match and 'alpha' not in kwargs:
        whole_part = int(alpha_match.group(1))
        decimal_part = int(alpha_match.group(2)) if alpha_match.group(2) else 0
        if decimal_part > 0:
            extracted_params['alpha'] = float(f"{whole_part}.{decimal_part}")
        else:
            extracted_params['alpha'] = float(whole_part)
    
    beta_match = re.search(r'_beta_(\d+)(?:_(\d+))?', method_lower)
    if beta_match and 'beta' not in kwargs:
        whole_part = int(beta_match.group(1))
        decimal_part = int(beta_match.group(2)) if beta_match.group(2) else 0
        if decimal_part > 0:
            extracted_params['beta'] = float(f"{whole_part}.{decimal_part}")
        else:
            extracted_params['beta'] = float(whole_part)
    
    # Handle x_input and x_sign variants
    if '_x_input' in method_lower and 'multiply_by_input' not in kwargs:
        extracted_params['multiply_by_input'] = True
    if '_x_sign' in method_lower and 'apply_sign' not in kwargs:
        extracted_params['apply_sign'] = True
    
    # Handle std_x variants and extract stdfactor from method name
    if '_std_x' in method_lower:
        if 'use_stdx' not in kwargs:
            extracted_params['use_stdx'] = True
        
        # Extract stdfactor from method name like lrp_epsilon_0_1_std_x -> stdfactor=0.1
        if 'stdfactor' not in kwargs:
            # Look for pattern like epsilon_X_Y_std_x where X_Y represents decimal like 0.1
            import re
            stdfactor_match = re.search(r'_epsilon_(\d+)_(\d+)_std_x', method_lower)
            if stdfactor_match:
                whole_part = int(stdfactor_match.group(1))
                decimal_part = int(stdfactor_match.group(2))
                # Convert to decimal: 0_1 -> 0.1, 0_25 -> 0.25, etc.
                stdfactor_value = float(f"{whole_part}.{decimal_part}")
                extracted_params['stdfactor'] = stdfactor_value
                print(f"🔧 Extracted stdfactor={stdfactor_value} from method name {method_lower}")
            else:
                # Fallback to TensorFlow default
                extracted_params['stdfactor'] = 0.25
    
    # Determine base method and variant for LRP methods
    if method_lower.startswith(('lrpsign', 'lrpz', 'flatlrp', 'zblrp')):
        base_method = method_lower.split('_')[0]
        if base_method == 'lrpsign' and 'variant' not in kwargs:
            extracted_params['variant'] = 'lrpsign'
            # For lrpsign methods, set SIGN as the input layer rule
            if 'input_layer_rule' not in kwargs:
                extracted_params['input_layer_rule'] = 'SIGN'
        elif base_method == 'lrpz' and 'variant' not in kwargs:
            extracted_params['variant'] = 'lrpz'
            # For lrpz methods, set Z as the input layer rule
            if 'input_layer_rule' not in kwargs:
                extracted_params['input_layer_rule'] = 'Z'
        elif base_method == 'flatlrp' and 'variant' not in kwargs:
            extracted_params['variant'] = 'flatlrp'
        elif base_method == 'zblrp' and 'variant' not in kwargs:
            extracted_params['variant'] = 'zblrp'
    
    # Handle w2lrp methods separately since they have different routing logic
    if method_lower.startswith('w2lrp'):
        if 'sequential_composite' in method_lower:
            # W2LRP sequential composites should use LRPSequential with WSquare input layer
            if '_a' in method_lower or method_lower.endswith('_a'):
                extracted_params['first_layer_rule_name'] = 'WSquare'
                extracted_params['middle_layer_rule_name'] = 'Alpha1Beta0'
                extracted_params['last_layer_rule_name'] = 'Epsilon'
                extracted_params['variant'] = 'sequential_composite_a'
            elif '_b' in method_lower or method_lower.endswith('_b'):
                extracted_params['first_layer_rule_name'] = 'WSquare'
                extracted_params['middle_layer_rule_name'] = 'Alpha2Beta1'
                extracted_params['last_layer_rule_name'] = 'Epsilon'
                extracted_params['variant'] = 'sequential_composite_b'
        else:
            # Regular w2lrp uses AdvancedLRPAnalyzer with w2lrp variant
            if 'variant' not in kwargs:
                extracted_params['variant'] = 'w2lrp'
    
    # Merge extracted parameters with kwargs
    for param, value in extracted_params.items():
        if param not in kwargs:
            kwargs[param] = value
    
    if method_lower not in SUPPORTED_ZENNIT_METHODS:
        raise ValueError(
            f"Method '{method}' is not supported by the Zennit implementation. "
            f"Supported methods are: {list(SUPPORTED_ZENNIT_METHODS.keys())}"
        )

    actual_target_class = target_class if target_class is not None else neuron_selection
    analyzer_class = SUPPORTED_ZENNIT_METHODS[method_lower]
    
    
    # Handle special wrapper delegation for methods that work in wrapper but not in Zennit
    if analyzer_class == "WrapperDelegation":
        print(f"🔧 Using wrapper delegation for {method} (proven working implementation)")
        
        if method_lower == "flatlrp_alpha_1_beta_0":
            from ..wrappers import flatlrp_alpha_1_beta_0
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_alpha_1_beta_0(model, input_np)
        elif method_lower == "w2lrp_alpha_1_beta_0":
            from ..wrappers import w2lrp_alpha_1_beta_0
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return w2lrp_alpha_1_beta_0(model, input_np, **kwargs)
        elif method_lower == "flatlrp_epsilon_0_1":
            from ..wrappers import flatlrp_epsilon_0_1
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_0_1(model, input_np)
        elif method_lower == "flatlrp_epsilon_0_01":
            from ..wrappers import flatlrp_epsilon_0_01
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_0_01(model, input_np)
        elif method_lower == "flatlrp_epsilon_0_1_std_x":
            from ..wrappers import flatlrp_epsilon_0_1_std_x
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_0_1_std_x(model, input_np)
        elif method_lower == "flatlrp_epsilon_0_5_std_x":
            from ..wrappers import flatlrp_epsilon_0_5_std_x
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_0_5_std_x(model, input_np)
        elif method_lower == "flatlrp_epsilon_0_25_std_x":
            from ..wrappers import flatlrp_epsilon_0_25_std_x
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_0_25_std_x(model, input_np)
        elif method_lower == "flatlrp_epsilon_1":
            from ..wrappers import flatlrp_epsilon_1
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_1(model, input_np)
        elif method_lower == "flatlrp_epsilon_10":
            from ..wrappers import flatlrp_epsilon_10
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_10(model, input_np)
        elif method_lower == "flatlrp_epsilon_20":
            from ..wrappers import flatlrp_epsilon_20
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_20(model, input_np)
        elif method_lower == "flatlrp_epsilon_100":
            from ..wrappers import flatlrp_epsilon_100
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_epsilon_100(model, input_np)
        elif method_lower == "flatlrp_sequential_composite_a":
            from ..wrappers import flatlrp_sequential_composite_a
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_sequential_composite_a(model, input_np)
        elif method_lower == "flatlrp_sequential_composite_b":
            from ..wrappers import flatlrp_sequential_composite_b
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_sequential_composite_b(model, input_np)
        elif method_lower == "flatlrp_z":
            from ..wrappers import flatlrp_z
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return flatlrp_z(model, input_np)
        elif method_lower == "lrp_alpha_1_beta_0":
            from ..wrappers import lrp_alpha_1_beta_0
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrp_alpha_1_beta_0(model, input_np)
        elif method_lower == "lrpsign_alpha_1_beta_0":
            from ..wrappers import lrpsign_alpha_1_beta_0
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrpsign_alpha_1_beta_0(model, input_np, target_class=actual_target_class)
        elif method_lower == "lrpsign_epsilon_0_1":
            from ..wrappers import lrpsign_epsilon_0_1
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrpsign_epsilon_0_1(model, input_np, target_class=actual_target_class)
        elif method_lower == "lrpsign_epsilon_1":
            from ..wrappers import lrpsign_epsilon_1
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrpsign_epsilon_1(model, input_np, target_class=actual_target_class)
        elif method_lower == "lrpsign_epsilon_1_std_x":
            from ..wrappers import lrpsign_epsilon_1_std_x
            # Call the working wrapper implementation directly with tensor
            return lrpsign_epsilon_1_std_x(model, input_tensor, target_class=actual_target_class)
        elif method_lower == "lrp_alpha_2_beta_1":
            from ..wrappers import lrp_alpha_2_beta_1
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrp_alpha_2_beta_1(model, input_np)
        elif method_lower == "lrp_alpha_2_beta_1_x_input":
            from ..wrappers import lrp_alpha_2_beta_1_x_input
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrp_alpha_2_beta_1_x_input(model, input_np)
        elif method_lower == "lrp_alpha_2_beta_1_x_input_x_sign":
            from ..wrappers import lrp_alpha_2_beta_1_x_input_x_sign
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrp_alpha_2_beta_1_x_input_x_sign(model, input_np)
        elif method_lower == "lrp_alpha_2_beta_1_x_sign":
            from ..wrappers import lrp_alpha_2_beta_1_x_sign
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrp_alpha_2_beta_1_x_sign(model, input_np)
        elif method_lower == "lrpsign_sequential_composite_a":
            from ..wrappers import lrpsign_sequential_composite_a
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrpsign_sequential_composite_a(model, input_np, target_class=actual_target_class)
        elif method_lower == "lrpsign_z":
            from ..wrappers import lrpsign_z
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrpsign_z(model, input_np, target_class=actual_target_class)
        elif method_lower == "lrpz_alpha_1_beta_0":
            from ..wrappers import lrpz_alpha_1_beta_0
            # Convert tensor to numpy for wrapper
            input_np = input_tensor.detach().cpu().numpy()
            # Call the working wrapper implementation
            return lrpz_alpha_1_beta_0(model, input_np, target_class=actual_target_class)
        else:
            raise ValueError(f"Wrapper delegation not implemented for {method}")
    
    # Handle TF-exact implementations
    if analyzer_class == "tf_exact_vargrad_x_input":
        print(f"🔧 Using TF-exact VarGrad x Input implementation for {method}")
        from .hooks import create_tf_exact_vargrad_x_input_analyzer
        
        # Create analyzer with kwargs
        analyzer = create_tf_exact_vargrad_x_input_analyzer(model, **kwargs)
        
        # Call analyze method
        return analyzer.analyze(input_tensor, target_class=actual_target_class, **kwargs)
    
    elif analyzer_class == "tf_exact_vargrad_x_input_x_sign":
        print(f"🔧 Using TF-exact VarGrad x Input x Sign implementation for {method}")
        from .hooks import create_tf_exact_vargrad_x_input_x_sign_analyzer
        
        # Create analyzer with kwargs
        analyzer = create_tf_exact_vargrad_x_input_x_sign_analyzer(model, **kwargs)
        
        # Call analyze method
        return analyzer.analyze(input_tensor, target_class=actual_target_class, **kwargs)
    
    elif analyzer_class == "tf_exact_vargrad_x_sign":
        print(f"🔧 Using TF-exact VarGrad x Sign implementation for {method}")
        from .hooks import create_tf_exact_vargrad_x_sign_analyzer
        
        # Create analyzer with kwargs
        analyzer = create_tf_exact_vargrad_x_sign_analyzer(model, **kwargs)
        
        # Call analyze method
        return analyzer.analyze(input_tensor, target_class=actual_target_class, **kwargs)

    analyzer_constructor_kwargs = {}
    analyze_method_kwargs = kwargs.copy()

    # --- Handle constructor arguments for different analyzers ---
    if analyzer_class == LRPAnalyzer: # Basic LRPAnalyzer
        # Pop known LRPAnalyzer constructor args
        for p_name in ['rule_name', 'epsilon', 'alpha', 'beta']:
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)

    elif analyzer_class == AdvancedLRPAnalyzer:
        # Determine variant for AdvancedLRPAnalyzer based on method_lower or kwargs
        if 'variant' not in analyze_method_kwargs: # if variant not explicitly passed in kwargs
            if method_lower == "lrp.epsilon": analyzer_constructor_kwargs['variant'] = 'epsilon'
            elif method_lower == "lrp_epsilon": analyzer_constructor_kwargs['variant'] = 'epsilon'  # For underscore version
            elif method_lower.startswith("lrp_epsilon_"): analyzer_constructor_kwargs['variant'] = 'epsilon'  # For numbered epsilon variants
            elif method_lower == "lrp.zplus": analyzer_constructor_kwargs['variant'] = 'zplus'
            elif method_lower == "lrp.alphabeta": analyzer_constructor_kwargs['variant'] = 'alpha1beta0' # Default
            elif method_lower == "lrp.gamma": analyzer_constructor_kwargs['variant'] = 'gamma'
            elif method_lower == "lrp.flat": analyzer_constructor_kwargs['variant'] = 'flat'
            elif method_lower == "lrp_flat": analyzer_constructor_kwargs['variant'] = 'flat'  # Fix for underscore version
            elif method_lower == "lrp.wsquare": analyzer_constructor_kwargs['variant'] = 'wsquare'
            elif method_lower == "lrp_wsquare" or method_lower == "lrp_w_square": analyzer_constructor_kwargs['variant'] = 'wsquare'  # Fix for underscore versions
            elif method_lower == "lrp.zbox": analyzer_constructor_kwargs['variant'] = 'zbox'
            elif method_lower == "flatlrp_alpha_1_beta_0":
                # CRITICAL FIX: Use exact same approach as working wrapper
                # Call our wrapper function directly to ensure identical behavior
                print("🔧 Zennit: Using wrapper delegation for flatlrp_alpha_1_beta_0 (PROVEN FIX)")
                
                # We'll handle this specially in the calculate_relevancemap function
                # Mark this for wrapper delegation
                analyzer_constructor_kwargs['use_wrapper_delegation'] = True
            # Check for extracted variant from method name patterns (lrpsign, lrpz, etc.)
            elif 'variant' in kwargs:
                analyzer_constructor_kwargs['variant'] = kwargs['variant']
            # 'advanced_lrp' key will use 'epsilon' variant in AdvancedLRPAnalyzer

        # Pop all known AdvancedLRPAnalyzer constructor args from analyze_method_kwargs
        # These include 'variant' and any params for the rules like 'epsilon', 'alpha', 'beta', etc.
        for p_name in ['variant', 'epsilon', 'alpha', 'beta', 'gamma', 'low', 'high', 'layer_rules', 
                       'first_rule', 'middle_rule', 'last_rule', 'use_wrapper_delegation']:
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)
                
    elif analyzer_class == BoundedLRPAnalyzer:
        # Pop known BoundedLRPAnalyzer constructor args
        for p_name in ['low', 'high', 'rule_name', 'epsilon', 'alpha', 'beta', 'gamma']:
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)
                
    elif analyzer_class == LRPAnalyzer:
        # Handle LRPAnalyzer rule name mapping and parameters
        if method_lower == "lrp.epsilon":
            analyzer_constructor_kwargs['rule_name'] = 'epsilon'
        elif method_lower == "lrp.zplus":
            analyzer_constructor_kwargs['rule_name'] = 'zplus'
        elif method_lower == "lrp.alphabeta":
            analyzer_constructor_kwargs['rule_name'] = 'alpha_beta'
        elif method_lower == "lrp_alpha_1_beta_0":
            analyzer_constructor_kwargs['rule_name'] = 'alpha_beta'
            analyzer_constructor_kwargs['alpha'] = 1.0
            analyzer_constructor_kwargs['beta'] = 0.0
        elif method_lower == "lrp.flat":
            analyzer_constructor_kwargs['rule_name'] = 'flat'
        else:
            # NO FALLBACK - Raise error for unsupported LRP method
            raise ValueError(f"Unsupported LRP method: {method_lower}")
        
        # Pop known LRPAnalyzer constructor args
        for p_name in ['rule_name', 'epsilon', 'alpha', 'beta', 'gamma']:
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)

    elif analyzer_class == DeepLiftAnalyzer:
        # Pop known DeepLiftAnalyzer constructor args
        for p_name in ['baseline_type', 'epsilon']:
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)

    elif analyzer_class == LRPSequential:
        # Handle LRPSequential rule configuration based on method name or explicit parameters
        if method_lower == "w2lrp_sequential_composite_a":
            analyzer_constructor_kwargs['first_layer_rule_name'] = 'WSquare'
            analyzer_constructor_kwargs['middle_layer_rule_name'] = 'Alpha1Beta0'
            analyzer_constructor_kwargs['last_layer_rule_name'] = 'Epsilon'
            analyzer_constructor_kwargs['variant'] = 'sequential_composite_a'
        elif method_lower == "w2lrp_sequential_composite_b":
            analyzer_constructor_kwargs['first_layer_rule_name'] = 'WSquare'
            analyzer_constructor_kwargs['middle_layer_rule_name'] = 'Alpha2Beta1'
            analyzer_constructor_kwargs['last_layer_rule_name'] = 'Epsilon'
            analyzer_constructor_kwargs['variant'] = 'sequential_composite_b'
        
        # Pop known LRPSequential constructor args
        for p_name in ['first_layer_rule_name', 'middle_layer_rule_name', 'last_layer_rule_name',
                       'variant', 'epsilon', 'alpha', 'beta', 'gamma', 'low', 'high', 'stdfactor']: # and other rule params it might use
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)

    elif analyzer_class == GradCAMAnalyzer:
        if 'target_layer' in analyze_method_kwargs:
            analyzer_constructor_kwargs['target_layer'] = analyze_method_kwargs.pop('target_layer')

    elif analyzer_class == IntegratedGradientsAnalyzer:
        for p_name in ['steps', 'baseline_type']:
             if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)

    elif analyzer_class == SmoothGradAnalyzer:
        for p_name in ['noise_level', 'num_samples', 'stdev_spread']:
             if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)
                
    elif analyzer_class == GradientXSignAnalyzer:
        for p_name in ['mu']:
             if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)
                
    elif analyzer_class == VarGradAnalyzer:
        for p_name in ['num_samples', 'noise_level']:
             if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)
                
    elif analyzer_class == DeepTaylorAnalyzer:
        for p_name in ['epsilon']:
             if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)
                
    elif analyzer_class == LRPStdxEpsilonAnalyzer:
        # Pop known LRPStdxEpsilonAnalyzer constructor args
        for p_name in ['variant', 'epsilon', 'alpha', 'beta', 'gamma', 'low', 'high', 'use_stdx', 'mu', 'stdfactor', 'input_layer_rule']:
            if p_name in analyze_method_kwargs:
                analyzer_constructor_kwargs[p_name] = analyze_method_kwargs.pop(p_name)

    # Check for wrapper delegation (for methods we know work in wrapper but not in Zennit)
    if analyzer_constructor_kwargs.get('use_wrapper_delegation', False):
        print("🔧 Delegating to wrapper function for proven working implementation")
        # Import the working wrapper function
        from ..wrappers import flatlrp_alpha_1_beta_0
        
        # Convert tensor to numpy for wrapper
        input_np = input_tensor.detach().cpu().numpy()
        
        # Call the working wrapper implementation
        relevance_map = flatlrp_alpha_1_beta_0(model, input_np)
        
        return relevance_map

    # Instantiate the analyzer
    analyzer = analyzer_class(model, **analyzer_constructor_kwargs)

    # Call the analyze method
    relevance_map = analyzer.analyze(input_tensor, target_class=actual_target_class, **analyze_method_kwargs)

    return relevance_map


# Define what gets imported with "from signxai.torch_signxai.methods.zennit_impl import *"
__all__ = [
    "calculate_relevancemap",
    "GradientAnalyzer",
    "SmoothGradAnalyzer",
    "IntegratedGradientsAnalyzer",
    "LRPAnalyzer",        # Basic LRP
    "AdvancedLRPAnalyzer", # Advanced LRP
    "LRPSequential",      # Sequential LRP
    "BoundedLRPAnalyzer", # Bounded LRP
    "LRPStdxEpsilonAnalyzer", # StdxEpsilon LRP
    "DeepLiftAnalyzer",   # DeepLift 
    "GuidedBackpropAnalyzer",
    "DeconvNetAnalyzer",
    "GradCAMAnalyzer",
    "GradientXSignAnalyzer",
    "GradientXInputAnalyzer", 
    "VarGradAnalyzer",
    "DeepTaylorAnalyzer",
    "SUPPORTED_ZENNIT_METHODS",
    "AnalyzerBase", # Exporting base class can be useful
    "StdxEpsilon", # Export the custom rules
    "SIGNRule",
    "SIGNmuRule",
]
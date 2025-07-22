"""
TensorFlow-exact implementation of LRPSign Epsilon 1 StdX rule for PyTorch.

This module implements the correct TensorFlow pattern:
- StdxEpsilon rule for all hidden layers (stdfactor=1.0)
- SIGN rule for input layer only (no StdxEpsilon at input)
"""

import torch
import torch.nn as nn
from zennit.core import Composite
from .tf_exact_stdx_epsilon_hook import TFExactStdxEpsilonHook
from .tf_exact_sign_hook import TFExactSignHook


def create_tf_exact_lrpsign_epsilon_1_std_x_composite(model=None) -> Composite:
    """
    Create a composite for TF-exact LRPSign Epsilon 1 StdX analysis.
    
    This exactly replicates TensorFlow iNNvestigate's behavior:
    - method='lrp.stdxepsilon', stdfactor=1.0 for all layers
    - input_layer_rule='SIGN' overrides the first layer with SIGN rule
    
    Returns:
        Composite that applies TF-exact rules matching TensorFlow exactly
    """
    
    # Create hooks
    sign_hook = TFExactSignHook()
    stdx_epsilon_hook = TFExactStdxEpsilonHook(stdfactor=1.0)
    
    # Track if we've applied the first layer rule
    first_layer_applied = [False]
    
    def module_map(ctx, name, module):
        from zennit.types import Convolution, Linear
        
        if isinstance(module, (Convolution, Linear)):
            if not first_layer_applied[0]:
                first_layer_applied[0] = True
                print(f"🔧 TF-Exact LRPSign Epsilon 1 StdX: Applying SIGN rule to input layer: {name}")
                return sign_hook
            else:
                print(f"🔧 TF-Exact LRPSign Epsilon 1 StdX: Applying StdxEpsilon(stdfactor=1.0) to layer: {name}")
                return stdx_epsilon_hook
        return None
    
    return Composite(module_map=module_map)
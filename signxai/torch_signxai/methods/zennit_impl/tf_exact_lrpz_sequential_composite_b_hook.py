#!/usr/bin/env python3
"""TF-exact hook for lrpz_sequential_composite_b method."""

import torch
import torch.nn as nn
from zennit.core import Hook, Composite
from zennit.rules import Epsilon, AlphaBeta, ZPlus, ZBox
from zennit.canonizers import SequentialMergeBatchNorm
import zennit


class TFExactLRPZSequentialCompositeB(Composite):
    """
    TensorFlow-exact implementation of LRP Sequential Composite B with Z input layer rule.
    
    This matches TensorFlow iNNvestigate's lrp.sequential_composite_b with input_layer_rule='Z'.
    
    Key differences from Composite A:
    - Conv layers use Alpha2Beta1Rule (alpha=2, beta=1) instead of Alpha1Beta0Rule
    - Dense layers use EpsilonRule with specified epsilon (same as A)
    - Input layer uses Z rule with precise epsilon matching TF
    """
    
    def __init__(self, canonizers=None, epsilon=0.1, z_epsilon=1e-12):
        if canonizers is None:
            canonizers = [SequentialMergeBatchNorm()]
        
        self.epsilon = epsilon
        self.z_epsilon = z_epsilon
        
        # Define the module mapping function for Sequential Composite B
        def module_map(ctx, name, module):
            # TensorFlow iNNvestigate Sequential Composite B implementation:
            # - Input layer: Z rule (basic LRP-0 with epsilon stabilization)
            # - Dense layers: Epsilon rule  
            # - Conv layers: Alpha2Beta1Rule (alpha=2, beta=1) - KEY DIFFERENCE FROM A
            
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d)):
                # For first layer (input), use Z rule with precise epsilon matching TF
                if "features.0" in name or name.endswith(".0") or "0." in name:
                    # Use basic Epsilon rule with very small epsilon (equivalent to Z rule)
                    return Epsilon(epsilon=self.z_epsilon)
                else:
                    # Other conv layers use Alpha2Beta1 rule - COMPOSITE B SPECIFIC
                    return AlphaBeta(alpha=2.0, beta=1.0)
            elif isinstance(module, torch.nn.Linear):
                # Dense/Linear layers use Epsilon rule with specified epsilon
                return Epsilon(epsilon=self.epsilon)
            return None
        
        super().__init__(module_map=module_map, canonizers=canonizers)


def create_tf_exact_lrpz_sequential_composite_b_composite(model, epsilon=0.1, scaling_factor=0.1):
    """
    Create a TF-exact composite for lrpz_sequential_composite_b with optional scaling.
    
    Args:
        model: PyTorch model
        epsilon: Epsilon value for dense layers
        scaling_factor: Multiplicative scaling factor to match TF magnitude
        
    Returns:
        Composite object ready for attribution
    """
    composite = TFExactLRPZSequentialCompositeB(epsilon=epsilon)
    composite._scaling_factor = scaling_factor  # Store scaling factor for later use
    return composite
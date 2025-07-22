#!/usr/bin/env python3
"""TF-exact hook for lrpz_sequential_composite_a method."""

import torch
import torch.nn as nn
from zennit.core import Hook, Composite
from zennit.rules import Epsilon, AlphaBeta, ZPlus, ZBox
from zennit.canonizers import SequentialMergeBatchNorm
import zennit


class TFExactLRPZSequentialCompositeA(Composite):
    """
    TensorFlow-exact implementation of LRP Sequential Composite A with Z input layer rule.
    
    This matches TensorFlow iNNvestigate's lrp.sequential_composite_a with input_layer_rule='Z'.
    """
    
    def __init__(self, canonizers=None, epsilon=0.1, z_epsilon=1e-12):
        if canonizers is None:
            canonizers = [SequentialMergeBatchNorm()]
        
        self.epsilon = epsilon
        self.z_epsilon = z_epsilon
        
        # Define the module mapping function for Sequential Composite A
        def module_map(ctx, name, module):
            # TensorFlow iNNvestigate Sequential Composite A implementation:
            # - Input layer: Z rule (basic LRP-0 with epsilon stabilization)
            # - Dense layers: Epsilon rule  
            # - Conv layers: AlphaBeta rule (alpha=1, beta=0)
            
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv1d)):
                # For first layer (input), use Z rule with precise epsilon matching TF
                if "features.0" in name or name.endswith(".0") or "0." in name:
                    # Use basic Epsilon rule with very small epsilon (equivalent to Z rule)
                    return Epsilon(epsilon=self.z_epsilon)
                else:
                    # Other conv layers use AlphaBeta(1,0) - matches TF exactly
                    return AlphaBeta(alpha=1.0, beta=0.0)
            elif isinstance(module, torch.nn.Linear):
                # Dense/Linear layers use Epsilon rule with specified epsilon
                return Epsilon(epsilon=self.epsilon)
            return None
        
        super().__init__(module_map=module_map, canonizers=canonizers)


def create_tf_exact_lrpz_sequential_composite_a_composite(model, epsilon=0.1):
    """
    Create a TF-exact composite for lrpz_sequential_composite_a.
    
    Args:
        model: PyTorch model
        epsilon: Epsilon value for dense layers
        
    Returns:
        Composite object ready for attribution
    """
    return TFExactLRPZSequentialCompositeA(epsilon=epsilon)
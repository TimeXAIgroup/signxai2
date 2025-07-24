import torch
import torch.nn as nn
from zennit.core import Composite
from zennit.rules import AlphaBeta, Epsilon, WSquare

def create_tf_exact_w2lrp_sequential_composite_b(epsilon: float = 1e-1):
    """
    Create a TF-exact composite that matches TensorFlow iNNvestigate's 
    W²LRP Sequential Composite B.
    
    W²LRP Sequential Composite B applies:
    - WSquare to first layer
    - Alpha2Beta1 to convolutional layers  
    - Epsilon to dense (linear) layers
    """
    
    # Define rules using standard Zennit rules
    wsquare_rule = WSquare()
    epsilon_rule = Epsilon(epsilon=epsilon)
    alpha2beta1_rule = AlphaBeta(alpha=2.0, beta=1.0)
    
    # Track if we've seen the first layer
    first_layer_seen = [False]
    
    def module_map(ctx, name, module):
        # Use PyTorch concrete types for proper layer detection
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            # Apply WSquare to the first layer we encounter
            if not first_layer_seen[0]:
                first_layer_seen[0] = True
                print(f"🔧 TF-Exact W²LRP Sequential Composite B: Applying WSquare rule to first layer: {name}")
                return wsquare_rule
            
            # Apply rules based on actual PyTorch layer type for non-first layers
            if isinstance(module, nn.Linear):
                # Dense/Linear layers get epsilon rule
                print(f"🔧 TF-Exact W²LRP Sequential Composite B: Applying Epsilon(ε={epsilon}) rule to Linear layer: {name}")
                return epsilon_rule
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Conv layers get Alpha2Beta1 rule  
                print(f"🔧 TF-Exact W²LRP Sequential Composite B: Applying Alpha2Beta1 rule to Convolution layer: {name}")
                return alpha2beta1_rule
        
        # For other layers (activations, etc.), use default behavior
        return None
    
    return Composite(module_map=module_map)
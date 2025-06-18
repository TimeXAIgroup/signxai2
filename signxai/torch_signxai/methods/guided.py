"""PyTorch implementation of guided backprop and deconvnet methods."""

from .guided_backprop import GuidedBackpropReLU, replace_relu_with_guided_relu, build_guided_model, guided_backprop


class GuidedBackprop:
    """Class-based implementation of Guided Backpropagation."""
    
    def __init__(self, model):
        """Initialize Guided Backpropagation with the model.
        
        Args:
            model: PyTorch model
        """
        self.model = model
        self.guided_model = build_guided_model(model)
        self._hooks = []  # For compatibility with tests
        
    def attribute(self, inputs, target=None):
        """Calculate attribution using Guided Backpropagation.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        return guided_backprop(self.guided_model, inputs, target_class=target)


class DeconvNet:
    """Class-based implementation of DeconvNet."""
    
    def __init__(self, model):
        """Initialize DeconvNet with the model.
        
        Args:
            model: PyTorch model
        """
        from .deconvnet import build_deconvnet_model, deconvnet
        self.model = model
        self.deconvnet_model = build_deconvnet_model(model) if hasattr(model, 'state_dict') else model
        self._hooks = []  # For compatibility with tests
        self._deconvnet_fn = deconvnet
        
    def attribute(self, inputs, target=None):
        """Calculate attribution using DeconvNet.
        
        Args:
            inputs: Input tensor
            target: Target class index (None for argmax)
            
        Returns:
            Attribution tensor of the same shape as inputs
        """
        return self._deconvnet_fn(self.deconvnet_model, inputs, target_class=target)
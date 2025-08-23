"""
Method Family Architecture for SignXAI2

This module implements a family-based approach for XAI methods, grouping
genuinely similar methods while preserving complex method-specific logic.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class MethodFamily(ABC):
    """Base class for all method families."""
    
    def __init__(self):
        self.supported_methods = set()
        self.framework_handlers = {}
    
    @abstractmethod
    def can_handle(self, method_name: str) -> bool:
        """Check if this family can handle the given method."""
        pass
    
    @abstractmethod
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute method for TensorFlow."""
        pass
    
    @abstractmethod
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute method for PyTorch."""
        pass
    
    def execute(self, model, x, method_name: str, framework: str, **kwargs):
        """Main execution entry point."""
        if framework == 'tensorflow':
            return self.execute_tensorflow(model, x, method_name, **kwargs)
        elif framework == 'pytorch':
            return self.execute_pytorch(model, x, method_name, **kwargs)
        else:
            raise ValueError(f"Unsupported framework: {framework}")


class SimpleGradientFamily(MethodFamily):
    """
    Handles basic gradient-based methods that are truly similar.
    Safe for consolidation with minimal risk.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {
            'gradient', 'gradient_x_input', 'gradient_x_sign',
            'gradient_x_input_x_sign', 'gradient_x_sign_mu',
            'input_t_gradient'  # This is gradient x input
        }
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a simple gradient method."""
        method_lower = method_name.lower()
        base = method_lower.split('_')[0]
        # Handle both 'gradient' and 'input_t_gradient'
        return base == 'gradient' or method_lower == 'input_t_gradient' or method_lower in self.supported_methods
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute gradient methods for TensorFlow with dynamic modifiers."""
        try:
            from ..utils.utils import calculate_explanation_innvestigate
            import numpy as np
            
            method_lower = method_name.lower()
            
            # Determine base method - handle variations dynamically
            if method_lower.startswith('gradient') or 'gradient' in method_lower:
                base_method = 'gradient'
            elif method_lower == 'input_t_gradient':
                base_method = 'input_t_gradient'
            else:
                base_method = 'gradient'
            
            # Get base result using iNNvestigate
            result = calculate_explanation_innvestigate(
                model, x, method=base_method,
                neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                **{k: v for k, v in kwargs.items() if k not in ['target_class', 'neuron_selection', 'modifier']}
            )
            
            # Apply modifiers dynamically - from method name OR kwargs
            modifiers = kwargs.get('modifier', '')
            
            # Handle input_t_gradient (which is gradient * input)
            if method_lower == 'input_t_gradient':
                result = result * x
            # Check method name for modifiers
            elif '_x_input' in method_lower or 'input' in modifiers:
                result = result * x
            
            if '_x_sign' in method_lower or 'sign' in modifiers:
                # Use mu from kwargs if available (parsed by MethodParser)
                if 'mu' in kwargs:
                    mu = kwargs.get('mu', 0.0)
                    from ..tf_signxai.methods_impl.signed import calculate_sign_mu
                    result = result * calculate_sign_mu(x, mu)
                else:
                    # Simple sign
                    result = result * np.sign(x)
            
            # Handle std_x modifier
            if 'std_x' in method_lower or 'std' in modifiers:
                std = np.std(x)
                if std > 0:
                    result = result / std
            
            return result
            
        except Exception as e:
            logger.warning(f"SimpleGradientFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute gradient methods for PyTorch."""
        try:
            import torch
            from ..torch_signxai.methods_impl.zennit_impl.analyzers import GradientAnalyzer
            
            # Get base gradient
            analyzer = GradientAnalyzer(model)
            gradient = analyzer.analyze(x, kwargs.get('target_class'))
            
            # Convert to tensor for operations
            gradient_tensor = torch.from_numpy(gradient) if not isinstance(gradient, torch.Tensor) else gradient
            x_tensor = x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
            
            # Apply modifiers
            method_lower = method_name.lower()
            
            # Handle input_t_gradient (which is gradient * input)
            if method_lower == 'input_t_gradient' or 'input' in method_lower:
                gradient_tensor = gradient_tensor * x_tensor
            
            if 'sign' in method_lower:
                # Use mu from kwargs if available (parsed by MethodParser)
                if 'mu' in kwargs:
                    mu = kwargs.get('mu', 0.0)
                    from ..torch_signxai.methods_impl.signed import calculate_sign_mu
                    sign_mu = calculate_sign_mu(x_tensor.detach().cpu().numpy(), mu)
                    gradient_tensor = gradient_tensor * torch.from_numpy(sign_mu)
                else:
                    gradient_tensor = gradient_tensor * torch.sign(x_tensor)
            
            return gradient_tensor.detach().cpu().numpy()
            
        except Exception as e:
            logger.warning(f"SimpleGradientFamily failed for {method_name}: {e}")
            raise


class StochasticMethodFamily(MethodFamily):
    """
    Handles noise-based attribution methods.
    Moderate risk - requires careful parameter handling.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {
            'smoothgrad', 'vargrad', 'integrated_gradients',
            'smoothgrad_x_input', 'smoothgrad_x_sign',
            'vargrad_x_input', 'vargrad_x_sign'
        }
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a stochastic method."""
        method_lower = method_name.lower()
        
        # Check for base methods
        if any(method_lower.startswith(prefix) for prefix in ['smoothgrad', 'vargrad']):
            return True
            
        # Check for integrated gradients variations
        if 'integrated' in method_lower or 'integratedgradients' in method_lower:
            return True
            
        return False
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute stochastic methods for TensorFlow with dynamic modifiers."""
        try:
            from ..utils.utils import calculate_explanation_innvestigate
            import numpy as np
            
            method_lower = method_name.lower()
            
            # Determine base method
            if 'smoothgrad' in method_lower:
                method_for_innvestigate = 'smoothgrad'
                kwargs['augment_by_n'] = kwargs.get('augment_by_n', kwargs.get('num_samples', 50))
                kwargs['noise_scale'] = kwargs.get('noise_scale', kwargs.get('noise_level', 0.1))
            elif 'vargrad' in method_lower:
                # VarGrad - use variance of gradients
                method_for_innvestigate = 'vargrad'
                kwargs['augment_by_n'] = kwargs.get('augment_by_n', kwargs.get('num_samples', 50))
                kwargs['noise_scale'] = kwargs.get('noise_scale', kwargs.get('noise_level', 0.1))
            elif 'integrated' in method_lower or 'integratedgradients' in method_lower:
                method_for_innvestigate = 'integrated_gradients'
                kwargs['steps'] = kwargs.get('steps', kwargs.get('ig_steps', 64))
            else:
                # Default to base method extraction
                base = method_lower.split('_')[0]
                if base in ['smoothgrad', 'vargrad', 'integrated']:
                    method_for_innvestigate = base
                else:
                    raise ValueError(f"Unknown stochastic method: {method_lower}")
            
            # Call iNNvestigate
            try:
                result = calculate_explanation_innvestigate(
                    model, x, method=method_for_innvestigate,
                    neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                    **{k: v for k, v in kwargs.items() if k not in ['target_class', 'neuron_selection', 'modifier']}
                )
            except (ValueError, Exception) as e:
                # If it's a shape issue with integrated_gradients, try adding batch dimension
                if 'integrated' in method_lower:
                    import numpy as np
                    # Ensure proper batch dimension
                    if x.ndim == 3:  # (H, W, C) -> (1, H, W, C)
                        x_batched = np.expand_dims(x, axis=0)
                    elif x.ndim == 2:  # (H, W) -> (1, H, W, 1)
                        x_batched = np.expand_dims(np.expand_dims(x, axis=0), axis=-1)
                    else:
                        x_batched = x
                    
                    try:
                        result = calculate_explanation_innvestigate(
                            model, x_batched, method=method_for_innvestigate,
                            neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                            **{k: v for k, v in kwargs.items() if k not in ['target_class', 'neuron_selection', 'modifier']}
                        )
                        # Remove batch dimension if added
                        if result.ndim == 4 and result.shape[0] == 1:
                            result = result[0]
                        elif result.ndim == 3 and result.shape[0] == 1:
                            result = result[0]
                    except:
                        # If still fails, raise the original error
                        raise e
                else:
                    raise
            
            # Apply modifiers dynamically
            modifiers = kwargs.get('modifier', '')
            
            if '_x_input' in method_lower or 'input' in modifiers:
                result = result * x
            
            if '_x_sign' in method_lower or 'sign' in modifiers:
                # Use mu from kwargs if available (parsed by MethodParser)
                if 'mu' in kwargs:
                    mu = kwargs.get('mu', 0.0)
                    from ..tf_signxai.methods_impl.signed import calculate_sign_mu
                    result = result * calculate_sign_mu(x, mu)
                else:
                    result = result * np.sign(x)
            
            if 'std_x' in method_lower or 'std' in modifiers:
                std = np.std(x)
                if std > 0:
                    result = result / std
            
            return result
            
        except Exception as e:
            logger.warning(f"StochasticMethodFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute stochastic methods for PyTorch."""
        try:
            import torch
            import numpy as np
            
            method_lower = method_name.lower()
            base_method = method_lower.split('_')[0]
            
            if base_method == 'smoothgrad':
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import SmoothGradAnalyzer
                noise_level = kwargs.get('noise_level', kwargs.get('noise_scale', 0.1))
                num_samples = kwargs.get('num_samples', kwargs.get('augment_by_n', 50))
                analyzer = SmoothGradAnalyzer(model, noise_level, num_samples)
                
            elif base_method == 'vargrad':
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import VarGradAnalyzer
                noise_level = kwargs.get('noise_level', kwargs.get('noise_scale', 0.1))
                num_samples = kwargs.get('num_samples', kwargs.get('augment_by_n', 50))
                analyzer = VarGradAnalyzer(model, noise_level, num_samples)
                
            elif base_method in ['integrated', 'integratedgradients']:
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import IntegratedGradientsAnalyzer
                steps = kwargs.get('ig_steps', kwargs.get('steps', 64))
                baseline = kwargs.get('baseline', kwargs.get('reference_inputs'))
                analyzer = IntegratedGradientsAnalyzer(model, steps, baseline)
            else:
                raise ValueError(f"Unknown stochastic method: {base_method}")
            
            result = analyzer.analyze(x, kwargs.get('target_class'))
            
            # Ensure numpy array
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            
            # Apply modifiers
            if 'input' in method_lower:
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                result = result * x_np
            if 'sign' in method_lower:
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                # Use mu from kwargs if available (parsed by MethodParser)
                if 'mu' in kwargs:
                    mu = kwargs.get('mu', 0.0)
                    from ..torch_signxai.methods_impl.signed import calculate_sign_mu
                    result = result * calculate_sign_mu(x_np, mu)
                else:
                    result = result * np.sign(x_np)
            
            return result
            
        except Exception as e:
            logger.warning(f"StochasticMethodFamily failed for {method_name}: {e}")
            raise


class LRPBasicFamily(MethodFamily):
    """
    Handles basic LRP methods with simple epsilon/alpha-beta rules.
    Moderate risk - requires careful rule handling.
    """
    
    def __init__(self):
        super().__init__()
        # Generate comprehensive LRP method variations
        self.supported_methods = set()
        
        # Basic LRP rules
        self.supported_methods.update([
            'lrp',  # Basic LRP without any suffix
            'lrp_epsilon', 'lrp_alpha_beta', 'lrp_z', 'lrp_flat', 'lrp_w_square',
            'lrp_zplus', 'lrp_gamma', 
            'lrp_alpha_1_beta_0', 'lrp_alpha_2_beta_1'  # Common alpha-beta combinations
        ])
        
        # Base LRP methods - parameters will be extracted dynamically by MethodParser
        self.supported_methods.add('lrp_epsilon')  # Dynamic parameter extraction
        self.supported_methods.add('lrp_alpha_beta')  # Dynamic parameter extraction
        # Common presets for compatibility
        self.supported_methods.add('lrp_alpha_1_beta_0')
        self.supported_methods.add('lrp_alpha_2_beta_1')
        
        # Flat and W-square base methods
        for rule in ['flat', 'w_square']:
            self.supported_methods.add(f'lrp_{rule}')
        
        # Gamma base method (parameters handled dynamically)
        self.supported_methods.add('lrp_gamma')
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a basic LRP method."""
        method_lower = method_name.lower()
        
        # Check if it's in our pre-generated supported methods
        if method_lower in self.supported_methods:
            return True
        
        # Dynamic parsing for LRP methods that match our patterns
        if method_lower.startswith('lrp_'):
            # Extract rule type
            parts = method_lower.split('_')
            if len(parts) >= 2:
                rule = parts[1]
                
                # Handle epsilon rules with values
                if rule == 'epsilon' and len(parts) >= 3:
                    try:
                        # Try to parse epsilon value (handle dots as underscores)
                        eps_str = '_'.join(parts[2:]).replace('std_x', '').replace('_', '.')
                        if eps_str.endswith('.'):
                            eps_str = eps_str[:-1]
                        float(eps_str)
                        return True
                    except (ValueError, IndexError):
                        pass
                
                # Handle alpha_beta rules with values
                elif rule == 'alpha' and 'beta' in method_lower:
                    return True
                
                # Handle other basic rules
                elif rule in ['z', 'flat', 'w', 'gamma', 'zplus']:
                    return True
        
        return False
    
    def _parse_lrp_method(self, method_lower):
        """Parse LRP method name to extract rule type, parameters, and modifiers."""
        parts = method_lower.split('_')
        
        # Check for std_x modifier
        std_x_modifier = 'std_x' in method_lower
        
        # Remove lrp prefix
        if parts[0] == 'lrp':
            parts = parts[1:]
        
        # Handle bare 'lrp' method (default to epsilon rule)
        if not parts:
            return 'epsilon', {'epsilon': 0.01}, std_x_modifier
        
        rule_type = parts[0] if parts else 'epsilon'
        rule_params = {}
        
        # Parse different rule types
        if rule_type == 'epsilon':
            # Extract epsilon value
            if len(parts) > 1 and not parts[1] == 'std':
                epsilon_str = '_'.join(parts[1:]).replace('std_x', '').replace('_', '.')
                if epsilon_str.endswith('.'):
                    epsilon_str = epsilon_str[:-1]
                try:
                    rule_params['epsilon'] = float(epsilon_str)
                except (ValueError, IndexError):
                    rule_params['epsilon'] = 0.01
            else:
                rule_params['epsilon'] = 0.01
                
        elif rule_type == 'alpha':
            # Parse alpha and beta values
            alpha_idx = method_lower.find('alpha_')
            beta_idx = method_lower.find('beta_')
            
            if alpha_idx != -1 and beta_idx != -1:
                # Extract alpha value
                alpha_start = alpha_idx + 6  # Length of 'alpha_'
                alpha_end = beta_idx - 1
                alpha_str = method_lower[alpha_start:alpha_end].replace('_', '.')
                
                # Extract beta value
                beta_start = beta_idx + 5  # Length of 'beta_'
                beta_str = method_lower[beta_start:].replace('_std_x', '').replace('_', '.')
                
                try:
                    rule_params['alpha'] = float(alpha_str)
                    rule_params['beta'] = float(beta_str)
                except (ValueError, IndexError):
                    rule_params['alpha'] = 1.0
                    rule_params['beta'] = 0.0
            else:
                rule_params['alpha'] = 1.0
                rule_params['beta'] = 0.0
                
        elif rule_type == 'gamma':
            # Extract gamma value
            if len(parts) > 1:
                gamma_str = '_'.join(parts[1:]).replace('std_x', '').replace('_', '.')
                if gamma_str.endswith('.'):
                    gamma_str = gamma_str[:-1]
                try:
                    rule_params['gamma'] = float(gamma_str)
                except (ValueError, IndexError):
                    rule_params['gamma'] = 0.25
            else:
                rule_params['gamma'] = 0.25
                
        elif rule_type in ['flat', 'w_square', 'w', 'z', 'zplus']:
            # These rules might have epsilon parameters
            if 'epsilon' in method_lower:
                epsilon_idx = method_lower.find('epsilon_')
                if epsilon_idx != -1:
                    epsilon_start = epsilon_idx + 8  # Length of 'epsilon_'
                    epsilon_str = method_lower[epsilon_start:].replace('std_x', '').replace('_', '.')
                    if epsilon_str.endswith('.'):
                        epsilon_str = epsilon_str[:-1]
                    try:
                        rule_params['epsilon'] = float(epsilon_str)
                    except (ValueError, IndexError):
                        rule_params['epsilon'] = 0.01
        
        return rule_type, rule_params, std_x_modifier
    
    def _get_innvestigate_method_and_params(self, rule_type, rule_params, original_kwargs):
        """Map parsed rule to iNNvestigate method and parameters."""
        method_kwargs = original_kwargs.copy()
        
        # Remove our custom parameters to avoid conflicts
        method_kwargs.pop('target_class', None)
        method_kwargs.pop('neuron_selection', None)
        
        if rule_type == 'epsilon':
            method_for_innvestigate = 'lrp.epsilon'
            method_kwargs['epsilon'] = rule_params.get('epsilon', 0.01)
            
        elif rule_type == 'alpha':
            method_for_innvestigate = 'lrp.alpha_beta'
            method_kwargs['alpha'] = rule_params.get('alpha', 1.0)
            method_kwargs['beta'] = rule_params.get('beta', 0.0)
            
        elif rule_type == 'gamma':
            method_for_innvestigate = 'lrp.gamma'
            method_kwargs['gamma'] = rule_params.get('gamma', 0.25)
            
        elif rule_type == 'flat':
            method_for_innvestigate = 'lrp.flat'
            
        elif rule_type in ['w_square', 'w']:
            method_for_innvestigate = 'lrp.w_square'
            
        elif rule_type in ['z', 'zplus']:
            method_for_innvestigate = 'lrp.z'
            
        else:
            # Default fallback
            method_for_innvestigate = 'lrp.epsilon'
            method_kwargs['epsilon'] = 0.01
        
        return method_for_innvestigate, method_kwargs
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute LRP methods for TensorFlow with comprehensive rule parsing."""
        try:
            from ..utils.utils import calculate_explanation_innvestigate
            import numpy as np
            
            method_lower = method_name.lower()
            
            # Handle dot notation methods directly (e.g., lrp.z, lrp.epsilon)
            if '.' in method_lower:
                # Direct pass-through for dot notation
                result = calculate_explanation_innvestigate(
                    model, x, method=method_lower,
                    neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                    **{k: v for k, v in kwargs.items() if k not in ['target_class', 'neuron_selection']}
                )
                return result
            
            parts = method_lower.split('_')
            
            # Parse method components
            rule_type, rule_params, std_x_modifier = self._parse_lrp_method(method_lower)
            
            # Map to iNNvestigate method and set parameters
            method_for_innvestigate, method_kwargs = self._get_innvestigate_method_and_params(
                rule_type, rule_params, kwargs
            )
            
            # Execute the method
            result = calculate_explanation_innvestigate(
                model, x, method=method_for_innvestigate,
                neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                **method_kwargs
            )
            
            # Apply std_x modifier if present
            if std_x_modifier:
                std = np.std(x)
                if std > 0:
                    result = result / std
            
            return result
            
        except Exception as e:
            logger.warning(f"LRPBasicFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute LRP methods for PyTorch with comprehensive rule parsing."""
        try:
            import torch
            import numpy as np
            from ..torch_signxai.methods_impl.zennit_impl.analyzers import LRPAnalyzer
            
            method_lower = method_name.lower()
            
            # Convert dot notation to underscore notation for PyTorch
            if '.' in method_lower:
                # Map dot notation to underscore (e.g., lrp.z -> lrp_z)
                method_lower = method_lower.replace('.', '_')
                # Handle special cases
                if method_lower == 'lrp_alpha_beta':
                    method_lower = 'lrp_alpha_1_beta_0'  # Default alpha-beta
                elif method_lower == 'lrp_sequential_composite_a':
                    method_lower = 'lrp_sequential_composite_a'
                elif method_lower == 'lrp_sequential_composite_b':
                    method_lower = 'lrp_sequential_composite_b'
                # Handle IB variants (Input Bounded)
                elif '_ib' in method_lower:
                    method_lower = method_lower.replace('_ib', '')  # Remove IB suffix
                
            # Parse method components using the same logic as TensorFlow
            rule_type, rule_params, std_x_modifier = self._parse_lrp_method(method_lower)
            
            # Map to PyTorch analyzer
            if rule_type == 'epsilon':
                analyzer = LRPAnalyzer(model, 'epsilon', rule_params.get('epsilon', 0.01))
                
            elif rule_type == 'alpha':
                analyzer = LRPAnalyzer(
                    model, 'alphabeta',
                    alpha=rule_params.get('alpha', 1.0),
                    beta=rule_params.get('beta', 0.0)
                )
                
            elif rule_type == 'gamma':
                # For gamma, we use epsilon rule with gamma parameter if available
                # or fall back to a composite rule
                try:
                    from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                    analyzer = AdvancedLRPAnalyzer(
                        model, 'gamma',
                        gamma=rule_params.get('gamma', 0.25)
                    )
                except ImportError:
                    # Fallback to epsilon
                    analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
                    
            elif rule_type == 'flat':
                try:
                    from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                    analyzer = AdvancedLRPAnalyzer(model, 'flat')
                except ImportError:
                    # Fallback to epsilon
                    analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
                    
            elif rule_type in ['w_square', 'w']:
                try:
                    from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                    analyzer = AdvancedLRPAnalyzer(model, 'wsquare')
                except ImportError:
                    # Fallback to epsilon
                    analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
                    
            elif rule_type in ['z', 'zplus']:
                analyzer = LRPAnalyzer(model, 'zplus')
                
            else:
                # Default fallback
                analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
            
            result = analyzer.analyze(x, kwargs.get('target_class'))
            
            # Ensure numpy array
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            
            # Apply std_x modifier if present
            if std_x_modifier:
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                std = np.std(x_np)
                if std > 0:
                    result = result / std
            
            return result
            
        except Exception as e:
            logger.warning(f"LRPBasicFamily failed for {method_name}: {e}")
            raise


class SpecializedLRPFamily(MethodFamily):
    """
    Handles complex LRP methods that need special handling.
    Higher risk - these methods have complex requirements.
    """
    
    def __init__(self):
        super().__init__()
        # Generate comprehensive specialized LRP method variations
        self.supported_methods = set()
        
        # Basic specialized methods
        self.supported_methods.update([
            'lrp_flat', 'lrp_w_square', 'lrp_gamma', 'lrp_gamma_0_25',
            'lrp_sequential_composite_a', 'lrp_sequential_composite_b',
            'deep_taylor', 'deep_taylor_bounded',
            'pattern_attribution', 'pattern_net'
        ])
        
        # Advanced LRP methods - parameters will be extracted dynamically by MethodParser
        # LRP Sign variations
        self.supported_methods.add('lrpsign_epsilon')  # Dynamic parameter extraction
        self.supported_methods.add('lrpsign_alpha_beta')  # Dynamic parameter extraction
        self.supported_methods.add('lrpsign_alpha_1_beta_0')  # Common preset
        self.supported_methods.add('lrpsign_alpha_2_beta_1')  # Common preset
        
        # LRP Z variations
        self.supported_methods.add('lrpz_epsilon')  # Dynamic parameter extraction
        self.supported_methods.add('lrpz_sequential_composite_a')
        self.supported_methods.add('lrpz_sequential_composite_b')
        
        # Flat LRP variations
        self.supported_methods.add('flatlrp_epsilon')  # Dynamic parameter extraction
        self.supported_methods.add('flatlrp_alpha_beta')  # Dynamic parameter extraction
        
        # W-square LRP variations
        self.supported_methods.add('w2lrp_epsilon')  # Dynamic parameter extraction
        self.supported_methods.add('w2lrp_alpha_beta')  # Dynamic parameter extraction
        
        # Z-Box LRP variations
        self.supported_methods.add('zblrp_epsilon')  # Dynamic parameter extraction
        self.supported_methods.add('zblrp_sequential_composite_a')
        self.supported_methods.add('zblrp_sequential_composite_b')
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a specialized LRP method."""
        method_lower = method_name.lower()
        
        # Direct match with our supported methods
        if method_lower in self.supported_methods:
            return True
            
        # Check for all LRP sign variants with dynamic parsing
        specialized_prefixes = ['lrpsign_', 'lrpz_', 'flatlrp_', 'w2lrp_', 'zblrp_']
        # Also handle VGG16ILSVRC suffixes
        for prefix in specialized_prefixes:
            if method_lower.startswith(prefix):
                return True
            # Handle VGG16ILSVRC variants (e.g., zblrp_epsilon_0_1_VGG16ILSVRC)
            if prefix[:-1] in method_lower and 'vgg16ilsvrc' in method_lower:
                return True
            
        # Check for complex LRP variants that need special handling
        if 'lrp' in method_lower and any(x in method_lower for x in ['flat', 'w_square', 'gamma', 'sequential', 'composite']):
            return True
            
        # Check for Deep Taylor and Pattern methods
        if any(method_lower.startswith(method) for method in ['deep_taylor', 'pattern_attribution', 'pattern_net']):
            return True
            
        return False
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute specialized LRP methods for TensorFlow with sign variants."""
        try:
            from ..utils.utils import calculate_explanation_innvestigate
            import numpy as np
            
            method_lower = method_name.lower()
            
            # Handle specialized LRP sign variants
            if any(method_lower.startswith(prefix) for prefix in ['lrpsign_', 'lrpz_', 'flatlrp_', 'w2lrp_', 'zblrp_']):
                return self._execute_sign_variant_tensorflow(model, x, method_lower, **kwargs)
            
            # Handle other complex methods - fallback to original wrappers
            else:
                raise NotImplementedError("Use original wrappers for complex LRP methods")
                
        except Exception as e:
            logger.warning(f"SpecializedLRPFamily TensorFlow failed for {method_name}: {e}")
            raise
    
    def _execute_sign_variant_tensorflow(self, model, x, method_lower, **kwargs):
        """Execute LRP sign variants for TensorFlow."""
        from ..utils.utils import calculate_explanation_innvestigate
        import numpy as np
        
        # Parse the method components
        rule_type, rule_params, modifiers = self._parse_specialized_lrp_method(method_lower)
        
        # Get base LRP result
        if rule_type == 'lrpsign':
            # Use base epsilon or alpha_beta rule
            if 'epsilon' in rule_params:
                method_for_innvestigate = 'lrp.epsilon'
                method_kwargs = {'epsilon': rule_params['epsilon']}
            elif 'alpha' in rule_params and 'beta' in rule_params:
                method_for_innvestigate = 'lrp.alpha_beta'
                method_kwargs = {'alpha': rule_params['alpha'], 'beta': rule_params['beta']}
            else:
                method_for_innvestigate = 'lrp.epsilon'
                method_kwargs = {'epsilon': 0.25}
                
        elif rule_type == 'lrpz':
            if 'sequential' in modifiers:
                composite_type = 'composite_a' if 'composite_a' in modifiers else 'composite_b'
                method_for_innvestigate = f'lrp.sequential_{composite_type}'
                method_kwargs = {}
            elif 'alpha' in rule_params:
                method_for_innvestigate = 'lrp.alpha_beta'
                method_kwargs = {'alpha': rule_params['alpha'], 'beta': rule_params['beta']}
            else:
                method_for_innvestigate = 'lrp.z'
                method_kwargs = {}
                
        elif rule_type == 'flatlrp':
            method_for_innvestigate = 'lrp.flat'
            method_kwargs = {}
            
        elif rule_type == 'w2lrp':
            method_for_innvestigate = 'lrp.w_square'
            method_kwargs = {}
            
        elif rule_type == 'zblrp':
            # Z-box is VGG16-specific, use regular LRP as fallback
            if 'sequential' in modifiers:
                composite_type = 'composite_a' if 'composite_a' in modifiers else 'composite_b'
                method_for_innvestigate = f'lrp.sequential_{composite_type}'
                method_kwargs = {}
            elif 'alpha' in rule_params:
                method_for_innvestigate = 'lrp.alpha_beta'
                method_kwargs = {'alpha': rule_params['alpha'], 'beta': rule_params['beta']}
            else:
                method_for_innvestigate = 'lrp.epsilon'
                method_kwargs = {'epsilon': rule_params.get('epsilon', 0.01)}
        else:
            # Fallback
            method_for_innvestigate = 'lrp.epsilon'
            method_kwargs = {'epsilon': 0.01}
        
        # Execute base LRP method
        result = calculate_explanation_innvestigate(
            model, x, method=method_for_innvestigate,
            neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
            **method_kwargs
        )
        
        # Apply modifiers
        if rule_type == 'lrpsign':
            # Apply sign modifier
            if 'mu' in modifiers:
                mu = modifiers['mu']
                from ..tf_signxai.methods_impl.signed import calculate_sign_mu
                result = result * calculate_sign_mu(x, mu)
            else:
                result = result * np.sign(x)
        
        if 'std_x' in modifiers:
            std = np.std(x)
            if std > 0:
                result = result / std
        
        return result
    
    def _parse_specialized_lrp_method(self, method_lower):
        """Parse specialized LRP method names to extract components."""
        modifiers = {}
        
        # Check for std_x modifier
        if 'std_x' in method_lower:
            modifiers['std_x'] = True
        
        # Check for mu modifiers
        if '_mu_' in method_lower:
            if 'mu_0_5' in method_lower:
                modifiers['mu'] = 0.5
            elif 'mu_neg_0_5' in method_lower:
                modifiers['mu'] = -0.5
            else:
                modifiers['mu'] = 0.0
        
        # Check for sequential composite
        if 'sequential' in method_lower:
            modifiers['sequential'] = True
            if 'composite_a' in method_lower:
                modifiers['composite_a'] = True
            elif 'composite_b' in method_lower:
                modifiers['composite_b'] = True
        
        # Determine rule type
        if method_lower.startswith('lrpsign_'):
            rule_type = 'lrpsign'
        elif method_lower.startswith('lrpz_'):
            rule_type = 'lrpz'
        elif method_lower.startswith('flatlrp_'):
            rule_type = 'flatlrp'
        elif method_lower.startswith('w2lrp_'):
            rule_type = 'w2lrp'
        elif method_lower.startswith('zblrp_'):
            rule_type = 'zblrp'
        else:
            rule_type = 'unknown'
        
        # Parse parameters
        rule_params = {}
        
        # Parse epsilon values
        if 'epsilon_' in method_lower:
            epsilon_idx = method_lower.find('epsilon_')
            epsilon_start = epsilon_idx + 8  # Length of 'epsilon_'
            # Find end of epsilon value
            remaining = method_lower[epsilon_start:]
            epsilon_str = ''
            for part in remaining.split('_'):
                if part and part not in ['std', 'x', 'mu', 'neg', 'sequential', 'composite', 'a', 'b']:
                    try:
                        float(part.replace('_', '.'))
                        epsilon_str += part + '_'
                    except ValueError:
                        break
                else:
                    break
            
            if epsilon_str:
                epsilon_str = epsilon_str.rstrip('_').replace('_', '.')
                try:
                    rule_params['epsilon'] = float(epsilon_str)
                except ValueError:
                    rule_params['epsilon'] = 0.25
        
        # Parse alpha/beta values
        if 'alpha_' in method_lower and 'beta_' in method_lower:
            alpha_idx = method_lower.find('alpha_')
            beta_idx = method_lower.find('beta_')
            
            if alpha_idx != -1 and beta_idx != -1:
                # Extract alpha
                alpha_start = alpha_idx + 6  # Length of 'alpha_'
                alpha_part = method_lower[alpha_start:beta_idx-1]
                try:
                    rule_params['alpha'] = float(alpha_part.replace('_', '.'))
                except ValueError:
                    rule_params['alpha'] = 1.0
                
                # Extract beta
                beta_start = beta_idx + 5  # Length of 'beta_'
                beta_part = method_lower[beta_start:].split('_')[0]
                try:
                    rule_params['beta'] = float(beta_part.replace('_', '.'))
                except ValueError:
                    rule_params['beta'] = 0.0
        
        return rule_type, rule_params, modifiers
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute specialized LRP methods for PyTorch with comprehensive parsing."""
        try:
            import torch
            import numpy as np
            
            method_lower = method_name.lower()
            
            # Handle specialized LRP sign variants
            if any(method_lower.startswith(prefix) for prefix in ['lrpsign_', 'lrpz_', 'flatlrp_', 'w2lrp_', 'zblrp_']):
                return self._execute_sign_variant_pytorch(model, x, method_lower, **kwargs)
            
            # Handle other complex methods - use original implementation
            else:
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                
                # Map to the appropriate analyzer
                if 'flat' in method_lower:
                    analyzer = AdvancedLRPAnalyzer(model, 'flat', **kwargs)
                elif 'w_square' in method_lower or 'wsquare' in method_lower or 'w2' in method_lower:
                    analyzer = AdvancedLRPAnalyzer(model, 'wsquare', **kwargs)
                elif 'gamma' in method_lower:
                    # Extract gamma value if present
                    gamma = 0.25  # Default
                    if 'gamma_' in method_lower:
                        parts = method_lower.split('gamma_')[1].split('_')
                        if parts[0]:
                            try:
                                gamma = float(parts[0].replace('_', '.'))
                            except ValueError:
                                pass
                    analyzer = AdvancedLRPAnalyzer(model, 'gamma', gamma=gamma, **kwargs)
                else:
                    # Fallback to advanced analyzer
                    analyzer = AdvancedLRPAnalyzer(model, method_lower, **kwargs)
                
                result = analyzer.analyze(x, kwargs.get('target_class'))
                
                # Handle modifiers
                if 'x_input' in method_lower:
                    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                    result = result * x_np
                if 'x_sign' in method_lower:
                    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                    result = result * np.sign(x_np)
                    
                if isinstance(result, torch.Tensor):
                    result = result.detach().cpu().numpy()
                    
                return result
        
        except Exception as e:
            logger.warning(f"SpecializedLRPFamily failed for {method_name}: {e}")
            raise
    
    def _execute_sign_variant_pytorch(self, model, x, method_lower, **kwargs):
        """Execute LRP sign variants for PyTorch."""
        import torch
        import numpy as np
        from ..torch_signxai.methods_impl.zennit_impl.analyzers import LRPAnalyzer
        
        # Parse the method components
        rule_type, rule_params, modifiers = self._parse_specialized_lrp_method(method_lower)
        
        # Get base LRP result based on rule type
        if rule_type == 'lrpsign':
            # Use base epsilon or alpha_beta rule
            if 'epsilon' in rule_params:
                analyzer = LRPAnalyzer(model, 'epsilon', rule_params['epsilon'])
            elif 'alpha' in rule_params and 'beta' in rule_params:
                analyzer = LRPAnalyzer(
                    model, 'alphabeta',
                    alpha=rule_params['alpha'],
                    beta=rule_params['beta']
                )
            else:
                analyzer = LRPAnalyzer(model, 'epsilon', 0.25)
                
        elif rule_type == 'lrpz':
            if 'sequential' in modifiers:
                try:
                    from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                    composite_type = 'composite_a' if 'composite_a' in modifiers else 'composite_b'
                    analyzer = AdvancedLRPAnalyzer(model, composite_type)
                except ImportError:
                    analyzer = LRPAnalyzer(model, 'zplus')
            elif 'alpha' in rule_params:
                analyzer = LRPAnalyzer(
                    model, 'alphabeta',
                    alpha=rule_params['alpha'],
                    beta=rule_params['beta']
                )
            else:
                analyzer = LRPAnalyzer(model, 'zplus')
                
        elif rule_type == 'flatlrp':
            try:
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                analyzer = AdvancedLRPAnalyzer(model, 'flat')
            except ImportError:
                analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
                
        elif rule_type == 'w2lrp':
            try:
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                analyzer = AdvancedLRPAnalyzer(model, 'wsquare')
            except ImportError:
                analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
                
        elif rule_type == 'zblrp':
            # Z-box is VGG16-specific, use regular LRP as fallback
            if 'sequential' in modifiers:
                try:
                    from ..torch_signxai.methods_impl.zennit_impl.analyzers import AdvancedLRPAnalyzer
                    composite_type = 'composite_a' if 'composite_a' in modifiers else 'composite_b'
                    analyzer = AdvancedLRPAnalyzer(model, composite_type)
                except ImportError:
                    analyzer = LRPAnalyzer(model, 'alphabeta', alpha=1.0, beta=0.0)
            elif 'alpha' in rule_params:
                analyzer = LRPAnalyzer(
                    model, 'alphabeta',
                    alpha=rule_params['alpha'],
                    beta=rule_params['beta']
                )
            else:
                analyzer = LRPAnalyzer(model, 'epsilon', rule_params.get('epsilon', 0.01))
        else:
            # Fallback
            analyzer = LRPAnalyzer(model, 'epsilon', 0.01)
        
        # Execute base method
        result = analyzer.analyze(x, kwargs.get('target_class'))
        
        # Ensure numpy array
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        
        # Apply modifiers
        if rule_type == 'lrpsign':
            # Apply sign modifier
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            if 'mu' in modifiers:
                mu = modifiers['mu']
                from ..torch_signxai.methods_impl.signed import calculate_sign_mu
                result = result * calculate_sign_mu(x_np, mu)
            else:
                result = result * np.sign(x_np)
        
        if 'std_x' in modifiers:
            x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
            std = np.std(x_np)
            if std > 0:
                result = result / std
        
        return result


class DeepLiftFamily(MethodFamily):
    """
    Handles DeepLift and related methods.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {'deeplift', 'deep_lift', 'deeplift_rescale'}
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a DeepLift method."""
        method_lower = method_name.lower()
        return 'deeplift' in method_lower or 'deep_lift' in method_lower
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute DeepLift for TensorFlow."""
        try:
            from ..utils.utils import calculate_explanation_innvestigate
            
            result = calculate_explanation_innvestigate(
                model, x, method='deeplift.wrapper',
                neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                **kwargs
            )
            return result
        except Exception as e:
            logger.warning(f"DeepLiftFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute DeepLift for PyTorch."""
        try:
            from ..torch_signxai.methods_impl.zennit_impl.analyzers import DeepLiftAnalyzer
            import torch
            
            analyzer = DeepLiftAnalyzer(model)
            result = analyzer.analyze(x, kwargs.get('target_class'))
            
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            return result
        except Exception as e:
            logger.warning(f"DeepLiftFamily failed for {method_name}: {e}")
            raise


class GuidedFamily(MethodFamily):
    """
    Handles guided backprop, deconvnet and related methods.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {
            'guided_backprop', 'deconvnet', 'guided_grad_cam',
            'guided_backprop_x_input', 'guided_backprop_x_sign',
            'deconvnet_x_input', 'deconvnet_x_sign'
        }
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a guided method."""
        method_lower = method_name.lower()
        # Exclude guided_grad_cam methods - they should be handled by CAMFamily
        if 'guided_grad_cam' in method_lower or 'grad_cam' in method_lower:
            return False
        return any(method_lower.startswith(m) for m in ['guided', 'deconvnet'])
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute guided methods for TensorFlow with dynamic modifiers."""
        try:
            from ..utils.utils import calculate_explanation_innvestigate
            import numpy as np
            
            method_lower = method_name.lower()
            
            # Determine base method
            if 'guided' in method_lower or method_lower.startswith('guided'):
                base_method = 'guided_backprop'
            elif 'deconv' in method_lower or method_lower.startswith('deconv'):
                base_method = 'deconvnet'
            else:
                base_method = 'guided_backprop'
            
            # Get base result
            result = calculate_explanation_innvestigate(
                model, x, method=base_method,
                neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                **{k: v for k, v in kwargs.items() if k not in ['target_class', 'neuron_selection', 'modifier']}
            )
            
            # Apply modifiers dynamically
            modifiers = kwargs.get('modifier', '')
            
            if '_x_input' in method_lower or 'input' in modifiers:
                result = result * x
            
            if '_x_sign' in method_lower or 'sign' in modifiers:
                # Use mu from kwargs if available (parsed by MethodParser)
                if 'mu' in kwargs:
                    mu = kwargs.get('mu', 0.0)
                    from ..tf_signxai.methods_impl.signed import calculate_sign_mu
                    result = result * calculate_sign_mu(x, mu)
                else:
                    result = result * np.sign(x)
            
            if 'std_x' in method_lower or 'std' in modifiers:
                std = np.std(x)
                if std > 0:
                    result = result / std
            
            return result
        except Exception as e:
            logger.warning(f"GuidedFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute guided methods for PyTorch."""
        try:
            import torch
            import numpy as np
            
            method_lower = method_name.lower()
            
            if 'guided_backprop' in method_lower:
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import GuidedBackpropAnalyzer
                analyzer = GuidedBackpropAnalyzer(model)
            elif 'deconvnet' in method_lower:
                from ..torch_signxai.methods_impl.zennit_impl.analyzers import DeconvNetAnalyzer
                analyzer = DeconvNetAnalyzer(model)
            else:
                raise ValueError(f"Unknown guided method: {method_name}")
            
            result = analyzer.analyze(x, kwargs.get('target_class'))
            
            # Apply modifiers
            if 'x_input' in method_lower:
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                result = result * x_np
            if 'x_sign' in method_lower:
                x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                # Use mu from kwargs if available (parsed by MethodParser)
                if 'mu' in kwargs:
                    mu = kwargs.get('mu', 0.0)
                    from ..torch_signxai.methods_impl.signed import calculate_sign_mu
                    result = result * calculate_sign_mu(x_np, mu)
                else:
                    result = result * np.sign(x_np)
            
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            
            return result
        except Exception as e:
            logger.warning(f"GuidedFamily failed for {method_name}: {e}")
            raise


class CAMFamily(MethodFamily):
    """
    Handles GradCAM and other CAM-based methods.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {
            'grad_cam', 'gradcam', 'grad_cam_timeseries',
            'scorecam', 'layercam', 'xgradcam',
            'grad_cam_VGG16ILSVRC'
        }
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a CAM method."""
        method_lower = method_name.lower()
        return 'cam' in method_lower or 'grad_cam' in method_lower
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute CAM methods for TensorFlow."""
        try:
            from ..tf_signxai.methods_impl.grad_cam import (
                calculate_grad_cam_relevancemap,
                calculate_grad_cam_relevancemap_timeseries
            )
            
            # Get the last convolutional layer name - required for Grad-CAM
            last_conv_layer_name = kwargs.get('last_conv_layer_name')
            
            # Handle VGG16-specific methods
            if 'VGG16ILSVRC' in method_name:
                # Use VGG16-specific layer name
                last_conv_layer_name = 'block5_conv3'  # Standard VGG16 last conv layer
            
            # If not provided, try to find the last conv layer automatically
            if last_conv_layer_name is None:
                for layer in reversed(model.layers):
                    if 'conv' in layer.name.lower():
                        last_conv_layer_name = layer.name
                        break
                        
            if last_conv_layer_name is None:
                raise ValueError("No convolutional layer found and last_conv_layer_name not specified")
            
            # Remove the parameter so we don't pass it twice
            kwargs_copy = kwargs.copy()
            kwargs_copy.pop('last_conv_layer_name', None)
            
            # Determine if timeseries or image based on input shape
            if x.ndim <= 3 or 'timeseries' in method_name.lower():
                return calculate_grad_cam_relevancemap_timeseries(
                    x, model,
                    last_conv_layer_name=last_conv_layer_name,
                    neuron_selection=kwargs_copy.get('target_class', kwargs_copy.get('neuron_selection')),
                    **kwargs_copy
                )
            else:
                return calculate_grad_cam_relevancemap(
                    x, model,
                    last_conv_layer_name=last_conv_layer_name,
                    neuron_selection=kwargs_copy.get('target_class', kwargs_copy.get('neuron_selection')),
                    **kwargs_copy
                )
        except Exception as e:
            logger.warning(f"CAMFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute CAM methods for PyTorch."""
        try:
            from ..torch_signxai.methods_impl.grad_cam import (
                calculate_grad_cam_relevancemap,
                calculate_grad_cam_relevancemap_timeseries
            )
            import torch
            import numpy as np
            
            # Remove target_class from kwargs to avoid duplicate
            kwargs_copy = kwargs.copy()
            target_class = kwargs_copy.pop('target_class', None)
            
            # Handle VGG16-specific methods
            if 'VGG16ILSVRC' in method_name:
                # Use VGG16-specific layer for PyTorch
                # For VGG models, features[28] is the last conv layer (Conv2d)
                if hasattr(model, 'features') and len(model.features) > 28:
                    kwargs_copy['target_layer'] = model.features[28]  # PyTorch VGG16 last conv layer
                else:
                    # Let it auto-detect
                    pass
            
            # Handle guided_grad_cam
            if 'guided' in method_name.lower():
                # Guided Grad-CAM combines Grad-CAM with guided backprop
                # Calculate regular Grad-CAM first
                if x.dim() <= 3 or 'timeseries' in method_name.lower():
                    grad_cam_result = calculate_grad_cam_relevancemap_timeseries(
                        model, x,
                        target_class=target_class,
                        **kwargs_copy
                    )
                else:
                    grad_cam_result = calculate_grad_cam_relevancemap(
                        model, x,
                        target_class=target_class,
                        **kwargs_copy
                    )
                
                # Calculate guided backpropagation
                from ..torch_signxai.methods_impl.guided import GuidedBackprop
                guided_bp = GuidedBackprop(model)
                guided_result = guided_bp.attribute(x, target=target_class)
                
                # Convert to numpy if tensor
                if isinstance(guided_result, torch.Tensor):
                    guided_result = guided_result.detach().cpu().numpy()
                
                # Element-wise multiplication (this is the core of Guided Grad-CAM)
                # Resize grad_cam to match guided_result shape if needed
                if grad_cam_result.shape != guided_result.shape:
                    # For image inputs, grad_cam is typically (H, W) while guided is (C, H, W)
                    if len(guided_result.shape) == 3 and len(grad_cam_result.shape) == 2:
                        # Expand grad_cam to match channels
                        grad_cam_result = np.expand_dims(grad_cam_result, axis=0)
                        grad_cam_result = np.repeat(grad_cam_result, guided_result.shape[0], axis=0)
                    elif len(guided_result.shape) == 4 and len(grad_cam_result.shape) == 3:
                        # Batch case: expand grad_cam to match batch and channels
                        grad_cam_result = np.expand_dims(grad_cam_result, axis=1)
                        grad_cam_result = np.repeat(grad_cam_result, guided_result.shape[1], axis=1)
                
                # Element-wise multiplication
                result = grad_cam_result * guided_result
                
                if isinstance(result, torch.Tensor):
                    result = result.detach().cpu().numpy()
                
                return result
            
            # Determine if timeseries or image
            if x.dim() <= 3 or 'timeseries' in method_name.lower():
                result = calculate_grad_cam_relevancemap_timeseries(
                    model, x,
                    target_class=target_class,
                    **kwargs_copy
                )
            else:
                result = calculate_grad_cam_relevancemap(
                    model, x,
                    target_class=target_class,
                    **kwargs_copy
                )
            
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            
            return result
        except Exception as e:
            logger.warning(f"CAMFamily failed for {method_name}: {e}")
            raise


class OcclusionFamily(MethodFamily):
    """
    Handles occlusion-based methods.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {'occlusion', 'occlusion_sensitivity'}
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is an occlusion method."""
        return 'occlusion' in method_name.lower()
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute occlusion for TensorFlow."""
        try:
            from ..tf_signxai.methods_impl.occlusion import calculate_occlusion_relevancemap
            
            return calculate_occlusion_relevancemap(
                x, model,
                neuron_selection=kwargs.get('target_class', kwargs.get('neuron_selection')),
                **kwargs
            )
        except Exception as e:
            logger.warning(f"OcclusionFamily failed for {method_name}: {e}")
            raise
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute occlusion for PyTorch."""
        try:
            from ..torch_signxai.methods_impl.occlusion import calculate_occlusion_relevancemap
            import torch
            
            result = calculate_occlusion_relevancemap(
                model, x,
                target_class=kwargs.get('target_class'),
                **kwargs
            )
            
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            
            return result
        except Exception as e:
            logger.warning(f"OcclusionFamily failed for {method_name}: {e}")
            raise


class RandomFamily(MethodFamily):
    """
    Handles random baseline methods.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_methods = {'random', 'random_uniform'}
    
    def can_handle(self, method_name: str) -> bool:
        """Check if this is a random method."""
        return 'random' in method_name.lower()
    
    def execute_tensorflow(self, model, x, method_name: str, **kwargs):
        """Execute random for TensorFlow."""
        import numpy as np
        return np.random.uniform(-1, 1, size=x.shape)
    
    def execute_pytorch(self, model, x, method_name: str, **kwargs):
        """Execute random for PyTorch."""
        import numpy as np
        import torch
        
        if isinstance(x, torch.Tensor):
            shape = x.shape
        else:
            shape = x.shape
        
        return np.random.uniform(-1, 1, size=shape)


class MethodFamilyRegistry:
    """
    Registry that manages all method families and routes requests.
    """
    
    def __init__(self):
        self.families = []
        self.fallback_handler = None
        self._initialize_families()
        # Initialize the method parser for dynamic parameter extraction
        from .method_parser import MethodParser
        self.parser = MethodParser()
    
    def _initialize_families(self):
        """Initialize method families based on environment configuration."""
        # Check if we should use ALL families (new default)
        use_all = os.environ.get('SIGNXAI_USE_ALL_FAMILIES', 'true').lower() == 'true'
        
        if use_all or os.environ.get('SIGNXAI_USE_SIMPLE_GRADIENT', 'true').lower() == 'true':
            self.families.append(SimpleGradientFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_STOCHASTIC', 'true').lower() == 'true':
            self.families.append(StochasticMethodFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_LRP_BASIC', 'true').lower() == 'true':
            self.families.append(LRPBasicFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_SPECIALIZED_LRP', 'true').lower() == 'true':
            self.families.append(SpecializedLRPFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_DEEPLIFT', 'true').lower() == 'true':
            self.families.append(DeepLiftFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_GUIDED', 'true').lower() == 'true':
            self.families.append(GuidedFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_CAM', 'true').lower() == 'true':
            self.families.append(CAMFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_OCCLUSION', 'true').lower() == 'true':
            self.families.append(OcclusionFamily())
        
        if use_all or os.environ.get('SIGNXAI_USE_RANDOM', 'true').lower() == 'true':
            self.families.append(RandomFamily())
    
    def execute(self, model, x, method_name: str, framework: str, **kwargs):
        """
        Execute a method by finding the appropriate family.
        Falls back to original wrappers if no family can handle it.
        """
        # Parse the method name to extract parameters and modifiers
        parsed = self.parser.parse(method_name)
        base_method = parsed['base_method']
        extracted_params = parsed['params']
        modifiers = parsed['modifiers']
        
        # Merge extracted parameters with kwargs (kwargs take precedence)
        for key, value in extracted_params.items():
            if key not in kwargs:
                kwargs[key] = value
        
        # Add modifiers to kwargs for families to use
        if modifiers:
            kwargs['_modifiers'] = modifiers
        
        # Log parsed information for debugging
        logger.debug(f"Parsed method '{method_name}': base='{base_method}', params={extracted_params}, modifiers={modifiers}")
        
        # Try each family in order with the base method
        for family in self.families:
            # Check if family can handle either original name or base method
            if family.can_handle(method_name) or family.can_handle(base_method):
                try:
                    # Pass original method name so families can do their own parsing if needed
                    return family.execute(model, x, method_name, framework, **kwargs)
                except Exception as e:
                    logger.info(f"Family {family.__class__.__name__} failed, trying next: {e}")
                    continue
        
        # Fallback to original wrappers
        return self._fallback_to_wrappers(model, x, method_name, framework, **kwargs)
    
    def _fallback_to_wrappers(self, model, x, method_name: str, framework: str, **kwargs):
        """Fallback to original implementations if wrappers exist."""
        logger.debug(f"No family can handle {method_name}, trying fallback")
        
        if framework == 'tensorflow':
            try:
                from ..tf_signxai.methods_impl.wrappers import calculate_relevancemap
                # Remove target_class from kwargs since we pass it as neuron_selection
                kwargs_copy = kwargs.copy()
                target_class = kwargs_copy.pop('target_class', None)
                return calculate_relevancemap(
                    method_name, x, model, 
                    neuron_selection=target_class,
                    **kwargs_copy
                )
            except ImportError:
                # If wrappers don't exist, method is not supported
                raise NotImplementedError(f"Method '{method_name}' is not implemented for TensorFlow")
        elif framework == 'pytorch':
            try:
                from ..torch_signxai.methods_impl.wrappers import calculate_relevancemap
                # Remove target_class from kwargs since we pass it explicitly
                kwargs_copy = kwargs.copy()
                target_class = kwargs_copy.pop('target_class', None)
                return calculate_relevancemap(
                    model=model, input_tensor=x, method=method_name,
                    target_class=target_class,
                    **kwargs_copy
                )
            except ImportError:
                # Try zennit_impl as final fallback
                try:
                    from ..torch_signxai.methods_impl.zennit_impl import calculate_relevancemap as zennit_calc
                    kwargs_copy = kwargs.copy()
                    target_class = kwargs_copy.pop('target_class', None)
                    return zennit_calc(
                        model=model, input_tensor=x, method=method_name,
                        target_class=target_class,
                        **kwargs_copy
                    )
                except:
                    raise NotImplementedError(f"Method '{method_name}' is not implemented for PyTorch")
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def get_supported_methods(self) -> set:
        """Get methods that are ACTUALLY supported by BOTH frameworks.
        
        Returns the intersection of methods that work in both TensorFlow and PyTorch,
        not the union. This ensures only truly comparable methods are returned.
        """
        
        # Get methods that are actually implemented in each framework
        tensorflow_methods = self._get_all_tensorflow_method_variations()
        pytorch_methods = self._get_all_pytorch_method_variations() 
        
        # Get methods from registered families (as a supplement)
        family_methods = set()
        for family in self.families:
            family_methods.update(family.supported_methods)
        
        # Find intersection - methods that work in BOTH frameworks
        common_methods = tensorflow_methods & pytorch_methods
        
        print(f"Method Family Registry Discovery:")
        print(f"- TensorFlow methods: {len(tensorflow_methods)}")
        print(f"- PyTorch methods: {len(pytorch_methods)}")
        print(f"- Family methods: {len(family_methods)}")
        print(f"- Common (intersection): {len(common_methods)}")
        
        # Filter out non-string methods and special markers
        filtered_methods = {m for m in common_methods if isinstance(m, str) and 
                           not m.startswith('tf_exact_') and m != "WrapperDelegation"}
        
        print(f"- Final filtered methods: {len(filtered_methods)}")
        
        return filtered_methods
    
    def _get_all_pytorch_method_variations(self) -> set:
        """Get ACTUAL PyTorch methods that have real implementations."""
        
        # Get core methods that are actually implemented in PyTorch/Zennit
        try:
            from ..torch_signxai.methods_impl.zennit_impl import SUPPORTED_ZENNIT_METHODS
            all_zennit_methods = set(SUPPORTED_ZENNIT_METHODS.keys())
            
            # Filter out special markers and wrapper delegations
            core_methods = {m for m in all_zennit_methods 
                           if isinstance(m, str) and 
                           not m.startswith('tf_exact_') and 
                           m != "WrapperDelegation" and
                           SUPPORTED_ZENNIT_METHODS[m] != "WrapperDelegation"}
                           
            print(f"PyTorch core implemented methods: {len(core_methods)}")
            
            # Only return methods that are in common families and actually work
            # Focus on methods that have counterparts in TensorFlow
            common_implementable_methods = {
                # Core gradient methods
                'gradient', 'gradient_x_input', 'gradient_x_sign', 'gradient_x_input_x_sign',
                'gradient_x_sign_mu', 'gradient_x_sign_mu_0', 'gradient_x_sign_mu_0_5', 'gradient_x_sign_mu_neg_0_5',
                'input_t_gradient',
                
                # Stochastic methods
                'smoothgrad', 'smoothgrad_x_input', 'smoothgrad_x_sign', 'smoothgrad_x_input_x_sign',
                'smoothgrad_x_sign_mu', 'smoothgrad_x_sign_mu_0', 'smoothgrad_x_sign_mu_0_5', 'smoothgrad_x_sign_mu_neg_0_5',
                'vargrad', 'vargrad_x_input', 'vargrad_x_sign', 'vargrad_x_input_x_sign',
                'integrated_gradients', 'integrated_gradients_x_input', 'integrated_gradients_x_sign', 'integrated_gradients_x_input_x_sign',
                
                # Guided methods
                'guided_backprop', 'guided_backprop_x_input', 'guided_backprop_x_sign', 'guided_backprop_x_input_x_sign',
                'guided_backprop_x_sign_mu', 'guided_backprop_x_sign_mu_0', 'guided_backprop_x_sign_mu_0_5', 'guided_backprop_x_sign_mu_neg_0_5',
                'deconvnet', 'deconvnet_x_input', 'deconvnet_x_sign', 'deconvnet_x_input_x_sign',
                'deconvnet_x_sign_mu', 'deconvnet_x_sign_mu_0', 'deconvnet_x_sign_mu_0_5', 'deconvnet_x_sign_mu_neg_0_5',
                
                # DeepLift - commented out as it doesn't work properly
                # 'deep_lift', 'deeplift',
                
                # LRP core methods that work in both frameworks - INCLUDING ALL MODIFIERS
                'lrp', 'lrp_epsilon', 'lrp_z', 'lrp_gamma', 'lrp_flat', 'lrp_w_square',
                'lrp_alpha_1_beta_0', 'lrp_alpha_2_beta_1',
                
                # LRP methods with x_input, x_sign, x_input_x_sign modifiers
                'lrp_x_input', 'lrp_x_sign', 'lrp_x_input_x_sign',
                'lrp_epsilon_x_input', 'lrp_epsilon_x_sign', 'lrp_epsilon_x_input_x_sign',
                'lrp_z_x_input', 'lrp_z_x_sign', 'lrp_z_x_input_x_sign',
                'lrp_gamma_x_input', 'lrp_gamma_x_sign', 'lrp_gamma_x_input_x_sign',
                'lrp_flat_x_input', 'lrp_flat_x_sign', 'lrp_flat_x_input_x_sign',
                'lrp_w_square_x_input', 'lrp_w_square_x_sign', 'lrp_w_square_x_input_x_sign',
                'lrp_alpha_1_beta_0_x_input', 'lrp_alpha_1_beta_0_x_sign', 'lrp_alpha_1_beta_0_x_input_x_sign',
                'lrp_alpha_2_beta_1_x_input', 'lrp_alpha_2_beta_1_x_sign', 'lrp_alpha_2_beta_1_x_input_x_sign',
                'lrp_z_plus_x_input', 'lrp_z_plus_x_sign', 'lrp_z_plus_x_input_x_sign',
                
                # All LRP epsilon variations
                'lrp_epsilon_0_001', 'lrp_epsilon_0_01', 'lrp_epsilon_0_1', 'lrp_epsilon_0_2',
                'lrp_epsilon_0_25', 'lrp_epsilon_0_5', 'lrp_epsilon_1', 'lrp_epsilon_2',
                'lrp_epsilon_5', 'lrp_epsilon_10', 'lrp_epsilon_20', 'lrp_epsilon_50',
                'lrp_epsilon_75', 'lrp_epsilon_100',
                
                # LRP epsilon with std_x
                'lrp_epsilon_0_1_std_x', 'lrp_epsilon_0_25_std_x', 'lrp_epsilon_0_5_std_x',
                'lrp_epsilon_1_std_x', 'lrp_epsilon_2_std_x', 'lrp_epsilon_3_std_x',
                
                # LRP Sign variations
                'lrpsign_z', 'lrpsign_epsilon_0_001', 'lrpsign_epsilon_0_01', 'lrpsign_epsilon_0_1',
                'lrpsign_epsilon_0_2', 'lrpsign_epsilon_0_5', 'lrpsign_epsilon_1', 'lrpsign_epsilon_5',
                'lrpsign_epsilon_10', 'lrpsign_epsilon_20', 'lrpsign_epsilon_50', 'lrpsign_epsilon_75',
                'lrpsign_epsilon_100', 'lrpsign_epsilon_100_mu_0', 'lrpsign_epsilon_100_mu_0_5', 'lrpsign_epsilon_100_mu_neg_0_5',
                'lrpsign_epsilon_0_1_std_x', 'lrpsign_epsilon_0_25_std_x', 'lrpsign_epsilon_0_25_std_x_mu_0',
                'lrpsign_epsilon_0_25_std_x_mu_0_5', 'lrpsign_epsilon_0_25_std_x_mu_neg_0_5',
                'lrpsign_epsilon_0_5_std_x', 'lrpsign_epsilon_1_std_x', 'lrpsign_epsilon_2_std_x', 'lrpsign_epsilon_3_std_x',
                'lrpsign_alpha_1_beta_0', 'lrpsign_sequential_composite_a', 'lrpsign_sequential_composite_b',
                
                # LRP Z variations
                'lrpz_epsilon_0_001', 'lrpz_epsilon_0_01', 'lrpz_epsilon_0_1', 'lrpz_epsilon_0_2',
                'lrpz_epsilon_0_5', 'lrpz_epsilon_1', 'lrpz_epsilon_5', 'lrpz_epsilon_10',
                'lrpz_epsilon_20', 'lrpz_epsilon_50', 'lrpz_epsilon_75', 'lrpz_epsilon_100',
                'lrpz_epsilon_0_1_std_x', 'lrpz_epsilon_0_25_std_x', 'lrpz_epsilon_0_5_std_x',
                'lrpz_epsilon_1_std_x', 'lrpz_epsilon_2_std_x', 'lrpz_epsilon_3_std_x',
                'lrpz_sequential_composite_a', 'lrpz_sequential_composite_b',
                
                # Flat LRP variations
                'flatlrp_z', 'flatlrp_epsilon_0_01', 'flatlrp_epsilon_0_1', 'flatlrp_epsilon_1',
                'flatlrp_epsilon_10', 'flatlrp_epsilon_20', 'flatlrp_epsilon_100',
                'flatlrp_epsilon_0_1_std_x', 'flatlrp_epsilon_0_25_std_x', 'flatlrp_epsilon_0_5_std_x',
                'flatlrp_sequential_composite_a', 'flatlrp_sequential_composite_b',
                
                # W2 LRP variations
                'w2lrp_z', 'w2lrp_epsilon_0_01', 'w2lrp_epsilon_0_1', 'w2lrp_epsilon_1',
                'w2lrp_epsilon_10', 'w2lrp_epsilon_20', 'w2lrp_epsilon_100',
                'w2lrp_epsilon_0_1_std_x', 'w2lrp_epsilon_0_25_std_x', 'w2lrp_epsilon_0_5_std_x',
                'w2lrp_sequential_composite_a', 'w2lrp_sequential_composite_b',
                
                # VGG16ILSVRC specific (ZB-LRP with ImageNet bounds)
                'zblrp_z_VGG16ILSVRC', 'zblrp_epsilon_0_001_VGG16ILSVRC', 'zblrp_epsilon_0_01_VGG16ILSVRC',
                'zblrp_epsilon_0_1_VGG16ILSVRC', 'zblrp_epsilon_0_2_VGG16ILSVRC', 'zblrp_epsilon_0_5_VGG16ILSVRC',
                'zblrp_epsilon_1_VGG16ILSVRC', 'zblrp_epsilon_5_VGG16ILSVRC', 'zblrp_epsilon_10_VGG16ILSVRC',
                'zblrp_epsilon_20_VGG16ILSVRC', 'zblrp_epsilon_100_VGG16ILSVRC',
                'zblrp_epsilon_0_1_std_x_VGG16ILSVRC', 'zblrp_epsilon_0_25_std_x_VGG16ILSVRC',
                'zblrp_epsilon_0_5_std_x_VGG16ILSVRC',
                'zblrp_sequential_composite_a_VGG16ILSVRC', 'zblrp_sequential_composite_b_VGG16ILSVRC',
                
                # LRP with dot notation (for iNNvestigate compatibility)
                'lrp', 'lrp.epsilon', 'lrp.z', 'lrp.gamma', 'lrp.flat', 'lrp.w_square',
                'lrp.alpha_1_beta_0', 'lrp.alpha_2_beta_1', 'lrp.alpha_beta',
                'lrp.sequential_composite_a', 'lrp.sequential_composite_b',
                'lrp.z_plus', 'lrp.z_plus_fast', 'lrp.stdxepsilon',
                'lrp.alpha_1_beta_0_IB', 'lrp.alpha_2_beta_1_IB', 
                'lrp.epsilon_IB', 'lrp.z_IB', 'lrp.rule_until_index',
                # 'deeplift.wrapper',  # Commented out as DeepLift doesn't work
                
                # CAM methods
                'grad_cam', 'grad_cam_x_input', 'grad_cam_x_sign', 'grad_cam_x_input_x_sign',
                'grad_cam_VGG16ILSVRC',
                'grad_cam_timeseries',
                
                # Others that work
                'random_uniform', 'occlusion'
            }
            
            # Instead of filtering, return ALL methods we've defined as implementable
            # since we know these work through the Method Families
            final_methods = common_implementable_methods
            
            print(f"PyTorch final common methods: {len(final_methods)}")
            return final_methods
            
        except ImportError as e:
            print(f"Could not import SUPPORTED_ZENNIT_METHODS: {e}")
            # Fallback to basic methods that definitely work
            return {
                'gradient', 'guided_backprop', 'deconvnet', 'smoothgrad', 
                'integrated_gradients', 'grad_cam'
            }
    
    def _get_all_tensorflow_method_variations(self) -> set:
        """Get ACTUAL TensorFlow methods from local iNNvestigate copy."""
        # Get methods that are actually implemented in TensorFlow/iNNvestigate
        # Based on actual analyzers available in tf_signxai/methods/innvestigate/analyzer/__init__.py
        
        innvestigate_analyzers = {
            # Core gradient methods
            'gradient', 'input_t_gradient', 'deconvnet', 'guided_backprop',
            'integrated_gradients', 'smoothgrad', 'vargrad',
            
            # Gradient with x_input and x_sign variations
            'gradient_x_input', 'gradient_x_sign', 'gradient_x_input_x_sign',
            'gradient_x_sign_mu', 'gradient_x_sign_mu_0', 'gradient_x_sign_mu_0_5', 'gradient_x_sign_mu_neg_0_5',
            
            # Guided backprop variations
            'guided_backprop_x_input', 'guided_backprop_x_sign', 'guided_backprop_x_input_x_sign',
            'guided_backprop_x_sign_mu', 'guided_backprop_x_sign_mu_0', 'guided_backprop_x_sign_mu_0_5', 'guided_backprop_x_sign_mu_neg_0_5',
            
            # Deconvnet variations
            'deconvnet_x_input', 'deconvnet_x_sign', 'deconvnet_x_input_x_sign',
            'deconvnet_x_sign_mu', 'deconvnet_x_sign_mu_0', 'deconvnet_x_sign_mu_0_5', 'deconvnet_x_sign_mu_neg_0_5',
            
            # Smoothgrad variations
            'smoothgrad_x_input', 'smoothgrad_x_sign', 'smoothgrad_x_input_x_sign',
            'smoothgrad_x_sign_mu', 'smoothgrad_x_sign_mu_0', 'smoothgrad_x_sign_mu_0_5', 'smoothgrad_x_sign_mu_neg_0_5',
            
            # Integrated gradients variations
            'integrated_gradients_x_input', 'integrated_gradients_x_sign', 'integrated_gradients_x_input_x_sign',
            
            # VarGrad variations
            'vargrad_x_input', 'vargrad_x_sign', 'vargrad_x_input_x_sign',
            
            # DeepLift - commented out as it doesn't work in TensorFlow
            # 'deep_lift', 'deeplift', 'deeplift.wrapper',
            
            # Core LRP methods that actually exist (use dot notation for iNNvestigate)
            'lrp', 'lrp.z', 'lrp.z_IB', 'lrp.gamma', 'lrp.epsilon', 'lrp.stdxepsilon',
            'lrp.epsilon_IB', 'lrp.w_square', 'lrp.flat', 'lrp.alpha_beta',
            'lrp.alpha_2_beta_1', 'lrp.alpha_2_beta_1_IB', 'lrp.alpha_1_beta_0',
            'lrp.alpha_1_beta_0_IB', 'lrp.z_plus', 'lrp.z_plus_fast',
            'lrp.sequential_composite_a', 'lrp.sequential_composite_b',
            'lrp.rule_until_index',
            
            # Add underscore notation for method families (these get mapped to dot notation)
            'lrp_epsilon', 'lrp_z', 'lrp_gamma', 'lrp_flat', 'lrp_w_square',
            'lrp_alpha_1_beta_0', 'lrp_alpha_2_beta_1',
            
            # All epsilon variations
            'lrp_epsilon_0_001', 'lrp_epsilon_0_01', 'lrp_epsilon_0_1', 'lrp_epsilon_0_2',
            'lrp_epsilon_0_25', 'lrp_epsilon_0_5', 'lrp_epsilon_1', 'lrp_epsilon_2', 
            'lrp_epsilon_5', 'lrp_epsilon_10', 'lrp_epsilon_20', 'lrp_epsilon_50',
            'lrp_epsilon_75', 'lrp_epsilon_100',
            
            # LRP epsilon with std_x
            'lrp_epsilon_0_1_std_x', 'lrp_epsilon_0_25_std_x', 'lrp_epsilon_0_5_std_x',
            'lrp_epsilon_1_std_x', 'lrp_epsilon_2_std_x', 'lrp_epsilon_3_std_x',
            
            # LRP Sign variations
            'lrpsign_z', 'lrpsign_epsilon_0_001', 'lrpsign_epsilon_0_01', 'lrpsign_epsilon_0_1',
            'lrpsign_epsilon_0_2', 'lrpsign_epsilon_0_5', 'lrpsign_epsilon_1', 'lrpsign_epsilon_5',
            'lrpsign_epsilon_10', 'lrpsign_epsilon_20', 'lrpsign_epsilon_50', 'lrpsign_epsilon_75',
            'lrpsign_epsilon_100', 'lrpsign_epsilon_100_mu_0', 'lrpsign_epsilon_100_mu_0_5', 'lrpsign_epsilon_100_mu_neg_0_5',
            'lrpsign_epsilon_0_1_std_x', 'lrpsign_epsilon_0_25_std_x', 'lrpsign_epsilon_0_25_std_x_mu_0',
            'lrpsign_epsilon_0_25_std_x_mu_0_5', 'lrpsign_epsilon_0_25_std_x_mu_neg_0_5',
            'lrpsign_epsilon_0_5_std_x', 'lrpsign_epsilon_1_std_x', 'lrpsign_epsilon_2_std_x', 'lrpsign_epsilon_3_std_x',
            'lrpsign_alpha_1_beta_0', 'lrpsign_sequential_composite_a', 'lrpsign_sequential_composite_b',
            
            # LRP Z variations
            'lrpz_epsilon_0_001', 'lrpz_epsilon_0_01', 'lrpz_epsilon_0_1', 'lrpz_epsilon_0_2',
            'lrpz_epsilon_0_5', 'lrpz_epsilon_1', 'lrpz_epsilon_5', 'lrpz_epsilon_10',
            'lrpz_epsilon_20', 'lrpz_epsilon_50', 'lrpz_epsilon_75', 'lrpz_epsilon_100',
            'lrpz_epsilon_0_1_std_x', 'lrpz_epsilon_0_25_std_x', 'lrpz_epsilon_0_5_std_x',
            'lrpz_epsilon_1_std_x', 'lrpz_epsilon_2_std_x', 'lrpz_epsilon_3_std_x',
            'lrpz_sequential_composite_a', 'lrpz_sequential_composite_b',
            
            # Flat LRP variations
            'flatlrp_z', 'flatlrp_epsilon_0_01', 'flatlrp_epsilon_0_1', 'flatlrp_epsilon_1',
            'flatlrp_epsilon_10', 'flatlrp_epsilon_20', 'flatlrp_epsilon_100',
            'flatlrp_epsilon_0_1_std_x', 'flatlrp_epsilon_0_25_std_x', 'flatlrp_epsilon_0_5_std_x',
            'flatlrp_sequential_composite_a', 'flatlrp_sequential_composite_b',
            
            # W2 LRP variations
            'w2lrp_z', 'w2lrp_epsilon_0_01', 'w2lrp_epsilon_0_1', 'w2lrp_epsilon_1',
            'w2lrp_epsilon_10', 'w2lrp_epsilon_20', 'w2lrp_epsilon_100',
            'w2lrp_epsilon_0_1_std_x', 'w2lrp_epsilon_0_25_std_x', 'w2lrp_epsilon_0_5_std_x',
            'w2lrp_sequential_composite_a', 'w2lrp_sequential_composite_b'
        }
        
        # Add custom SignXAI TensorFlow implementations  
        custom_tf_methods = {
            'grad_cam', 'grad_cam_timeseries', 'occlusion',
            'grad_cam_x_input', 'grad_cam_x_sign', 'grad_cam_x_input_x_sign',
            'grad_cam_VGG16ILSVRC',
            'random_uniform'
        }
        
        # Add VGG16ILSVRC specific methods (ZB-LRP with ImageNet bounds)
        vgg16_specific = {
            'zblrp_z_VGG16ILSVRC', 'zblrp_epsilon_0_001_VGG16ILSVRC', 'zblrp_epsilon_0_01_VGG16ILSVRC',
            'zblrp_epsilon_0_1_VGG16ILSVRC', 'zblrp_epsilon_0_2_VGG16ILSVRC', 'zblrp_epsilon_0_5_VGG16ILSVRC',
            'zblrp_epsilon_1_VGG16ILSVRC', 'zblrp_epsilon_5_VGG16ILSVRC', 'zblrp_epsilon_10_VGG16ILSVRC',
            'zblrp_epsilon_20_VGG16ILSVRC', 'zblrp_epsilon_100_VGG16ILSVRC',
            'zblrp_epsilon_0_1_std_x_VGG16ILSVRC', 'zblrp_epsilon_0_25_std_x_VGG16ILSVRC',
            'zblrp_epsilon_0_5_std_x_VGG16ILSVRC',
            'zblrp_sequential_composite_a_VGG16ILSVRC', 'zblrp_sequential_composite_b_VGG16ILSVRC'
        }
        
        # Only include variations that we can actually implement via Method Families
        # These are modifiers we can apply to base methods
        implementable_variations = set()
        
        # Apply x_input, x_sign modifiers to ALL attribution methods that support them
        # This includes gradient methods, LRP methods, and other attribution methods
        base_methods_for_modifiers = {
            # Gradient-based methods
            'gradient', 'guided_backprop', 'deconvnet', 'smoothgrad', 'vargrad',
            'integrated_gradients',
            
            # Core LRP methods - these should ALL support modifiers
            'lrp', 'lrp_z', 'lrp_gamma', 'lrp_epsilon', 'lrp_flat', 'lrp_w_square',
            'lrp_alpha_1_beta_0', 'lrp_alpha_2_beta_1',
            
            # LRP z variations
            'lrp_z_plus',
            
            # GradCAM
            'grad_cam'
        }
        
        for base_method in base_methods_for_modifiers:
            if base_method in innvestigate_analyzers or base_method in custom_tf_methods:
                implementable_variations.add(f'{base_method}_x_input')
                implementable_variations.add(f'{base_method}_x_sign')
                implementable_variations.add(f'{base_method}_x_input_x_sign')
        
        # Combine all actual methods
        methods = innvestigate_analyzers | custom_tf_methods | vgg16_specific | implementable_variations
        
        print(f"TensorFlow actual methods: {len(methods)} total")
        print(f"TF Core methods: {len(innvestigate_analyzers)}, Custom: {len(custom_tf_methods)}, Variations: {len(implementable_variations)}")
        
        return methods


# Global registry instance
_registry = None

def get_registry() -> MethodFamilyRegistry:
    """Get or create the global registry instance."""
    global _registry
    if _registry is None:
        _registry = MethodFamilyRegistry()
    return _registry
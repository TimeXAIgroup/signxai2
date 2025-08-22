# signxai/common/method_parser.py
import re
from typing import Dict, Any, List, Optional, Tuple


class MethodParser:
    """
    Parses XAI method strings into a structured format with comprehensive parameter extraction.
    
    Handles complex method names like:
    - lrp_alpha_2_beta_1 -> lrp with alpha=2, beta=1
    - gradient_x_sign_mu_neg_0_5 -> gradient with sign modifier and mu=-0.5
    - lrp_epsilon_0_25_std_x -> lrp_epsilon with epsilon=0.25 and std_x modifier
    """

    def parse(self, method_name: str) -> Dict[str, Any]:
        """
        Parses a method string into base method, modifiers, and parameters.

        Args:
            method_name (str): The method string.

        Returns:
            A dictionary with:
                - base_method: The canonical base method name
                - modifiers: List of modifiers (x_input, x_sign, std_x, etc.)
                - params: Extracted parameters with proper types
                - original_name: The original method name
        """
        original = method_name
        method_lower = method_name.lower()
        
        # First, check for special compound methods
        base_method, params = self._extract_compound_method(method_lower)
        
        # Extract modifiers
        modifiers = self._extract_modifiers(method_lower)
        
        # Extract additional parameters not covered by compound extraction
        additional_params = self._extract_parameters(method_lower)
        params.update(additional_params)
        
        return {
            'base_method': base_method,
            'modifiers': modifiers,
            'params': params,
            'original_name': original
        }
    
    def _extract_compound_method(self, method: str) -> Tuple[str, Dict[str, Any]]:
        """Extract base method from compound patterns like lrp_alpha_X_beta_Y."""
        params = {}
        
        # LRP Alpha-Beta pattern (e.g., lrp_alpha_2_beta_1)
        alpha_beta_match = re.match(r'^([\w]+?)_alpha_(\d+)_beta_(\d+)', method)
        if alpha_beta_match:
            prefix = alpha_beta_match.group(1)
            params['alpha'] = float(alpha_beta_match.group(2))
            params['beta'] = float(alpha_beta_match.group(3))
            
            # Special handling for specific alpha-beta combinations
            if params['alpha'] == 1 and params['beta'] == 0:
                return f"{prefix}_alpha_1_beta_0", params
            elif params['alpha'] == 2 and params['beta'] == 1:
                return f"{prefix}_alpha_2_beta_1", params
            else:
                return f"{prefix}_alpha_beta", params
        
        # LRP Sequential Composite patterns
        if 'sequential_composite_a' in method:
            base = method.split('_sequential_composite_a')[0]
            return f"{base}_sequential_composite_a", params
        elif 'sequential_composite_b' in method:
            base = method.split('_sequential_composite_b')[0]
            return f"{base}_sequential_composite_b", params
        
        # Default: extract first component as base
        parts = method.split('_')
        base_method = parts[0]
        
        # Handle multi-part base methods
        if len(parts) > 1:
            # Check for known multi-part bases
            if parts[0] in ['lrp', 'lrpsign', 'lrpz', 'flatlrp', 'w2lrp', 'zblrp']:
                if parts[1] in ['epsilon', 'gamma', 'z', 'flat', 'w', 'alpha', 'sequential']:
                    if parts[1] == 'w' and len(parts) > 2 and parts[2] == 'square':
                        base_method = f"{parts[0]}_w_square"
                    elif parts[1] == 'sequential':
                        # Already handled above
                        pass
                    elif parts[1] in ['epsilon', 'gamma'] and len(parts) > 2:
                        # Extract parameter value for epsilon/gamma methods
                        base_method = f"{parts[0]}_{parts[1]}"
                        # Try to extract the parameter value
                        param_parts = []
                        i = 2
                        while i < len(parts) and parts[i] not in ['x', 'times', 'std', 'ib', 'timeseries']:
                            param_parts.append(parts[i])
                            i += 1
                        if param_parts:
                            # Convert underscores back to decimal points
                            param_str = '_'.join(param_parts)
                            if len(param_parts) == 1:
                                params[parts[1]] = float(param_parts[0])
                            elif len(param_parts) == 2:
                                params[parts[1]] = float(f"{param_parts[0]}.{param_parts[1]}")
                    else:
                        base_method = f"{parts[0]}_{parts[1]}"
            elif parts[0] == 'integrated' and len(parts) > 1 and parts[1] == 'gradients':
                base_method = 'integrated_gradients'
            elif parts[0] == 'guided' and len(parts) > 1 and parts[1] == 'backprop':
                base_method = 'guided_backprop'
            elif parts[0] == 'grad' and len(parts) > 1 and parts[1] == 'cam':
                base_method = 'grad_cam'
            elif parts[0] == 'deep' and len(parts) > 1 and (parts[1] == 'lift' or parts[1] == 'taylor'):
                base_method = f"{parts[0]}_{parts[1]}"
            elif parts[0] == 'input' and len(parts) > 1 and parts[1] == 't' and len(parts) > 2 and parts[2] == 'gradient':
                base_method = 'input_t_gradient'
        
        return base_method, params
    
    def _extract_modifiers(self, method: str) -> List[str]:
        """Extract modifiers like x_input, x_sign, std_x, etc."""
        modifiers = []
        
        # Check for input modifier
        if '_x_input' in method or '_times_input' in method:
            modifiers.append('x_input')
        
        # Check for sign modifier
        if '_x_sign' in method:
            modifiers.append('x_sign')
        
        # Check for std_x modifier
        if '_std_x' in method:
            modifiers.append('std_x')
        
        # Check for IB (ignore bias) modifier
        if '_ib' in method.lower() or method.endswith('_ib'):
            modifiers.append('ignore_bias')
        
        # Check for timeseries modifier
        if '_timeseries' in method:
            modifiers.append('timeseries')
        
        return modifiers
    
    def _extract_parameters(self, method: str) -> Dict[str, Any]:
        """Extract numerical parameters from method name."""
        params = {}
        
        # Epsilon values (e.g., epsilon_0_1 -> 0.1, epsilon_0_25 -> 0.25)
        epsilon_match = re.search(r'epsilon_(\d+)(?:_(\d+))?(?![_\d])', method)
        if epsilon_match and 'alpha' not in method:  # Avoid matching in alpha_beta patterns
            whole = int(epsilon_match.group(1))
            decimal = epsilon_match.group(2)
            if decimal:
                params['epsilon'] = float(f"{whole}.{decimal}")
            else:
                params['epsilon'] = float(whole)
        
        # Mu values (e.g., mu_0_5 -> 0.5, mu_neg_0_5 -> -0.5)
        mu_match = re.search(r'mu_(neg_)?(\d+)(?:_(\d+))?(?![_\d])', method)
        if mu_match:
            is_negative = bool(mu_match.group(1))
            whole = int(mu_match.group(2))
            decimal = mu_match.group(3)
            if decimal:
                value = float(f"{whole}.{decimal}")
            else:
                value = float(whole)
            params['mu'] = -value if is_negative else value
        
        # Gamma values
        gamma_match = re.search(r'gamma_(\d+)(?:_(\d+))?(?![_\d])', method)
        if gamma_match:
            whole = int(gamma_match.group(1))
            decimal = gamma_match.group(2)
            if decimal:
                params['gamma'] = float(f"{whole}.{decimal}")
            else:
                params['gamma'] = float(whole)
        
        # Steps for integrated gradients
        steps_match = re.search(r'steps_(\d+)', method)
        if steps_match:
            params['steps'] = int(steps_match.group(1))
        
        # Noise level for smoothgrad/vargrad
        noise_match = re.search(r'noise_(\d+)(?:_(\d+))?', method)
        if noise_match:
            whole = int(noise_match.group(1))
            decimal = noise_match.group(2)
            if decimal:
                params['noise_level'] = float(f"{whole}.{decimal}")
            else:
                params['noise_level'] = float(whole)
        
        # Number of samples
        samples_match = re.search(r'(?:samples|num_samples)_(\d+)', method)
        if samples_match:
            params['num_samples'] = int(samples_match.group(1))
        
        # Stdfactor for LRP methods
        stdfactor_match = re.search(r'stdfactor_(\d+)(?:_(\d+))?', method)
        if stdfactor_match:
            whole = int(stdfactor_match.group(1))
            decimal = stdfactor_match.group(2)
            if decimal:
                params['stdfactor'] = float(f"{whole}.{decimal}")
            else:
                params['stdfactor'] = float(whole)
        
        return params
    
    def _is_param(self, part: str) -> bool:
        """
        Checks if a part of the method string is a parameter name.
        """
        return part in ['epsilon', 'mu', 'alpha', 'beta', 'steps', 'noise_level', 
                       'num_samples', 'stdfactor', 'gamma', 'noise', 'samples']

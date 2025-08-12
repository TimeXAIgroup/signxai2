# signxai/common/method_parser.py
import re
from typing import Dict, Any


class MethodParser:
    """
    Parses XAI method strings into a structured format.
    """

    def parse(self, method_name: str) -> Dict[str, Any]:
        """
        Parses a method string like 'lrpsign_epsilon_40_mu_0_5'.

        Args:
            method_name (str): The method string.

        Returns:
            A dictionary with the parsed components.
        """
        parts = method_name.lower().split('_')
        base_method = parts[0]

        # Handle special cases and modifiers
        modifiers = [p for p in parts[1:] if not self._is_param(p)]

        # Extract parameters
        params = {}
        i = 1
        while i < len(parts):
            part = parts[i]
            if self._is_param(part):
                param_name = part
                # Check for negative values
                if i + 2 < len(parts) and parts[i + 1] == 'neg':
                    param_value = f"-{parts[i + 2]}"
                    i += 2
                elif i + 1 < len(parts):
                    param_value = parts[i + 1]
                    i += 1
                else:
                    param_value = None

                if param_value:
                    try:
                        params[param_name] = float(param_value.replace('p', '.'))
                    except (ValueError, TypeError):
                        # Handle cases where value is not a number
                        params[param_name] = param_value
            i += 1

        return {
            'base_method': base_method,
            'modifiers': modifiers,
            'params': params,
            'original_name': method_name
        }

    def _is_param(self, part: str) -> bool:
        """
        Checks if a part of the method string is a parameter name.
        """
        return part in ['epsilon', 'mu', 'alpha', 'beta', 'steps', 'noise_level', 'num_samples', 'stdfactor']

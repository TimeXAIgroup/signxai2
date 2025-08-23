"""
Method name normalizer to handle naming inconsistencies between frameworks.

This module provides canonical method naming and aliasing support to ensure
consistent method names across TensorFlow and PyTorch implementations.
"""

from typing import Dict, Optional, Set


class MethodNormalizer:
    """Normalize method names across frameworks to handle inconsistencies."""

    # Default parameters for methods
    METHOD_PRESETS: Dict[str, Dict] = {
        'smoothgrad': {
            'noise_level': 0.1,
            'num_samples': 25
        },
        'integrated_gradients': {
            'steps': 50,
            'baseline': None
        },
        'vargrad': {
            'noise_level': 0.1,
            'num_samples': 25
        },
        'lrp_epsilon': {
            'epsilon': 0.1
        },
        'lrp_alpha_1_beta_0': {
            'alpha': 1.0,
            'beta': 0.0
        },
        'lrp_alpha_2_beta_1': {
            'alpha': 2.0,
            'beta': 1.0
        }
    }

    # Canonical names mapping (aliases -> canonical)
    # Using underscored versions as the canonical names for proper TensorFlow/PyTorch compatibility
    CANONICAL_NAMES: Dict[str, str] = {
        # Integrated gradients variations (integrated_gradients is canonical)
        'integratedgradients': 'integrated_gradients',
        'integrated_gradient': 'integrated_gradients',
        'integratedgradient': 'integrated_gradients',
        'integrated_gradients': 'integrated_gradients',

        # Grad-CAM variations (grad_cam is canonical for consistency)
        'gradcam': 'grad_cam',
        'grad_cam': 'grad_cam',
        'gradCAM': 'grad_cam',
        'GradCAM': 'grad_cam',

        # Gradient variations (keep both for now, they may have different behaviors)
        'gradients': 'gradient',  # Map plural to singular as default
        'gradient': 'gradient',

        # DeepLift variations
        'deeplift_method': 'deeplift',
        'deeplift': 'deeplift',
        'deep_lift': 'deeplift',

        # LRP variations
        'lrp': 'lrp',
        'LRP': 'lrp',

        # Guided backprop variations
        'guided_backprop': 'guided_backprop',
        'guidedbackprop': 'guided_backprop',
        'guided_backpropagation': 'guided_backprop',

        # Deconvnet variations
        'deconvnet': 'deconvnet',
        'deconvolution': 'deconvnet',
        'deconv': 'deconvnet',
    }

    # Methods that are framework-specific (not available in both)
    FRAMEWORK_SPECIFIC: Dict[str, Set[str]] = {
        'pytorch': {
            'lrp_z_x_input',
            'lrp_z_x_input_x_sign',
            'lrp_z_x_sign',
            'lrpsign_epsilon_100_improved',
            'lrpsign_epsilon_20_improved',
            'flatlrp_z',
        },
        'tensorflow': set(),  # Currently no TF-only methods
    }

    # Methods that are known to be broken or disabled
    DISABLED_METHODS: Set[str] = {
        'deconvnet_x_input_DISABLED_BROKEN_WRAPPER',
        'deconvnet_x_input_x_sign_DISABLED_BROKEN_WRAPPER',
        'deconvnet_x_sign_DISABLED_BROKEN_WRAPPER',
        'deconvnet_x_sign_mu_0_5_DISABLED_BROKEN_WRAPPER',
        'smoothgrad_x_input_DISABLED_BROKEN_WRAPPER',
    }

    @classmethod
    def normalize(cls, method_name: str, framework: Optional[str] = None) -> str:
        """
        Normalize a method name to its canonical form.

        Args:
            method_name: The method name to normalize
            framework: The framework being used ('tensorflow' or 'pytorch')

        Returns:
            The canonical method name

        Raises:
            ValueError: If the method is disabled or not available in the framework
        """
        # Check if method is disabled
        if method_name in cls.DISABLED_METHODS:
            raise ValueError(f"Method '{method_name}' is currently disabled/broken")

        # Strip any DISABLED suffix if present
        if '_DISABLED_BROKEN_WRAPPER' in method_name:
            raise ValueError(f"Method '{method_name}' is marked as broken")

        # First, check if this is already a canonical name
        if method_name in cls.CANONICAL_NAMES.values():
            canonical = method_name
        else:
            # Try to find the base method name (before parameters/transformations)
            base_method = cls._extract_base_method(method_name)

            # Check if base method has a canonical form
            if base_method in cls.CANONICAL_NAMES:
                # Replace the base with canonical, keep the rest
                canonical_base = cls.CANONICAL_NAMES[base_method]
                canonical = method_name.replace(base_method, canonical_base, 1)
            else:
                # No mapping found, use as-is
                canonical = method_name

        # Check framework availability if specified
        if framework:
            framework = framework.lower()
            if framework in cls.FRAMEWORK_SPECIFIC:
                # Check if this method is specific to another framework
                other_frameworks = [fw for fw in cls.FRAMEWORK_SPECIFIC if fw != framework]
                for other_fw in other_frameworks:
                    if canonical in cls.FRAMEWORK_SPECIFIC[other_fw]:
                        raise ValueError(
                            f"Method '{canonical}' is not available in {framework}, "
                            f"only in {other_fw}"
                        )

        return canonical

    @classmethod
    def _extract_base_method(cls, method_name: str) -> str:
        """
        Extract the base method name without parameters or transformations.

        Args:
            method_name: Full method name

        Returns:
            Base method name
        """
        # Common patterns to remove
        # Remove x_input, x_sign, x_input_x_sign transformations
        base = method_name
        for transform in ['_x_input_x_sign', '_x_input', '_x_sign']:
            if transform in base:
                base = base.split(transform)[0]
                break

        # Remove parameter values (e.g., _0_1, _0_25, etc.)
        import re
        # Remove patterns like _0_1, _0_25, _100, etc.
        base = re.sub(r'_\d+(_\d+)*$', '', base)

        # Remove mu parameters
        base = re.sub(r'_mu(_\d+(_\d+)?)?$', '', base)

        # Remove model-specific suffixes
        models = ['_VGG16ILSVRC', '_VGG16', '_ResNet50', '_MNISTCNN']
        for model in models:
            if base.endswith(model):
                base = base[:-len(model)]
                break

        return base

    @classmethod
    def get_aliases(cls, canonical_name: str) -> Set[str]:
        """
        Get all known aliases for a canonical method name.

        Args:
            canonical_name: The canonical method name

        Returns:
            Set of all aliases (including the canonical name itself)
        """
        aliases = {canonical_name}
        for alias, canonical in cls.CANONICAL_NAMES.items():
            if canonical == canonical_name:
                aliases.add(alias)
        return aliases

    @classmethod
    def is_framework_specific(cls, method_name: str, framework: str) -> bool:
        """
        Check if a method is specific to a particular framework.

        Args:
            method_name: The method name to check
            framework: The framework to check against

        Returns:
            True if the method is specific to the given framework
        """
        framework = framework.lower()
        canonical = cls.normalize(method_name)
        return canonical in cls.FRAMEWORK_SPECIFIC.get(framework, set())

    @classmethod
    def get_framework_methods(cls, framework: str) -> Set[str]:
        """
        Get all framework-specific methods for a given framework.

        Args:
            framework: The framework name

        Returns:
            Set of framework-specific method names
        """
        return cls.FRAMEWORK_SPECIFIC.get(framework.lower(), set())
import numpy as np
import torch
from zennit.composites import register_composite, SpecialFirstLayerMapComposite, layer_map_base
from zennit.core import BasicHook, Stabilizer
from zennit.rules import NoMod
from zennit.types import Convolution, Linear


def sign_mu(x, mu=0, vlow=-1, vhigh=1):
    x[x < mu] = vlow
    x[x >= mu] = vhigh
    return x


def normalize_heatmap(h):
    h = np.mean(h, axis=2)
    h = h / np.max(np.abs(h.ravel()))
    return h


def filter_heatmap(h, posthresh=0.1, cmap_adjust=0.1):
    # Use only positives
    h[h < 0] = 0

    # Normalize relevance map
    h = normalize_heatmap(h)

    # Discard values <= posthresh
    h[h <= posthresh] = 0

    # Amplify positives for better visualisation
    h[h > posthresh] = h[h > posthresh] + cmap_adjust

    return h

def gradient_x_sign(x, h):
    s = np.nan_to_num(x / np.abs(x))
    return h * s

class EpsilonStdX(BasicHook):
    """LRP Epsilon rule :cite:p:`bach2015pixel`.
    Setting ``(epsilon=0)`` produces the LRP-0 rule :cite:p:`bach2015pixel`.
    LRP Epsilon is most commonly used in middle layers, LRP-0 is most commonly used in upper layers
    :cite:p:`montavon2019layer`.
    Sometimes higher values of ``epsilon`` are used, therefore it is not always only a stabilizer value.

    Std-x-Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/tutorial.ipynb

    Parameters
    ----------
    stdfactor: float, optional
        Stabilization parameter for multiplication with std(inputs).
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    """

    def extract_eps(self, x):
        self.epsilon = float(np.std(x.cpu().detach().numpy()) * self.stdfactor)
        return x

    def __init__(self, stdfactor=0.25, zero_params=None):
        self.epsilon = None
        self.stdfactor = stdfactor

        super().__init__(
            input_modifiers=[lambda input: self.extract_eps(input)],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / Stabilizer.ensure(self.epsilon)(outputs[0])),
            reducer=(lambda inputs, gradients: inputs[0] * gradients[0]),
        )

class EpsStdXSIGN(BasicHook):
    """ Epsilon (stdx) + SIGN rule

    Std-x-Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/tutorial.ipynb)

    Parameters
    ----------
    mu: float, optional
        expected value of the input distribution (for zero-centered scenarios, mu is 0)
    stdfactor: float, optional
        Stabilization parameter for multiplication with std(inputs).
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    """

    def extract_eps(self, x):
        self.epsilon = float(np.std(x.cpu().detach().numpy()) * self.stdfactor)
        return x

    def __init__(self, mu=0, stdfactor=0.25, zero_params=None):
        self.epsilon = None
        self.stdfactor = stdfactor

        super().__init__(
            input_modifiers=[lambda input: self.extract_eps(input)],
            param_modifiers=[NoMod(zero_params=zero_params)],
            output_modifiers=[lambda output: output],
            gradient_mapper=(lambda out_grad, outputs: out_grad / Stabilizer.ensure(self.epsilon)(outputs[0])),
            reducer=(lambda inputs, gradients: sign_mu(inputs[0], mu=mu) * gradients[0]),
        )

@register_composite('epsilon_stdx_comp')
class EpsilonStdXComp(SpecialFirstLayerMapComposite):
    """ Epsilon with std(x) composite.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    """
    def __init__(
        self, stabilizer=1e-6, stdfactor=0.25, layer_map=None, first_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, EpsilonStdX(stdfactor=stdfactor, **rule_kwargs)),
            (torch.nn.Linear, EpsilonStdX(stdfactor=stdfactor, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Linear, EpsilonStdX(stdfactor=stdfactor, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


@register_composite('epsilon_stdx_sign')
class EpsilonStdXSIGN(SpecialFirstLayerMapComposite):
    """ SIGN composite.

    Parameters
    ----------
    epsilon: callable or float, optional
        Stabilization parameter for the ``Epsilon`` rule. If ``epsilon`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator. Note that this is
        called ``stabilizer`` for all other rules.
    stabilizer: callable or float, optional
        Stabilization parameter for rules other than ``Epsilon``. If ``stabilizer`` is a float, it will be added to the
        denominator with the same sign as each respective entry. If it is callable, a function ``(input: torch.Tensor)
        -> torch.Tensor`` is expected, of which the output corresponds to the stabilized denominator.
    layer_map: list[tuple[tuple[torch.nn.Module, ...], Hook]]
        A mapping as a list of tuples, with a tuple of applicable module types and a Hook. This will be prepended to
        the ``layer_map`` defined by the composite.
    first_map: `list[tuple[tuple[torch.nn.Module, ...], Hook]]`
        Applicable mapping for the first layer, same format as `layer_map`. This will be prepended to the ``first_map``
        defined by the composite.
    zero_params: list[str], optional
        A list of parameter names that shall set to zero. If `None` (default), no parameters are set to zero.
    canonizers: list[:py:class:`zennit.canonizers.Canonizer`], optional
        List of canonizer instances to be applied before applying hooks.
    mu: float, optional
        expected value of the input distribution (for zero-centered scenarios, mu is 0)
    """
    def __init__(
        self, stabilizer=1e-6, signstdfactor=0.25, stdfactor=0.25, mu=0, layer_map=None, first_map=None, zero_params=None, canonizers=None
    ):
        if layer_map is None:
            layer_map = []
        if first_map is None:
            first_map = []

        rule_kwargs = {'zero_params': zero_params}
        layer_map = layer_map + layer_map_base(stabilizer) + [
            (Convolution, EpsilonStdX(stdfactor=stdfactor, **rule_kwargs)),
            (torch.nn.Linear, EpsilonStdX(stdfactor=stdfactor, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Convolution, EpsStdXSIGN(mu=mu, stdfactor=signstdfactor, **rule_kwargs))
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)


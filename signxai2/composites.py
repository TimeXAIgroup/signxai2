import torch
from zennit.composites import register_composite, SpecialFirstLayerMapComposite, layer_map_base
from zennit.types import Convolution, Linear
from signxai2.rules import EpsilonStdXRule, EpsStdXSIGNRule


@register_composite('epsilon_stdx_comp')
class EpsilonStdX(SpecialFirstLayerMapComposite):
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
            (Convolution, EpsilonStdXRule(stdfactor=stdfactor, **rule_kwargs)),
            (torch.nn.Linear, EpsilonStdXRule(stdfactor=stdfactor, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Convolution, EpsilonStdXRule(stdfactor=stdfactor, **rule_kwargs)),
            (torch.nn.Linear, EpsilonStdXRule(stdfactor=stdfactor, **rule_kwargs)),
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
            (Convolution, EpsilonStdXRule(stdfactor=stdfactor, **rule_kwargs)),
            (torch.nn.Linear, EpsilonStdXRule(stdfactor=stdfactor, **rule_kwargs)),
        ]
        first_map = first_map + [
            (Convolution, EpsStdXSIGNRule(mu=mu, stdfactor=signstdfactor, **rule_kwargs)),
            (torch.nn.Linear, EpsStdXSIGNRule(mu=mu, stdfactor=signstdfactor, **rule_kwargs)),
        ]
        super().__init__(layer_map=layer_map, first_map=first_map, canonizers=canonizers)
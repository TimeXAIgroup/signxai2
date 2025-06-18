# Get Python six functionality:
from __future__ import \
    absolute_import, print_function, division, unicode_literals

###############################################################################
###############################################################################
###############################################################################

from .base import NotAnalyzeableModelException
from .base import ReverseAnalyzerBase
from .deeplift import DeepLIFTWrapper
from .gradient_based import Gradient, VarGrad
from .gradient_based import InputTimesGradient
from .gradient_based import GuidedBackprop
from .gradient_based import Deconvnet
from .gradient_based import IntegratedGradients
from .gradient_based import SmoothGrad
from .relevance_based.relevance_analyzer import LRP, LRPStdxEpsilon
from .relevance_based.relevance_analyzer import LRPZ
from .relevance_based.relevance_analyzer import LRPZIgnoreBias
from .relevance_based.relevance_analyzer import LRPZPlus
from .relevance_based.relevance_analyzer import LRPZPlusFast
from .relevance_based.relevance_analyzer import LRPEpsilon
from .relevance_based.relevance_analyzer import LRPEpsilonIgnoreBias
from .relevance_based.relevance_analyzer import LRPWSquare
from .relevance_based.relevance_analyzer import LRPFlat
from .relevance_based.relevance_analyzer import LRPAlphaBeta
from .relevance_based.relevance_analyzer import LRPGamma
from .relevance_based.relevance_analyzer import LRPAlpha2Beta1
from .relevance_based.relevance_analyzer import LRPAlpha2Beta1IgnoreBias
from .relevance_based.relevance_analyzer import LRPAlpha1Beta0
from .relevance_based.relevance_analyzer import LRPAlpha1Beta0IgnoreBias
from .relevance_based.relevance_analyzer import LRPSequentialCompositeA
from .relevance_based.relevance_analyzer import LRPSequentialCompositeB
from .relevance_based.relevance_analyzer import LRPSequentialCompositeAFlat
from .relevance_based.relevance_analyzer import LRPSequentialCompositeBFlat
from .relevance_based.relevance_analyzer import LRPRuleUntilIndex
from .wrapper import WrapperBase
from .wrapper import AugmentReduceBase
from .wrapper import GaussianSmoother
from .wrapper import PathIntegrator

# Disable pyflaks warnings:
assert NotAnalyzeableModelException


###############################################################################
###############################################################################
###############################################################################


analyzers = {
    # Gradient based
    "gradient": Gradient,
    "input_t_gradient": InputTimesGradient,
    "deconvnet": Deconvnet,
    "guided_backprop": GuidedBackprop,
    "integrated_gradients": IntegratedGradients,
    "smoothgrad": SmoothGrad,
    "vargrad": VarGrad,

    # Other
    "deep_lift": DeepLIFTWrapper,

    # Relevance based
    "lrp": LRP,

    "lrp.z": LRPZ,
    "lrp.z_IB": LRPZIgnoreBias,
    "lrp.gamma": LRPGamma,
    "lrp.epsilon": LRPEpsilon,
    "lrp.stdxepsilon": LRPStdxEpsilon,
    "lrp.epsilon_IB": LRPEpsilonIgnoreBias,

    "lrp.w_square": LRPWSquare,
    "lrp.flat": LRPFlat,

    "lrp.alpha_beta": LRPAlphaBeta,

    "lrp.alpha_2_beta_1": LRPAlpha2Beta1,
    "lrp.alpha_2_beta_1_IB": LRPAlpha2Beta1IgnoreBias,
    "lrp.alpha_1_beta_0": LRPAlpha1Beta0,
    "lrp.alpha_1_beta_0_IB": LRPAlpha1Beta0IgnoreBias,
    "lrp.z_plus": LRPZPlus,
    "lrp.z_plus_fast": LRPZPlusFast,

    "lrp.sequential_composite_a": LRPSequentialCompositeA,
    "lrp.sequential_composite_b": LRPSequentialCompositeB,

    "lrp.rule_until_index": LRPRuleUntilIndex,
}


def create_analyzer(name, model, **kwargs):
    """Instantiates the analyzer with the name 'name'

    This convenience function takes an analyzer name
    creates the respective analyzer.

    Alternatively analyzers can be created directly by
    instantiating the respective classes.

    :param name: Name of the analyzer.
    :param model: The model to analyze, passed to the analyzer's __init__.
    :param kwargs: Additional parameters for the analyzer's .
    :return: An instance of the chosen analyzer.
    :raise KeyError: If there is no analyzer with the passed name.
    """
    try:
        analyzer_class = analyzers[name]
    except KeyError:
        raise KeyError(
            "No analyzer with the name '%s' could be found."
            " All possible names are: %s" % (name, list(analyzers.keys())))
    return analyzer_class(model, **kwargs)

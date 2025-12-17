import numpy as np
from zennit.core import BasicHook, Stabilizer
from zennit.rules import NoMod
from signxai2.sign import sign_mu


class EpsilonStdXRule(BasicHook):
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


class EpsStdXSIGNRule(BasicHook):
    """ Epsilon (stdx) + SIGN rule

    Std-x-Source: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/blob/main/tutorial.ipynb

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

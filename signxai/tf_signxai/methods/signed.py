import numpy as np


def calculate_sign_mu(x, mu=None, vlow=-1, vhigh=1, **kwargs):
    assert vlow < vhigh
    s = np.array(x)

    if mu is None:
        mu = vlow + (abs(vhigh - vlow) / 2.0)

    s[s < mu] = vlow
    s[s >= mu] = vhigh

    return s

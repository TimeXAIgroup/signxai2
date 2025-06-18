import numpy as np


def normalize_ecg_relevancemap(R, local=False):
    if local is False:
        # Normalize R to [-1, 1]
        Rn = R / np.max(np.abs(R))
    else:
        # Normalize each lead in R to [-1, 1]
        Rn = np.zeros_like(R)
        for i in range(np.shape(R)[1]):
            Rn[..., i] = R[..., i] / np.max(np.abs(R[..., i]))

    # Replace any nan to 0
    Rn = np.nan_to_num(Rn, nan=0)

    return Rn
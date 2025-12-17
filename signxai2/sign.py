import numpy as np


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

def sign_mu(x, mu=0, vlow=-1, vhigh=1):
    x[x < mu] = vlow
    x[x >= mu] = vhigh
    return x

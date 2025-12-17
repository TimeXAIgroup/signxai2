import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import zennit
from signxai2.composites import EpsilonStdX, EpsilonStdXSIGN
from signxai2.dftutils import calculate_dft_explanation
from signxai2.misc import get_dft_example


class MLPModel(nn.Module):
    def __init__(self, signal_length, n_out, n_layer, use_dropout=True, p_drop=0.1):
        super().__init__()
        layers = [nn.Linear(signal_length, 2 * signal_length)]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(2 * signal_length, 2 * signal_length), nn.ReLU()]
            if use_dropout:
                layers.append(nn.Dropout(p_drop))
        layers.append(nn.Linear(2 * signal_length, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = MLPModel(
        signal_length=cfg["signal_length"],
        n_out=cfg["n_label"],
        n_layer=cfg["n_layer"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Loaded checkpoint: {path}")

    freqs_gt = [5, 16, 32, 53]
    print(f"Ground truth frequencies: {freqs_gt}")

    return model, cfg["signal_length"], cfg["n_label"], freqs_gt


def run_tutorial(attribution_method="lrpsign", onlypos=True, cuda=False, k_max=100):
    np.random.seed(2)
    device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

    model, signal_length, n_label, freqs_gt = load_checkpoint("frequency_model.pt", device=device)
    signal_time = torch.tensor(np.array([np.load('signal.npy')[0]]), dtype=torch.float)
    target_index = np.load('y_test.npy')[0]
    target = torch.eye(n_label)[[target_index]]

    if attribution_method == "lrpepsstdx":
        composite = EpsilonStdX(stdfactor=0.1)
    elif attribution_method == "lrpepsstdxsign":
        composite = EpsilonStdXSIGN(mu=0, stdfactor=0.1, signstdfactor=0.1)
    else:
        raise Exception("Unknown method: " + attribution_method)

    with zennit.attribution.Gradient(model=model, composite=composite) as attributor:
        _, attribution = attributor(signal_time, target)
        relevance_time = attribution.cpu().numpy()

    signal_freq, relevance_freq, signal_timefreq, relevance_timefreq = calculate_dft_explanation(signal_time, relevance_time)

    def replace_positive(x, positive=True):
        mask = x > 0 if positive else x < 0
        x_mod = x.copy()
        x_mod[mask] = 0
        return x_mod

    nrows, ncols = 3, 2
    figsize = (2 * 3.29, 2 * 3.29 / ncols * nrows * 1.4 / 2)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    t = np.linspace(0, signal_length, signal_length)
    axs[0, 0].plot(t, signal_time[0])
    axs[0, 0].set_title("signal in time domain")
    axs[0, 0].set_xlabel("$t$")
    axs[0, 0].set_ylabel("$x_t$")

    axs[0, 1].fill_between(t, replace_positive(relevance_time[0], positive=False), color="red")
    axs[0, 1].fill_between(t, replace_positive(relevance_time[0]), color="blue")
    axs[0, 1].set_title("Relevance in time domain")
    axs[0, 1].set_xlabel("$t$")
    axs[0, 1].set_ylabel("$R_t$")

    axs[1, 0].imshow(np.abs(signal_timefreq[0, :, :k_max].T), aspect="auto", origin="lower")
    axs[1, 0].set_title("signal in time-freq. domain")
    axs[1, 0].set_xlabel("$t$")
    axs[1, 0].set_ylabel("$k$")

    v = float(np.max(np.abs(np.ravel(relevance_timefreq[0, :, :k_max]))))
    axs[1, 1].imshow(relevance_timefreq[0, :, :k_max].T, aspect="auto", origin="lower", cmap="seismic", clim=(-v, v))
    axs[1, 1].set_title("Relevance in time-freq. domain")
    axs[1, 1].set_xlabel("$t$")
    axs[1, 1].set_ylabel("$k$")

    k = np.linspace(0, k_max, k_max)
    axs[2, 0].plot(k, np.abs(signal_freq[0, :k_max]))
    axs[2, 0].set_title("signal in freq. domain")
    axs[2, 0].set_xlabel("$k$")
    axs[2, 0].set_ylabel("$a_k$")

    axs[2, 1].fill_between(replace_positive(relevance_freq[0, :k_max], positive=False), k, color="red")
    axs[2, 1].fill_between(replace_positive(relevance_freq[0, :k_max]), k, color="blue")
    axs[2, 1].set_title("Relevance in freq. domain")
    axs[2, 1].set_ylabel("$k$")
    axs[2, 1].set_xlabel("$R_k$")

    for k in freqs_gt:
        axs[1, 1].axhline(y=k, color='yellow', linewidth=6, alpha=0.4)
        axs[2, 1].axhline(y=k, color='yellow', linewidth=6, alpha=0.4)

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"dftxai_{attribution_method}.png")


if __name__ == "__main__":
    get_dft_example()
    run_tutorial(attribution_method="lrpepsstdx")
    run_tutorial(attribution_method="lrpepsstdxsign")

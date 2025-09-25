import os
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import colormaps

FOLDER = os.path.dirname(__file__)
FILES = [
    "ngd_toy_d4_r1.npy",
    "ngd_toy_d4_r2.npy",
    "ngd_toy_d4_r3.npy",  # TODO: unstable in the last iterations
    "ngd_toy_d4_r4.npy",
    "ngd_toy_d4.npy",
]
LABELS = ["$r=1$", "$r=2$", "$r=3$", "$r=4$", "Full"]
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
LINESTYLES = ["-", "--", "-.", ":"]
# LINESTYLES = ["-"]
MARKERS = ["o", "v", "*", "s", "d", "+", ">", "h"]
plt.rcParams["svg.fonttype"] = "none"
## Set up LaTeX fonts
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"],
        # "font.size": 14,
    }
)
plt.rcParams.update({"axes.prop_cycle": cycler(color=colormaps["Set1"].colors)})
plt.figure(figsize=(15, 8))


def ema_bias_corrected(y, beta=0.99):
    m = 0.0
    out = np.empty_like(y, dtype=float)
    pow_beta = 1.0
    for t, v in enumerate(y, start=1):
        m = beta * m + (1 - beta) * v
        pow_beta *= beta
        out[t - 1] = m / (1 - pow_beta)  # bias-corrected
    return out


for filename, label, color, linestyle, marker in zip(
    FILES, LABELS, COLORS, cycle(LINESTYLES), cycle(MARKERS)
):
    path = os.path.join(FOLDER, filename)
    data = np.load(path)
    # Smooth the data using Savitzky-Golay filter
    # y_smooth = savgol_filter(data, window_length=100, polyorder=1)
    y_smooth = ema_bias_corrected(data, beta=0.95)

    x = np.arange(len(data))
    # plt.semilogy(
    #     x,
    #     y_smooth,
    #     ls=linestyle,
    #     color=color,
    #     marker=marker,
    #     ms=7,
    #     markevery=1000,
    #     label=label,
    # )
    # plt.semilogy(x, data, color=color, alpha=0.3)
    plt.semilogy(
        x,
        data,
        color=color,
        ls=linestyle,
        marker=marker,
        ms=7,
        markevery=1000,
        label=label,
    )

plt.xlim(0, len(data))
plt.grid()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Relative L2 error")
plt.tight_layout(pad=1.10)
# plt.savefig("convergence_ranks.eps", format="eps", dpi=1200)
plt.savefig("convergence_ranks.pdf", format="pdf", dpi=1200)
plt.show()

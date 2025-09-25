import os

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import colormaps

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

problem = "sos"
dim = 4

directory = os.path.dirname(__file__)
with np.load(
    os.path.join(directory, "data", f"ngd_{problem}_d{dim}.npz"), allow_pickle=True
) as data:
    losses = data["losses"].item()
    loss_ngd, loss_gd, loss_bfgs = losses["ngd"], losses["gd"], losses["bfgs"]

    times = data["times"].item()
    times_ngd, times_gd, times_bfgs = times["ngd"], times["gd"], times["bfgs"]
    times_ngd = np.asarray(times_ngd)
    times_gd = np.asarray(times_gd)
    times_bfgs = np.asarray(times_bfgs)

# plot losses
plt.semilogy(
    np.arange(len(loss_gd)),
    loss_gd,
    label="Relative L2 (Adam)",
    ls="-",
    marker="o",
    ms=7,
    markevery=1000,
)
plt.semilogy(
    np.arange(len(loss_ngd)),
    loss_ngd,
    label="Relative L2 (Natural)",
    ls="--",
    marker="D",
    ms=7,
    markevery=1000,
)
plt.semilogy(
    np.arange(len(loss_bfgs)),
    loss_bfgs,
    label="Relative L2 (BFGS)",
    ls="-.",
    marker="*",
    ms=7,
    markevery=1000,
)
plt.grid()
plt.xlim(0, len(loss_gd))
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Relative L2 error")
plt.tight_layout(pad=1.10)
# plt.savefig("error.eps", format="eps", dpi=1200)
plt.savefig("error.pdf", format="pdf", dpi=1200)
plt.show()

# plot also times
plt.figure(figsize=(15, 8))
indices = times_gd < 400
plt.loglog(
    np.cumsum(times_gd[indices]),
    loss_gd[indices],
    label="Relative L2 (Adam)",
    ls="-",
    marker="o",
    ms=7,
    markevery=1000,
)
indices = times_ngd < 400
plt.semilogy(
    np.cumsum(times_ngd[indices]),
    loss_ngd[indices],
    label="Relative L2 (Natural)",
    ls="--",
    marker="D",
    ms=7,
    markevery=1000,
)
indices = times_bfgs < 400
plt.semilogy(
    np.cumsum(times_bfgs[indices]),
    loss_bfgs[indices],
    label="Relative L2 (BFGS)",
    ls="-.",
    marker="*",
    ms=7,
    markevery=1000,
)
plt.grid()
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("Relative L2 error")
plt.tight_layout(pad=1.10)
plt.savefig("time.pdf", format="pdf", dpi=1200)
plt.show()

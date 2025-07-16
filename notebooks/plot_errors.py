import os

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import colormaps

plt.rcParams["svg.fonttype"] = "none"
## Set up LaTeX fonts
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 14,
    }
)
model = "tt"

dir = os.path.dirname(__file__)
loss_gd = np.load(os.path.join(dir, f"{model}_adam.npy"))
loss_ngd = np.load(os.path.join(dir, f"./{model}_ngd.npy"))

plt.rcParams.update({"axes.prop_cycle": cycler(color=colormaps["Set1"].colors)})
plt.semilogy(np.arange(len(loss_gd)), loss_gd, label="Relative L2 (Adam)")
plt.semilogy(np.arange(len(loss_ngd)), loss_ngd, label="Relative L2 (Natural)")
plt.grid()
# plt.tight_layout()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Relative L2 error")
plt.savefig("error.eps", format="eps", dpi=1200)
plt.show()

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import colormaps

FOLDER = "notebooks"
FILES = [
    "ngd_truncate_bivariate_d4_r3.npy",
    "ngd_retract_bivariate_d4_r3.npy",
    "ngd_truncate_retract_bivariate_d4_r3.npy",
]
LABELS = ["Truncation", "Retraction", "Truncation+Retraction"]
STYLES = ["-", "-", "-"]
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

for filename, label, style in zip(FILES, LABELS, STYLES):
    path = os.path.join(FOLDER, filename)
    data = jnp.load(path)
    plt.semilogy(jnp.arange(len(data)), data, style, label=label)

plt.xlabel("Iterations")
plt.ylabel("Relative L2 error")
plt.grid()
plt.legend()
plt.savefig("perturbation.eps", format="eps", dpi=1200)
plt.show()

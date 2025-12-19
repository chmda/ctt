import os
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
directory = os.path.dirname(__file__)
timestr = time.strftime("%Y%m%d-%H%M%S")


# Matplotlib configuration
plt.rcParams["svg.fonttype"] = "none"
## Set up LaTeX fonts
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Verdana", "Arial", "Open Sans", "DejaVu Sans"],
        "font.size": 14,  # 12,
        "axes.labelsize": 14,  # axis labels
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


# List of (sketching_size, file_path) pairs
files = [
    (-1, "20250919-154906_sos_d4_ngd.npz"),
    # (20, "20251007-102946_sos_sketching_d4_ngd.npz"),
    # (25, "20251007-104136_sos_sketching_d4_ngd.npz"),
    # (30, "20251007-095618_sos_sketching_d4_ngd.npz"),
    # (35, "20251007-104624_sos_sketching_d4_ngd.npz"),
    # (20, "20251014-132511_sos_sketching_d4_ngd.npz"),
    # (25, "20251014-142710_sos_sketching_d4_ngd.npz"),
    # (30, "20251014-144617_sos_sketching_d4_ngd.npz"),
    # (35, "20251014-144714_sos_sketching_d4_ngd.npz"),
    (20, "20251127-100456_sos_sketching_d4_ngd.npz"),
    (30, "20251127-100636_sos_sketching_d4_ngd.npz"),
    (40, "20251127-100758_sos_sketching_d4_ngd.npz"),
]

# Different line styles, markers, and colors for distinction
line_styles = ["-", "--", "-.", ":"]
markers = ["o", "s", "^", "d", "x", "*"]
colors = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
]

# Load data and find maximum length
data = []
max_len = 0
for sketching_size, path in files:
    with jnp.load(os.path.join(directory, "data", path)) as npz:
        rel_l2 = jnp.array(npz["rel_l2"])
        data.append((sketching_size, rel_l2))
        max_len = max(max_len, len(rel_l2))

# Plot setup
# plt.figure(figsize=(15, 8))
plt.figure(figsize=(6, 4))
for i, (sketching_size, rel_l2) in enumerate(data):
    x = jnp.arange(1, len(rel_l2) + 1)
    style = line_styles[i % len(line_styles)]
    marker = markers[i % len(markers)]
    color = colors[i % len(colors)]

    num_markers = 10
    marker_positions = jnp.unique(
        jnp.round(jnp.logspace(0, jnp.log10(len(x) - 1), num=num_markers)).astype(int)
    )
    marker_positions = jnp.clip(marker_positions, 0, len(x) - 1)

    label = f"s={sketching_size}"
    if sketching_size == -1:
        label = "Original"

    plt.plot(
        x,
        rel_l2,
        linestyle=style,
        marker=marker,
        color=color,
        label=label,
        linewidth=1.5,
        markevery=marker_positions,
    )

# Formatting
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, max_len + 1)
plt.xlabel("Iteration")
plt.ylabel("Relative L2 error")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
# plt.grid()
plt.tight_layout()
plt.savefig(
    os.path.join(directory, "data", f"{timestr}_sketching.pdf"), format="pdf", dpi=1200
)
plt.show()

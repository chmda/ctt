import argparse
import os
import time

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import tomllib
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from benchmark_functions import benchmark_function
from ctt.bases import make_legendre_polynomials
from ctt.model import _eval_bases, eval_ftt
from ctt.solvers import als_ls
from ctt.tt import validate_ranks

# ------------------------------
# Global setup
# ------------------------------

Tensor = Float[Array, "..."]
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
        "font.size": 12,
    }
)

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


# ------------------------------
# Config + CLI
# ------------------------------

# Load experiment configs from TOML
with open("experiments.toml", "rb") as f:
    config = tomllib.load(f)

experiments = config["experiment"]
exp_names = [exp["name"] for exp in experiments]

# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Run experiment of learning a function by CTTs."
)
parser.add_argument("experiment", choices=exp_names, help="Name of the experiment")
parser.add_argument("rank", type=int, help="TT ranks")
args = parser.parse_args()
name = args.experiment
rank = args.rank

experiment = next(exp for exp in experiments if exp["name"] == name)


# ------------------------------
# Experiment setup
# ------------------------------
key = random.key(experiment["seed"])
d = experiment["dim"]
n_features = 3
basis = make_legendre_polynomials(dim=n_features)
# basis = make_canonical_polynomials(n_features)
bases = [basis] * d
function = name


filename = f"{timestr}_{function}_tt_d{d}_r{rank}"

# ------------------------------
# Target functions
# ------------------------------
target = benchmark_function(function, d)

ranks = [1] + [rank] * (d - 1) + [1]
ranks = validate_ranks([n_features] * d, ranks)
R = jnp.prod(jnp.asarray(ranks))
sigma = 1.0
c = jnp.sqrt((sigma**2 / R) ** (1 / d))

print("Ranks:", ranks)
print("Shape:", [(ranks[i], n_features, ranks[i + 1]) for i in range(d)])
print(
    "Size:",
    jnp.sum(jnp.asarray([ranks[i] * n_features * ranks[i + 1] for i in range(d)])),
)


def run_experiment(key: PRNGKeyArray):
    # ------------------------------
    # Data generation
    # ------------------------------
    key, train_key, val_key = random.split(key, 3)

    domain = experiment["domain"]

    # Training samples
    N_train = experiment["num_train"]
    X_train = random.uniform(
        key=train_key, shape=(N_train, d), minval=domain[0], maxval=domain[1]
    )
    y_train = jax.vmap(target)(X_train)
    y_train = jnp.ravel(y_train)
    features_train = jax.vmap(_eval_bases, in_axes=(None, 0))(bases, X_train)

    # Validation samples
    N_val = experiment["num_test"]
    X_val = random.uniform(
        key=val_key, shape=(N_val, d), minval=domain[0], maxval=domain[1]
    )
    y_val = jax.vmap(target)(X_val)
    y_val = jnp.ravel(y_val)

    y_val_norm = 0.5 * jnp.mean(y_val**2)

    # ------------------------------
    # Run optimizer
    # ------------------------------
    key, *tt_keys = random.split(key, 1 + d)
    init_tt = [
        c * random.normal(tt_keys[i], shape=(ranks[i], n_features, ranks[i + 1]))
        for i in range(d)
    ]  # we multiply by `c` so that the result TT has mean 0 and variance sigma^2

    iters, stag, sol_tt, info = als_ls(
        features_train,
        y_train,
        init_tt,
        max_iters=50,
        stagnation=1e-7,
    )
    # print(info)

    residuals = info["residual"]
    stagnations = info["stagnation"]

    # compute val loss
    hat_y_val = jax.vmap(eval_ftt, in_axes=(None, 0, None))(bases, X_val, sol_tt)
    hat_y_val = jnp.ravel(hat_y_val)
    loss_val = 0.5 * jnp.mean((hat_y_val - y_val) ** 2)
    rel_l2 = jnp.sqrt(loss_val / y_val_norm)

    return key, (residuals, stagnations, loss_val, rel_l2)


N_repeats = 25
residuals = []
stagnations = []
loss_val = []
rel_l2 = []

with tqdm(range(N_repeats)) as pbar:
    for i in pbar:
        key, (res, stag, loss, l2) = run_experiment(key)
        residuals.append(res)
        stagnations.append(stag)
        loss_val.append(loss)
        rel_l2.append(l2)
        pbar.set_postfix_str(
            f"residual={res[-1]:.3e}, stag={stag[-1]:.3e}, l2={l2:.3e}"
        )

loss_val = jnp.asarray(loss_val)
rel_l2 = jnp.asarray(rel_l2)
mean_rel_l2 = jnp.mean(jnp.asarray(rel_l2))
std_rel_l2 = jnp.std(jnp.asarray(rel_l2))
print("Relative L2 - Mean:", mean_rel_l2, ", Std:", std_rel_l2)

jnp.savez(
    os.path.join(directory, "data", f"{filename}_als.npz"),
    val_loss=loss_val,
    rel_l2=rel_l2,
)

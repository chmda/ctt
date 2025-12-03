"""
Experiment runner for learning functions with Compositional Tensor(-Trains) (CT(T)s) and random sketching.
"""

import argparse
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Union

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import optax
import tomllib
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

from benchmark_functions import benchmark_function
from ctt.bases import make_canonical_polynomials
from ctt.model import _eval_bases

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
# Dataclasses
# ------------------------------
@dataclass
class Optimizer:
    """Optimizer configuration for experiments."""

    name: Literal["ngd", "adam", "lbfgs"]
    num_iterations: int
    learning_rate: Union[float, Literal["line_search"]]
    damping_coeff: Optional[float] = None
    adaptive_damping_coeff: bool = False


@dataclass
class Experiment:
    """Experiment configuration loaded from TOML."""

    name: str
    seed: int
    dim: int
    n_features: int
    domain: tuple[float, float]
    num_train: int
    num_test: int
    num_layers: int
    optimizer: list[Optimizer]
    rank: Optional[int] = None
    init_constant: float = 2.0
    d_lift: Optional[float] = None
    sketching_rank: Optional[int] = 30

    def __post_init__(self):
        """Ensure optimizers are initialized as Optimizer objects."""
        for i, opt in enumerate(self.optimizer):
            if isinstance(opt, dict):
                self.optimizer[i] = Optimizer(**opt)
        if self.d_lift is None:
            self.d_lift = self.dim
        self.d_lift = max(self.d_lift or self.dim, self.dim)


# ------------------------------
# Config + CLI
# ------------------------------

# Load experiment configs from TOML
with open("experiments.toml", "rb") as f:
    config = tomllib.load(f)

experiments = [Experiment(**exp) for exp in config["experiment"]]
exp_names = [exp.name for exp in experiments]

# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Run experiment of learning a function by CTTs."
)
parser.add_argument("experiment", choices=exp_names, help="Name of the experiment")
args = parser.parse_args()
name = args.experiment

experiment = next(exp for exp in experiments if exp.name == name)


# ------------------------------
# Experiment setup
# ------------------------------
key = random.key(experiment.seed)
d = experiment.dim
n_features = experiment.n_features
basis = make_canonical_polynomials(dim=n_features)
function = experiment.name
sketching_rank = experiment.sketching_rank


filename = f"{timestr}_{function}_sketching_d{d}"
if experiment.rank is not None:
    filename += f"_r{experiment.rank}"


@contextmanager
def elapsed_timer():
    start = time.perf_counter_ns() // 1000
    try:
        yield lambda: time.perf_counter_ns() // 1000 - start
    finally:
        # just ensures that timing ends even if an exception happens
        end = time.perf_counter_ns() // 1000
        elapsed = end - start


@jax.jit
def compute_loss(
    tensors: list[Tensor], x: Float[Array, "B d"], y: Float[Array, "B d_o"]
) -> float:
    B = x.shape[0]
    y_pred = jax.vmap(_eval_ct, in_axes=(None, 0))(tensors, x)
    loss = jnp.sum((y_pred - y) ** 2) / B
    return 0.5 * loss


@jax.jit
def grad_loss(
    y_hat: Float[Array, "B d_o"], y: Float[Array, "B d_o"]
) -> Float[Array, "B d_o"]:
    # return y_hat - y
    return (y_hat - y) / y_hat.shape[0]


# ------------------------------
# Tensor evaluation functions
# ------------------------------
d_lift = experiment.d_lift


def lift(x: Float[Array, "d"]) -> Float[Array, "m"]:
    """Lifting function to map input x into higher-dimensional space."""
    if d_lift > d:
        return jnp.hstack((x, jnp.ones((d_lift - d,))))
    else:
        return x


def retraction(x: Float[Array, "m"]) -> Float[Array, "o"]:
    """Retract lifted tensor back to output dimension."""
    return jnp.asarray((x[0],))


def _eval_tensor(tensor: Tensor, x: Float[Array, "d"]) -> Float[Array, "d"]:
    """Evaluate a tensor at input x using basis functions."""
    features = _eval_bases(bases, x)
    y = tensor  # (d, m, ..., m)
    for i in range(len(features)):
        y = jnp.einsum("di...,i->d...", y, features[i])
    return y


def _eval_ct(tensors: list[Tensor], x: Float[Array, "d"]) -> Float[Array, "d_o"]:
    """Evaluate a Compositional Tensor."""

    def _body_fn(i: int, val: Float[Array, "d"]) -> Float[Array, "d"]:
        layer = jax.lax.switch(i, [lambda u=u: u for u in tensors])
        return val + _eval_tensor(layer, val)

    x = lift(x)
    val = jax.lax.fori_loop(0, len(tensors), _body_fn, x)
    val = retraction(val)
    return val


d_lift = jax.eval_shape(lift, jax.ShapeDtypeStruct((d,), dtype=float)).shape[0]
bases = [basis] * d_lift


# ------------------------------
# Precision matrix construction
# ------------------------------

# Generate a precision matrix with exponentially decaying singular values
key, sample_key = random.split(key)

alpha = 1
theta = 1
subrank = 2
M = random.uniform(key=sample_key, shape=(d, d), minval=-1.0, maxval=1.0)
M = M @ M.T
subblock = M[d - subrank :, :subrank]

# Replace singular values of the subblock
U, S, Vh = jnp.linalg.svd(subblock, full_matrices=False)
S = alpha * jnp.exp(-theta * jnp.arange(1, subrank + 1))
subblock = U @ jnp.diag(S) @ Vh

# Update block structure
M = M.at[d - subrank :, :subrank].set(subblock)
M = M.at[:subrank, d - subrank :].set(subblock.T)

# Shift eigenvalues to fix minimum eigenvalue
lmbdmin = 0.5
ev = jnp.linalg.eigvalsh(M)
M = M + (lmbdmin - ev[0]) * jnp.eye(d)
Sigma = jnp.linalg.inv(M)


# ------------------------------
# Target functions
# ------------------------------


# @jax.jit
# def target(x: Float[Array, "d"]) -> Float[Array, "m"]:
#     """Compute target function value at x based on experiment type."""
#     if function == "gaussian":
#         val = jax.scipy.stats.multivariate_normal.pdf(
#             x=x, mean=jnp.zeros_like(x), cov=Sigma
#         )
#     elif function == "henon-heiles":
#         val = (
#             0.5 * jnp.sum(x**2)
#             + 0.2 * jnp.sum(x[:-1] * x[1:] ** 2 - x[:-1] ** 3)
#             + 0.2**2 / 16 * jnp.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2)
#         )
#     elif function == "toy":
#         # val = jnp.log(1e-1 + 3*jnp.sum(x)**2)
#         val = jnp.exp(jnp.prod(x)) / jnp.exp(1)
#     elif function == "sos":
#         # assert d % 2 == 0, "'d' should be even"
#         # m = d // 2
#         # x1, x2 = x[:m], x[m:]
#         # diff = x1 - x2
#         # p = jnp.prod(diff)
#         # val = p**2
#         # z = 0 if d % 2 == 0 else x[-1]
#         # m = d // 2 if d % 2 == 0 else (d - 1) // 2
#         # diff = x[:m] - x[m : 2 * m] + z
#         # val = jnp.prod(diff) ** 2
#         z = 1 if d % 2 == 0 else x[-1]
#         m = d // 2
#         diff = x[:m] * z - x[m : 2 * m]
#         val = jnp.prod(diff) ** 2
#     else:
#         raise ValueError(f"Unknown function type: {function}")
#     return jnp.atleast_1d(val)
target = benchmark_function(function, d)

# ------------------------------
# Data generation
# ------------------------------
key, train_key, val_key = random.split(key, 3)

domain = experiment.domain

# Training samples
N_train = experiment.num_train
X_train = random.uniform(
    key=train_key, shape=(N_train, d), minval=domain[0], maxval=domain[1]
)
y_train = jax.vmap(target)(X_train)

# Validation samples
N_val = experiment.num_test
X_val = random.uniform(
    key=val_key, shape=(N_val, d), minval=domain[0], maxval=domain[1]
)
y_val = jax.vmap(target)(X_val)

y_val_norm = 0.5 * jnp.sum(y_val**2) / N_val


# ------------------------------
# Architecture setup
# ------------------------------
rank = experiment.rank
num_layers = experiment.num_layers


# ------------------------------
# Learning algorithms
# ------------------------------


@jax.jit
def khatri_rao(a: Float[Array, "n k"], b: Float[Array, "m k"]) -> Float[Array, "n*m k"]:
    def _f(x: Float[Array, "n"], y: Float[Array, "m"]) -> Float[Array, "n*m"]:
        return jnp.outer(x, y).reshape((-1,))

    return jax.vmap(_f, in_axes=1, out_axes=1)(a, b)


def natural_gradient_descent(
    optimizer: optax.GradientTransformation,
    damping_coeff: float,
    sketching_rank: int,
    backtracking: bool = False,
):
    def _natural_grad(
        key: PRNGKeyArray,
        tensors: list[Tensor],
        x: Float[Array, "B d"],
        y: Float[Array, "B d_o"],
    ) -> tuple[PRNGKeyArray, tuple[list[Tensor], dict]]:
        # solving the least-squares problem ||du/dtheta_k @ dir - grad L||

        def _value_jac(params: list[Tensor], x: Float[Array, "B d"]):
            def _f(x: Float[Array, "d"]):
                ans, vjp_py = jax.vjp(_eval_ct, params, x)
                g = vjp_py(jnp.ones_like(ans))[0]
                return ans, g

            return jax.vmap(_f)(x)

        y_pred, J = _value_jac(
            tensors, x
        )  # y_pred: (B, d_o), J: (B, d_o, *params)*n_layers
        grad_losses = grad_loss(y_pred, y)  # (B, d_o)

        def _randomized_nystrom(
            key: PRNGKeyArray, jac: Float[Array, "B d_o n_params"]
        ) -> tuple[Float[Array, "n_params k"], Float[Array, "k"]]:
            B = jac.shape[0]

            # Tensor-structured random embedding
            # Omega = omega_1 o ... o omega_L
            # where omega_j ~ N(0, 1/k I) and o is the Khatri-Rao product
            # omega = random.normal(key, (jac.shape[1], sketching_rank))  # (n_params, k)
            # omega /= jnp.sqrt(sketching_rank)
            keys = random.split(key, jac.ndim - 1)
            omegas = [
                random.normal(subkey, shape=(jac.shape[i + 1], sketching_rank))
                # / jnp.sqrt(sketching_rank)
                for i, subkey in enumerate(keys)
            ]
            omega = jax.tree.reduce(khatri_rao, omegas) / jnp.sqrt(sketching_rank)
            omega, _ = jnp.linalg.qr(omega, mode="reduced")

            jac = jnp.reshape(
                jac, (jac.shape[0] * grad_losses.shape[1], -1)
            )  # (B*d_o, n_params)
            A = (
                jnp.linalg.multi_dot((jac.T, jac, omega)) / B + damping_coeff * omega
            )  # (n_params, k)
            # ???: orthogonalize A?

            nu = jnp.finfo(jac.dtype).eps * jnp.linalg.norm(A, ord="fro")
            Anu = A + nu * omega
            U = jnp.linalg.cholesky(omega.T @ Anu, upper=True)
            B = jax.lax.linalg.triangular_solve(U, Anu, left_side=False, lower=False)
            U, s, _ = jnp.linalg.svd(B, full_matrices=False)
            eigenvalues = jnp.maximum(0.0, s**2 - nu)
            return U, eigenvalues

        def _block_projection(
            key: PRNGKeyArray, jac: Float[Array, "B d_o n_params"]
        ) -> Tensor:
            shape = jac.shape

            U, S = _randomized_nystrom(key, jac)

            jac = jnp.reshape(
                jac, (jac.shape[0] * grad_losses.shape[1], -1)
            )  # (B*d_o, n_params)

            b = jac.T @ grad_losses.ravel()  # (n_params,)

            # ngrad = (
            #     jnp.linalg.multi_dot(
            #         (U * (1.0 / (S + damping_coeff) - 1.0 / (damping_coeff)), U.T, b)
            #     )
            #     + b / damping_coeff
            # )
            eps = jnp.finfo(jac.dtype).eps * S[0]
            inv_S = jnp.where(S > eps, 1.0 / S, 0.0)
            ngrad = jnp.linalg.multi_dot((U * inv_S, U.T, b))
            ngrad = jnp.reshape(ngrad, shape[1:])

            return ngrad

        key, *sketching_keys = random.split(key, len(tensors) + 1)

        ngrads = [
            _block_projection(sketching_keys[k], J[k]) for k in range(len(tensors))
        ]
        ngrads = list(ngrads)
        ngrad_norms = list(map(jnp.linalg.norm, ngrads))
        info = {
            "gradient_norm": jnp.sqrt(jnp.sum(jnp.asarray(ngrad_norms) ** 2)),
            # TODO: add condition number
        }
        return key, (ngrads, info)

    @jax.jit
    def step(
        key: PRNGKeyArray,
        tensors: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
    ):
        loss = compute_loss(tensors, x, y)
        key, (ngrad, info) = _natural_grad(key, tensors, x, y)
        if backtracking:
            updates, opt_state = optimizer.update(
                ngrad,
                opt_state,
                tensors,
                value=loss,
                grad=ngrad,
                value_fn=lambda p: compute_loss(p, x, y),
            )
        else:
            updates, opt_state = optimizer.update(ngrad, opt_state, tensors)
        params = optax.apply_updates(tensors, updates)

        return key, (params, opt_state, loss, info)

    def do_iter(
        key: PRNGKeyArray,
        params: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
        x_val: Float[Array, "B d"],
        y_val: Float[Array, "B d"],
    ):
        with elapsed_timer() as t:
            key, (params, opt_state, l2_error, info) = step(
                key, params, opt_state, x, y
            )
        elapsed = t()
        info["time"] = elapsed

        val_loss = compute_loss(params, x_val, y_val)

        if backtracking:
            for state in opt_state:
                if lr := getattr(state, "learning_rate", None) is not None:
                    info["learning_rate"] = lr
                    break

        return key, (params, opt_state, (l2_error, val_loss), info)

    return do_iter


# ------------------------------
# Run optimizers
# ------------------------------
key, *control_keys = random.split(key, num_layers + 1)

if experiment.rank is None:
    init_controls = [
        math.sqrt(experiment.init_constant / (num_layers * n_features**d_lift))
        * random.normal(key=control_key, shape=(d_lift,) + (n_features,) * d_lift)
        for control_key in control_keys
    ]
else:
    raise NotImplementedError("Low rank solver not implemented")

# NGD
opt_ngd = next(opt for opt in experiment.optimizer if opt.name == "ngd")
use_linesearch = (
    isinstance(opt_ngd.learning_rate, str) and opt_ngd.learning_rate == "line_search"
)

params = init_controls
if experiment.rank is None:
    if isinstance(opt_ngd.learning_rate, str):
        if opt_ngd.learning_rate != "line_search":
            raise ValueError(f"Learning rate '{opt_ngd.learning_rate}' unknown.")
        optimizer = optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_zoom_linesearch(
                max_linesearch_steps=20,
                max_learning_rate=1.0,
                initial_guess_strategy="one",
            ),
        )
    else:
        optimizer = optax.sgd(learning_rate=opt_ngd.learning_rate)

    state = optimizer.init(params)
    ngd_step = natural_gradient_descent(
        optimizer=optimizer,
        damping_coeff=opt_ngd.damping_coeff,
        sketching_rank=sketching_rank,
        backtracking=use_linesearch,
    )
else:
    raise NotImplementedError("Low rank solver not implemented")

train_losses_ngd = []
val_losses_ngd = []
rel_l2_ngd = []
infos_ngd = []
key, training_key = random.split(key)


print("------------------------------")
print("Running natural gradient descent...")

with tqdm(
    range(opt_ngd.num_iterations), desc="NGD", leave=True, total=opt_ngd.num_iterations
) as pbar:
    for _ in pbar:
        training_key, (params, state, (train_loss, val_loss), info) = ngd_step(
            training_key, params, state, X_train, y_train, X_val, y_val
        )

        train_losses_ngd.append(train_loss)
        val_losses_ngd.append(val_loss)
        infos_ngd.append(info)
        rl2_error = jnp.sqrt(val_loss / y_val_norm)
        rel_l2_ngd.append(rl2_error)

        postfix = dict(
            train_loss=f"{train_loss:.3e}",
            val_loss=f"{val_loss:.3e}",
            rel_l2=f"{rl2_error:.3e}",
            gradient_norm=f"{info['gradient_norm']:.3e}",
        )
        pbar.set_postfix(**postfix)

print("------------------------------")
jnp.savez(
    os.path.join(directory, "data", f"{filename}_ngd.npz"),
    val_loss=val_losses_ngd,
    rel_l2=rel_l2_ngd,
    info=infos_ngd,
)

# plot the results

# loss
plt.figure(figsize=(15, 8))

num_markers = 10
marker_positions = jnp.unique(
    jnp.round(
        jnp.logspace(0, jnp.log10(opt_ngd.num_iterations - 1), num=num_markers)
    ).astype(int)
)
marker_positions = jnp.clip(marker_positions, 0, opt_ngd.num_iterations - 1)
plt.plot(
    jnp.arange(1, opt_ngd.num_iterations + 1),
    rel_l2_ngd,
    label="NGD",
    ls=line_styles[0],
    marker=markers[0],
    color=colors[0],
    markevery=marker_positions,
)
# Formatting
plt.xscale("log")
plt.yscale("log")
plt.xlim(1, opt_ngd.num_iterations + 1)
plt.xlabel("Iteration")
plt.ylabel("Relative L2 error")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
# plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(directory, "data", f"{filename}_error.pdf"), format="pdf", dpi=1200
)
plt.show()

# timing
plt.figure(figsize=(15, 8))

times_ngd = jnp.asarray([info["time"] for info in infos_ngd])
indices_ngd = (
    times_ngd < 1000
)  # remove outliers, shouldn't take more than 1000 ms per iteration, except JIT
x_ngd = jnp.cumsum(times_ngd[indices_ngd])

num_markers = 10
marker_positions = jnp.unique(
    jnp.round(jnp.logspace(0, jnp.log10(len(x_ngd) - 1), num=num_markers)).astype(int)
)
marker_positions = jnp.clip(marker_positions, 0, len(x_ngd) - 1)
plt.plot(
    x_ngd,
    jnp.asarray(rel_l2_ngd)[indices_ngd],
    label="NGD",
    ls=line_styles[0],
    marker=markers[0],
    color=colors[0],
    markevery=marker_positions,
)
# Formatting
plt.xscale("log")
plt.yscale("log")

plt.xlim(
    jnp.min(x_ngd),
    jnp.max(x_ngd),
)
plt.xlabel("Time (ms)")
plt.ylabel("Relative L2 error")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.7)
# plt.grid()
plt.tight_layout()

plt.savefig(
    os.path.join(directory, "data", f"{filename}_time.pdf"), format="pdf", dpi=1200
)
plt.show()

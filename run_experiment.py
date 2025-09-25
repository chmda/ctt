"""
Experiment runner for learning functions with Compositional Tensor(-Trains) (CT(T)s).
"""

import argparse
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as random
import jaxopt
import matplotlib.pyplot as plt
import optax
import tomllib
from jaxtyping import Array, Float
from tqdm import tqdm

from ctt.bases import make_canonical_polynomials
from ctt.model import _eval_bases
from ctt.tt import full_to_tt_truncate, tt_to_full, validate_ranks

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
        "font.family": "serif",
        # "font.serif": ["Computer Modern Roman"],
        "font.size": 11,
    }
)


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
key = random.PRNGKey(experiment.seed)
d = experiment.dim
n_features = experiment.n_features
basis = make_canonical_polynomials(dim=n_features)
function = experiment.name


filename = f"{timestr}_{function}_d{d}"
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
        control = jax.lax.switch(i, [lambda u=u: u for u in tensors])
        return val + _eval_tensor(control, val)

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


@jax.jit
def target(x: Float[Array, "d"]) -> Float[Array, "m"]:
    """Compute target function value at x based on experiment type."""
    if function == "gaussian":
        val = jax.scipy.stats.multivariate_normal.pdf(
            x=x, mean=jnp.zeros_like(x), cov=Sigma
        )
    elif function == "henon-heiles":
        val = (
            0.5 * jnp.sum(x**2)
            + 0.2 * jnp.sum(x[:-1] * x[1:] ** 2 - x[:-1] ** 3)
            + 0.2**2 / 16 * jnp.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2)
        )
    elif function == "toy":
        # val = jnp.log(1e-1 + 3*jnp.sum(x)**2)
        val = jnp.exp(jnp.prod(x)) / jnp.exp(1)
    elif function == "sos":
        # assert d % 2 == 0, "'d' should be even"
        # m = d // 2
        # x1, x2 = x[:m], x[m:]
        # diff = x1 - x2
        # p = jnp.prod(diff)
        # val = p**2
        # z = 0 if d % 2 == 0 else x[-1]
        # m = d // 2 if d % 2 == 0 else (d - 1) // 2
        # diff = x[:m] - x[m : 2 * m] + z
        # val = jnp.prod(diff) ** 2
        z = 1 if d % 2 == 0 else x[-1]
        m = d // 2
        diff = x[:m] * z - x[m : 2 * m]
        val = jnp.prod(diff) ** 2
    else:
        raise ValueError(f"Unknown function type: {function}")
    return jnp.atleast_1d(val)


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


# def dataloader(arrays: list, batch_size: int, *, key: PRNGKeyArray):
#     """Yield batches of data from arrays indefinitely with shuffling."""
#     dataset_size = arrays[0].shape[0]
#     assert all(array.shape[0] == dataset_size for array in arrays)
#     indices = jnp.arange(dataset_size)
#     while True:
#         perm = random.permutation(key, indices)
#         (key,) = random.split(key, 1)
#         start = 0
#         end = batch_size
#         while end <= dataset_size:
#             batch_perm = perm[start:end]
#             yield tuple(array[batch_perm] for array in arrays)
#             start = end
#             end = start + batch_size


# ------------------------------
# Architecture setup
# ------------------------------
rank = experiment.rank
num_layers = experiment.num_layers


# ------------------------------
# Learning algorithms
# ------------------------------


def natural_gradient_descent(
    optimizer: optax.GradientTransformation,
    backtracking: bool = False,
    damping_coeff: float = None,
    adaptive_damping_coeff: bool = False,
    rank: Optional[int] = 3,
):
    def truncate(params, rank):
        new_params = []
        errors = []
        for i in range(len(params)):
            tensor = params[i]
            ranks = [1] + [rank] * (d_lift) + [1]
            ranks = validate_ranks(tensor.shape, ranks)
            truncated, error = full_to_tt_truncate(tensor, ranks)
            error /= jnp.linalg.norm(tensor)
            errors.append(error)
            new_params.append(tt_to_full(truncated))
        return new_params, errors

    def _natural_grad(
        tensors: list[Tensor], x: Float[Array, "B d"], y: Float[Array, "B d_o"]
    ) -> tuple[list[Tensor], dict]:
        # solving the least-squares problem ||du/dtheta_k @ dir - grad L||
        # y_pred = jax.vmap(_eval_ct, in_axes=(None, 0))(tensors, x)
        # grad_losses = grad_loss(y_pred, y)
        # J = jax.vmap(jax.jacfwd(_eval_ct), in_axes=(None, 0))(tensors, x)

        # we can speedup the computation of `y_pred` and `J` using vjp:
        """
        def _f(x):
            ans, vjp_py = jax.vjp(_eval_ct, params, x)
            g = vjp_py(jnp.ones_like(ans))[0]
            return ans, g
        return jax.vmap(_f)(X_train)
        """

        def _value_jac(params: list[Tensor], x: Float[Array, "B d"]):
            def _f(x: Float[Array, "d"]):
                ans, vjp_py = jax.vjp(_eval_ct, params, x)
                g = vjp_py(jnp.ones_like(ans))[0]
                return ans, g

            return jax.vmap(_f)(x)

        y_pred, J = _value_jac(tensors, x)
        grad_losses = grad_loss(y_pred, y)

        # @line_profiler.profile
        def _block_projection(jac: Tensor) -> tuple[Tensor, Float[Array, "m"]]:
            shape = jac.shape
            jac = jnp.reshape(
                jac, (jac.shape[0] * grad_losses.shape[1], -1)
            )  # (B*d_o, n_params)
            trace = jnp.sum(jac**2) / x.shape[0]

            # no Tikhonov regularization so we solve the problem using least-squares
            if damping_coeff is None:
                ngrad, resid, _, s = jnp.linalg.lstsq(jac, grad_losses.ravel())
                return jnp.reshape(ngrad, shape[2:]), s, resid, trace

            # computing the Gramian
            G = (
                jac.T @ jac / x.shape[0]
            )  # WARN: don't forget to divide by the number of samples!
            b = jac.T @ grad_losses.ravel()
            D = jnp.eye(G.shape[0])
            s = None

            if adaptive_damping_coeff:
                # based on [Dahmen et al, 25], we choose the damping parameter as a piecewise function
                # depending on the maximum diagonal entry of G.
                max_g = jnp.max(jnp.diag(G))
                n_knots = 6
                knots = 10 ** jnp.arange(n_knots)
                conds = (
                    [max_g < knots[0]]
                    + [
                        (knots[j - 1] <= max_g) & (max_g < knots[j])
                        for j in range(1, n_knots)
                    ]
                    + [max_g >= knots[-1]]
                )
                coeffs = damping_coeff * 10 ** jnp.arange(n_knots + 1)
                lmbd = jnp.sum(jnp.asarray(conds) * coeffs)
            else:
                lmbd = damping_coeff

            ngrad = jnp.linalg.solve(G + lmbd * D, b)
            # ngrad = jax.scipy.sparse.linalg.bicgstab(lambda x: (G+lmbd*D)@x, b, tol=1e-12)[0]
            ngrad = jnp.reshape(ngrad, shape[1:])
            # lmbd = damping_coeff
            # linear_op = lambda x: (jnp.linalg.multi_dot((jac.T, jac, x)) / x.shape[0] + lmbd * x)
            # rhs = jac.T @ grad_losses.ravel()
            # ngrad = jax.scipy.sparse.linalg.bicgstab(linear_op, rhs, tol=1e-12)[0]
            # resid = jnp.linalg.norm(jac @ ngrad - grad_losses.ravel())**2
            # ngrad = jnp.reshape(ngrad, shape[2:])

            error = None
            if rank:
                # ranks = [1] + [rank]*d_lift + [1]
                ranks = [1] + [d_lift] + [rank] * (d_lift - 1) + [1]
                ranks = validate_ranks(shape[2:], ranks)
                truncated_ngrad, _ = full_to_tt_truncate(ngrad, ranks)
                approx_ngrad = tt_to_full(truncated_ngrad)
                approx_resid = (
                    jnp.linalg.norm(jac @ approx_ngrad.ravel() - grad_losses.ravel())
                    ** 2
                )
                error = approx_resid
                ngrad = approx_ngrad
                # error = 0.
            return ngrad, s, error, trace

        ngrads, s, errors, traces = zip(
            *[_block_projection(J[k]) for k in range(len(tensors))]
        )
        ngrads = list(ngrads)
        info = {"singular_values": s, "trace": traces, "errors": errors}
        return (ngrads, info)

    @jax.jit
    def step(
        tensors: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
    ):
        loss = compute_loss(tensors, x, y)
        ngrad, info = _natural_grad(tensors, x, y)
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

        if rank:
            params, errors = truncate(params, rank=rank)
            info["truncation_errors"] = errors

        return params, opt_state, loss, info

    def do_iter(
        params: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
        x_val: Float[Array, "B d"],
        y_val: Float[Array, "B d"],
    ):
        with elapsed_timer() as t:
            params, opt_state, l2_error, info = step(params, opt_state, x, y)
        elapsed = t()
        info["time"] = elapsed

        val_loss = compute_loss(params, x_val, y_val)

        if backtracking:
            for state in opt_state:
                if lr := getattr(state, "learning_rate", None) is not None:
                    info["learning_rate"] = lr
                    break

        return params, opt_state, (l2_error, val_loss), info

    return do_iter


def gradient_descent(
    optimizer: optax.GradientTransformation,
    backtracking: bool = False,
):
    @jax.jit
    def step(
        tensors: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
    ):
        loss, grads = jax.value_and_grad(compute_loss)(tensors, x, y)
        if backtracking:
            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                tensors,
                value=loss,
                grad=grads,
                value_fn=lambda p: compute_loss(p, x, y),
            )
        else:
            updates, opt_state = optimizer.update(grads, opt_state, tensors)
        params = optax.apply_updates(tensors, updates)
        return params, opt_state, loss

    def do_iter(
        params: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
        x_val: Float[Array, "B d"],
        y_val: Float[Array, "B d"],
    ):
        with elapsed_timer() as t:
            params, opt_state, l2_error = step(params, opt_state, x, y)
        elapsed = t()

        info = {}
        info["time"] = elapsed

        val_loss = compute_loss(params, x_val, y_val)

        if backtracking:
            for state in opt_state:
                if lr := getattr(state, "learning_rate", None) is not None:
                    info["learning_rate"] = lr
                    break

        return params, opt_state, (l2_error, val_loss), info

    return do_iter


def bfgs(solver: jaxopt.BFGS):
    @jax.jit
    def step(
        tensors: list[Tensor], opt_state, x: Float[Array, "B d"], y: Float[Array, "B d"]
    ):
        return solver.update(tensors, opt_state, x=x, y=y)

    def do_iter(
        params: list[Tensor],
        opt_state,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
        x_val: Float[Array, "B d"],
        y_val: Float[Array, "B d"],
    ):
        with elapsed_timer() as t:
            params, opt_state = step(params, opt_state, x, y)
        elapsed = t()
        l2_error = opt_state.value

        info = {}
        info["time"] = elapsed

        val_loss = compute_loss(params, x_val, y_val)

        if solver.stepsize > 0.0:
            info["learning_rate"] = opt_state.stepsize

        return params, opt_state, (l2_error, val_loss), info

    return do_iter


# ------------------------------
# Run optimizers
# ------------------------------
key, *control_keys = random.split(key, num_layers + 1)
key, training_key = random.split(key)

init_controls = [
    math.sqrt(experiment.init_constant / (num_layers * n_features**d_lift))
    * random.normal(key=control_key, shape=(d_lift,) + (n_features,) * d_lift)
    for control_key in control_keys
]
if experiment.rank is not None:
    ranks = [1] + [d_lift] + [experiment.rank] * (d_lift - 1) + [1]
    init_controls = list(map(lambda x: full_to_tt_truncate(x, ranks)[0], init_controls))
    init_controls = list(map(tt_to_full, init_controls))

y_val_norm = 0.5 * jnp.mean(y_val**2)
print("Value range: [", jnp.min(y_val), ",", jnp.max(y_val), "]")

# NGD
opt_ngd = next(opt for opt in experiment.optimizer if opt.name == "ngd")
use_linesearch = (
    isinstance(opt_ngd.learning_rate, str) and opt_ngd.learning_rate == "line_search"
)

if isinstance(opt_ngd.learning_rate, str):
    if opt_ngd.learning_rate != "line_search":
        raise ValueError(f"Learning rate '{opt_ngd.learning_rate}' unknown.")
    optimizer = optax.chain(
        optax.sgd(learning_rate=1.0),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=20, max_learning_rate=1.0, initial_guess_strategy="one"
        ),
    )
else:
    optimizer = optax.sgd(learning_rate=opt_ngd.learning_rate)

ngd_step = natural_gradient_descent(
    optimizer=optimizer,
    backtracking=use_linesearch,
    damping_coeff=opt_ngd.damping_coeff,
    adaptive_damping_coeff=opt_ngd.adaptive_damping_coeff,
    rank=experiment.rank,
)

train_losses_ngd = []
val_losses_ngd = []
rel_l2_ngd = []
infos_ngd = []

params = init_controls
state = optimizer.init(params)

print("------------------------------")
print("Running natural gradient descent...")

with tqdm(
    range(opt_ngd.num_iterations), desc="NGD", leave=True, total=opt_ngd.num_iterations
) as pbar:
    for _ in pbar:
        params, state, (train_loss, val_loss), info = ngd_step(
            params, state, X_train, y_train, X_val, y_val
        )

        train_losses_ngd.append(train_loss)
        val_losses_ngd.append(val_loss)
        infos_ngd.append(info)
        rl2_error = jnp.sqrt(val_loss / y_val_norm)
        rel_l2_ngd.append(rl2_error)
        pbar.set_postfix(
            train_loss=f"{train_loss:.3e}",
            val_loss=f"{val_loss:.3e}",
            rel_l2=f"{rl2_error:.3e}",
        )

print("------------------------------")
jnp.savez(
    os.path.join(directory, "data", f"{filename}_ngd.npz"),
    val_loss=val_losses_ngd,
    rel_l2=rel_l2_ngd,
    info=infos_ngd,
)

# Adam
opt_adam = next(opt for opt in experiment.optimizer if opt.name == "adam")
use_linesearch = (
    isinstance(opt_adam.learning_rate, str) and opt_adam.learning_rate == "line_search"
)

if isinstance(opt_adam.learning_rate, str):
    if opt_adam.learning_rate != "line_search":
        raise ValueError(f"Learning rate '{opt_adam.learning_rate}' unknown.")
    optimizer = optax.chain(
        optax.adam(learning_rate=1.0),
        optax.scale_by_zoom_linesearch(
            max_linesearch_steps=55, max_learning_rate=1.0, initial_guess_strategy="one"
        ),
    )
else:
    optimizer = optax.adam(learning_rate=opt_adam.learning_rate)

adam_step = gradient_descent(optimizer=optimizer, backtracking=use_linesearch)

train_losses_adam = []
val_losses_adam = []
rel_l2_adam = []
infos_adam = []

params = init_controls
state = optimizer.init(params)

print("------------------------------")
print("Running Adam...")

with tqdm(
    range(opt_adam.num_iterations),
    desc="Adam",
    leave=True,
    total=opt_adam.num_iterations,
) as pbar:
    for _ in pbar:
        params, state, (train_loss, val_loss), info = adam_step(
            params, state, X_train, y_train, X_val, y_val
        )

        train_losses_adam.append(train_loss)
        val_losses_adam.append(val_loss)
        infos_adam.append(info)
        rl2_error = jnp.sqrt(val_loss / y_val_norm)
        rel_l2_adam.append(rl2_error)
        pbar.set_postfix(
            train_loss=f"{train_loss:.3e}",
            val_loss=f"{val_loss:.3e}",
            rel_l2=f"{rl2_error:.3e}",
        )

print("------------------------------")
jnp.savez(
    os.path.join(directory, "data", f"{filename}_adam.npz"),
    val_loss=val_losses_adam,
    rel_l2=rel_l2_adam,
    info=infos_adam,
)

# LBFGS
opt_lbfgs = next(opt for opt in experiment.optimizer if opt.name == "lbfgs")
use_linesearch = (
    isinstance(opt_lbfgs.learning_rate, str)
    and opt_lbfgs.learning_rate == "line_search"
)

if isinstance(opt_lbfgs.learning_rate, str):
    if opt_lbfgs.learning_rate != "line_search":
        raise ValueError(f"Learning rate '{opt_lbfgs.learning_rate}' unknown.")
    # optimizer = jaxopt.BFGS(fun=compute_loss)
    optimizer = optax.lbfgs(
        1.0,
        memory_size=25,
        scale_init_precond=True,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=55, initial_guess_strategy="one"
        ),
    )

else:
    # optimizer = jaxopt.BFGS(fun=compute_loss, stepsize=opt_bfgs.learning_rate)
    optimizer = optax.lbfgs(
        opt_lbfgs.learning_rate, memory_size=10, scale_init_precond=True
    )


# bfgs_step = bfgs(solver=optimizer)
lbfgs_step = gradient_descent(optimizer, backtracking=use_linesearch)

train_losses_lbfgs = []
val_losses_lbfgs = []
rel_l2_lbfgs = []
infos_lbfgs = []

params = init_controls
# state = optimizer.init_state(params, x=X_train, y=y_train)
state = optimizer.init(params)

print("------------------------------")
print("Running LBFGS...")

with tqdm(
    range(opt_lbfgs.num_iterations),
    desc="LBFGS",
    leave=True,
    total=opt_lbfgs.num_iterations,
) as pbar:
    for _ in pbar:
        params, state, (train_loss, val_loss), info = lbfgs_step(
            params, state, X_train, y_train, X_val, y_val
        )

        train_losses_lbfgs.append(train_loss)
        val_losses_lbfgs.append(val_loss)
        infos_lbfgs.append(info)
        rl2_error = jnp.sqrt(val_loss / y_val_norm)
        rel_l2_lbfgs.append(rl2_error)
        pbar.set_postfix(
            train_loss=f"{train_loss:.3e}",
            val_loss=f"{val_loss:.3e}",
            rel_l2=f"{rl2_error:.3e}",
        )

print("------------------------------")
jnp.savez(
    os.path.join(directory, "data", f"{filename}_lbfgs.npz"),
    val_loss=val_losses_lbfgs,
    rel_l2=rel_l2_lbfgs,
    info=infos_lbfgs,
)

# plot the results

# loss
plt.figure(figsize=(15, 8))

plt.loglog(
    jnp.arange(1, opt_ngd.num_iterations + 1),
    rel_l2_ngd,
    label="NGD",
    ls="-",
    marker="*",
    color="#bb5566",
    markevery=opt_ngd.num_iterations // 25,
)
plt.loglog(
    jnp.arange(1, opt_adam.num_iterations + 1),
    rel_l2_adam,
    label="Adam",
    ls="--",
    marker="s",
    color="#004488",
    markevery=opt_adam.num_iterations // 25,
)
plt.loglog(
    jnp.arange(1, opt_lbfgs.num_iterations + 1),
    rel_l2_lbfgs,
    label="LBFGS",
    ls=":",
    marker="^",
    color="#ddaa33",
    markevery=opt_lbfgs.num_iterations // 25,
)
plt.grid()
plt.legend()
plt.xlim(
    1,
    max(opt_ngd.num_iterations, opt_adam.num_iterations, opt_lbfgs.num_iterations) + 1,
)
plt.xlabel("Iterations")
plt.ylabel("Relative L2 error")
plt.tight_layout(pad=1.10)
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

times_adam = jnp.asarray([info["time"] for info in infos_adam])
indices_adam = times_adam < 1000
x_adam = jnp.cumsum(times_adam[indices_adam])

times_bfgs = jnp.asarray([info["time"] for info in infos_lbfgs])
indices_bfgs = times_bfgs < 1000
x_bfgs = jnp.cumsum(times_bfgs[indices_bfgs])

plt.loglog(
    x_ngd,
    jnp.asarray(rel_l2_ngd)[indices_ngd],
    label="NGD",
    ls="-",
    marker="*",
    color="#bb5566",
    markevery=opt_ngd.num_iterations // 25,
)
plt.loglog(
    x_adam,
    jnp.asarray(rel_l2_adam)[indices_adam],
    label="Adam",
    ls="--",
    marker="s",
    color="#004488",
    markevery=opt_adam.num_iterations // 25,
)
plt.loglog(
    x_bfgs,
    jnp.asarray(rel_l2_lbfgs)[indices_bfgs],
    label="LBFGS",
    ls=":",
    marker="^",
    color="#ddaa33",
    markevery=opt_lbfgs.num_iterations // 25,
)
plt.grid()
plt.legend()
plt.xlim(
    min(jnp.min(x_ngd), jnp.min(x_adam), jnp.min(x_bfgs)),
    max(jnp.max(x_ngd), jnp.max(x_adam), jnp.max(x_bfgs)),
)
plt.xlabel("Time (ms)")
plt.ylabel("Relative L2 error")
plt.tight_layout(pad=1.10)
plt.savefig(
    os.path.join(directory, "data", f"{filename}_time.pdf"), format="pdf", dpi=1200
)
plt.show()

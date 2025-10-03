"""
Experiment runner for learning functions with Compositional Tensor(-Trains) (CT(T)s).
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Literal, Union

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from tqdm import tqdm

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

    num_iterations: int
    learning_rate: Union[float, Literal["line_search"]]
    damping_coeff: float


@dataclass
class Experiment:
    """Experiment configuration loaded from TOML."""

    dim: int
    n_features: int
    domain: tuple[float, float]
    num_train: int
    num_test: int
    num_layers: int
    optimizer: Optimizer
    init_constant: float = 2.0


experiment = Experiment(
    dim=4,
    n_features=2,
    domain=(0.0, 1.0),
    num_train=2048,
    num_test=512,
    num_layers=2,
    optimizer=Optimizer(num_iterations=5_000, learning_rate=0.7, damping_coeff=1e-11),
)


# ------------------------------
# Experiment setup
# ------------------------------
d = experiment.dim
n_features = experiment.n_features
basis = make_canonical_polynomials(dim=n_features)

filename = f"{timestr}_conditioning_d{d}"


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
    return (y_hat - y) / y_hat.shape[0]


# ------------------------------
# Tensor evaluation functions
# ------------------------------


def lift(x: Float[Array, "d"]) -> Float[Array, "m"]:
    """Lifting function to map input x into higher-dimensional space."""
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
# Target function
# ------------------------------


@jax.jit
def target(x: Float[Array, "d"]) -> Float[Array, "m"]:
    z = 1 if d % 2 == 0 else x[-1]
    m = d // 2
    diff = x[:m] * z - x[m : 2 * m]
    val = jnp.prod(diff) ** 2
    return jnp.atleast_1d(val)


# ------------------------------
# Learning algorithm
# ------------------------------


def natural_gradient_descent(
    optimizer: optax.GradientTransformation,
    damping_coeff: float,
    backtracking: bool = False,
):
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

        def _block_projection(jac: Tensor) -> tuple[Tensor, float, int]:
            shape = jac.shape
            jac = jnp.reshape(
                jac, (jac.shape[0] * grad_losses.shape[1], -1)
            )  # (B*d_o, n_params)

            # computing the Gramian
            G = (
                jac.T @ jac / x.shape[0]
            )  # WARN: don't forget to divide by the number of samples!
            b = jac.T @ grad_losses.ravel()
            D = jnp.eye(G.shape[0])

            ngrad = jnp.linalg.solve(G + damping_coeff * D, b)
            ngrad = jnp.reshape(ngrad, shape[1:])

            s = jnp.linalg.eigvalsh(G)
            s = s[::-1]
            maxS = jnp.max(s, initial=0.0)
            rtol = G.shape[0] * jnp.finfo(G.dtype).eps
            val = maxS * rtol
            rank = jnp.sum(s > val)
            # jax.debug.print(
            #     "Singular values: {s} | Val: {val} | Idx: {idx}", s=s, val=val, idx=idx
            # )
            condition_number = s[0] / s[rank - 1]

            return ngrad, condition_number, rank

        ngrads, condition_numbers, ranks = zip(
            *[_block_projection(J[k]) for k in range(len(tensors))]
        )
        ngrads = list(ngrads)
        ngrad_norms = list(map(jnp.linalg.norm, ngrads))
        info = {
            "condition_number": condition_numbers,
            "rank": ranks,
            "gradient_norm": jnp.sqrt(jnp.sum(jnp.asarray(ngrad_norms) ** 2)),
        }
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

        return params, opt_state, loss, info

    def do_iter(
        params: list[Tensor],
        opt_state: optax.OptState,
        x: Float[Array, "B d"],
        y: Float[Array, "B d"],
        x_val: Float[Array, "B d"],
        y_val: Float[Array, "B d"],
    ):
        params, opt_state, l2_error, info = step(params, opt_state, x, y)

        val_loss = compute_loss(params, x_val, y_val)

        if backtracking:
            for state in opt_state:
                if lr := getattr(state, "learning_rate", None) is not None:
                    info["learning_rate"] = lr
                    break

        return params, opt_state, (l2_error, val_loss), info

    return do_iter


def run_experiment(
    key: PRNGKeyArray,
    experiment: Experiment,
) -> tuple[PRNGKeyArray, tuple[Float[Array, "n_iterations"], list[dict]]]:
    print("Key:", key)

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
    num_layers = experiment.num_layers

    # ------------------------------
    # Run optimizers
    # ------------------------------
    key, *control_keys = random.split(key, num_layers + 1)

    init_controls = [
        math.sqrt(experiment.init_constant / (num_layers * n_features**d_lift))
        * random.normal(key=control_key, shape=(d_lift,) + (n_features,) * d_lift)
        for control_key in control_keys
    ]

    opt = experiment.optimizer
    use_linesearch = (
        isinstance(opt.learning_rate, str) and opt.learning_rate == "line_search"
    )

    params = init_controls
    if isinstance(opt.learning_rate, str):
        if opt.learning_rate != "line_search":
            raise ValueError(f"Learning rate '{opt.learning_rate}' unknown.")
        optimizer = optax.chain(
            optax.sgd(learning_rate=1.0),
            optax.scale_by_zoom_linesearch(
                max_linesearch_steps=20,
                max_learning_rate=1.0,
                initial_guess_strategy="one",
            ),
        )
    else:
        optimizer = optax.sgd(learning_rate=opt.learning_rate)

    state = optimizer.init(params)
    ngd_step = natural_gradient_descent(
        optimizer=optimizer,
        damping_coeff=opt.damping_coeff,
        backtracking=use_linesearch,
    )

    train_losses = []
    val_losses = []
    rel_l2 = []
    infos = []

    with tqdm(
        range(opt.num_iterations),
        desc="Optimizing",
        leave=True,
        total=opt.num_iterations,
    ) as pbar:
        for _ in pbar:
            params, state, (train_loss, val_loss), info = ngd_step(
                params, state, X_train, y_train, X_val, y_val
            )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            infos.append(info)
            rl2_error = jnp.sqrt(val_loss / y_val_norm)
            rel_l2.append(rl2_error)

            postfix = dict(
                train_loss=f"{train_loss:.3e}",
                val_loss=f"{val_loss:.3e}",
                rel_l2=f"{rl2_error:.3e}",
                gradient_norm=f"{info['gradient_norm']:.3e}",
                condition_number="["
                + ",".join(map(lambda x: f"{x:.3e}", info["condition_number"]))
                + "]",
            )
            pbar.set_postfix(**postfix)

    return key, (jnp.asarray(rel_l2), infos)


print("------------------------------")
print("Running experiments...")

seed = 0
key = random.key(seed)
N_EXPERIMENTS = 20

rel_l2 = []
condition_numbers = []
ranks = []

for i in range(N_EXPERIMENTS):
    print(f"Experiment {i + 1}/{N_EXPERIMENTS}")
    key, (val_l2, infos) = run_experiment(key, experiment)
    rel_l2.append(val_l2)
    cond = jnp.asarray(
        [x["condition_number"] for x in infos]
    )  # (n_iterations, n_layers)
    rank = jnp.asarray([x["rank"] for x in infos])
    condition_numbers.append(cond)
    ranks.append(rank)


rel_l2 = jnp.asarray(rel_l2)
condition_numbers = jnp.asarray(
    condition_numbers
)  # (n_experiments, n_iterations, n_layers)
ranks = jnp.asarray(ranks)  # (n_experiments, n_iterations, n_layers)

# plot the results
n_iterations = experiment.optimizer.num_iterations

# condition number
condition_numbers_mean = jnp.mean(condition_numbers, axis=0)  # (n_iterations, n_layers)
condition_numbers_std = jnp.std(condition_numbers, axis=0)  # (n_iterations, n_layers)

plt.figure(figsize=(15, 8))

for i, ls, marker, color in zip(
    range(experiment.num_layers),
    ["-", "--", ":"],
    ["*", "s", "^"],
    ["#bb5566", "#004488", "#ddaa33"],
):
    plt.loglog(
        jnp.arange(1, n_iterations + 1),
        condition_numbers_mean[:, i],
        label=f"Layer {i + 1}",
        ls=ls,
        marker=marker,
        color=color,
        markevery=n_iterations // 25,
    )
    plt.fill_between(
        jnp.arange(1, n_iterations + 1),
        condition_numbers_mean[:, i] - condition_numbers_std[:, i] / N_EXPERIMENTS,
        condition_numbers_mean[:, i] + condition_numbers_std[:, i] / N_EXPERIMENTS,
        color=color,
        alpha=0.3,
    )
plt.grid()
plt.legend()
plt.xlim(
    1,
    n_iterations + 1,
)
plt.xlabel("Iterations")
plt.ylabel("Condition number")
plt.tight_layout(pad=1.10)
plt.savefig(os.path.join(directory, "data", f"{filename}.pdf"), format="pdf", dpi=1200)
plt.show()

# ranks
ranks_mean = jnp.mean(ranks, axis=0)  # (n_iterations, n_layers)
ranks_std = jnp.std(ranks, axis=0)  # (n_iterations, n_layers)

plt.figure(figsize=(15, 8))

for i, ls, marker, color in zip(
    range(experiment.num_layers),
    ["-", "--", ":"],
    ["*", "s", "^"],
    ["#bb5566", "#004488", "#ddaa33"],
):
    plt.semilogx(
        jnp.arange(1, n_iterations + 1),
        ranks_mean[:, i],
        label=f"Layer {i + 1}",
        ls=ls,
        marker=marker,
        color=color,
        markevery=n_iterations // 25,
    )
    plt.fill_between(
        jnp.arange(1, n_iterations + 1),
        ranks_mean[:, i] - ranks_std[:, i] / N_EXPERIMENTS,
        ranks_mean[:, i] + ranks_std[:, i] / N_EXPERIMENTS,
        color=color,
        alpha=0.3,
    )
plt.grid()
plt.legend()
plt.xlim(
    1,
    n_iterations + 1,
)
plt.xlabel("Iterations")
plt.ylabel("Rank")
plt.tight_layout(pad=1.10)
plt.savefig(
    os.path.join(directory, "data", f"{filename}_ranks.pdf"), format="pdf", dpi=1200
)
plt.show()

from typing import Optional

import jax
import jax.experimental
import jax.numpy as jnp
import jax.random as random
import jax.scipy.optimize
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from ctt.als import als_linear_system
from ctt.bases import make_canonical_polynomials
from ctt.model import make_ctt
from ctt.optimizer import mini_batch_pmp
from ctt.tt import (
    TT,
    cp_to_tt_truncate,
    tt_add,
    tt_matvec,
    tt_mul_scalar,
    tt_norm,
    tt_randn,
    tt_truncate,
    tt_zeros,
    tt_zeros_like,
    validate_ranks,
)
from ctt.tto import TTO, cpo_to_tto_truncate

jax.config.update("jax_enable_x64", True)

d = 1
n_features = 2
basis = make_canonical_polynomials(dim=n_features)
# basis = make_legendre_polynomials(dim=n_features)
# basis = make_fourier(n_features)
# alpha = 1e-1
# soft_relu = lambda x: alpha*jax.nn.softplus(x/alpha)
# basis = lambda x: jnp.asarray([1.0, jax.nn.relu(x)])
key = random.PRNGKey(42)


def lift(x: Float[Array, "d"]) -> Float[Array, "m"]:
    return jnp.hstack((1.0, x))
    # return x


def retraction(x: Float[Array, "m"]) -> Float[Array, "o"]:
    return jnp.asarray((x[0],))


d_lift = jax.eval_shape(lift, jax.ShapeDtypeStruct((d,), dtype=float)).shape[0]
d_retraction = jax.eval_shape(
    retraction, jax.ShapeDtypeStruct((d_lift,), dtype=float)
).shape[0]
bases = [basis] * (d_lift)

key, *control_keys = random.split(key, 2 + 1)

target_ranks = [d_lift] + [2] * (d_lift - 1) + [1]
target_tts = [
    tt_randn(control_key, [n_features] * d_lift, target_ranks, cov=1.0)
    for control_key in control_keys
]
target_ctt = make_ctt(lift, retraction, basis, d)


@jax.jit
def target(x: Float[Array, "d"]) -> Float[Array, "m"]:
    val = jnp.exp(-jnp.sum(x**2) / 2)
    # val = 1.0 / (2.0 + jnp.prod(x))
    # val = jnp.mean(x**2)
    # val = jnp.sum(jnp.sin(2*jnp.pi*x))
    # val = jnp.prod((x >= 0.0) * 1.0)
    # val = jnp.mean(jnp.exp(x))
    # val = jnp.exp(jnp.sum(x))
    # val = target_ctt(target_tts, x)
    # val = jnp.log(1.0 + jnp.sum(x**2))
    return jnp.atleast_1d(val)


def _eval_bases(x: Float[Array, "d"]) -> list[Float[Array, "m"]]:
    return list(map(lambda b, y: b(y), bases, x))


def _ftt(x: Float[Array, "d"], tt: TT) -> Float[Array, "d"]:
    features = _eval_bases(x)  # (m,)*d
    result = tt_matvec(tt, features)  # (d, 1)
    return result[:, 0]


@jax.jit
def ctt(tts: list[TT], x: Float[Array, "d"]) -> Float[Array, "d"]:
    def _body_fn(i: int, val: Float[Array, "d"]) -> Float[Array, "d"]:
        control = jax.lax.switch(i, [lambda u=u: u for u in tts])
        return val + _ftt(val, control)

    x = lift(x)
    val = jax.lax.fori_loop(0, len(tts), _body_fn, x)
    val = retraction(val)
    return val


key, train_key, val_key = random.split(key, 3)

domain = (-1.0, 1.0)
N_train = 1_000_000
X_train = random.uniform(
    key=train_key, shape=(N_train, d), minval=domain[0], maxval=domain[1]
)
y_train = jax.vmap(target)(X_train)

N_val = 2000
X_val = random.uniform(
    key=val_key, shape=(N_val, d), minval=domain[0], maxval=domain[1]
)
y_val = jax.vmap(target)(X_val)

num_steps = 4
radius = None
ranks = [d_lift] + [2] * (d_lift - 1) + [1]
ranks = validate_ranks([n_features] * d_lift, ranks)


def dataloader(arrays: list, batch_size: int, *, key: PRNGKeyArray):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = random.permutation(key, indices)
        (key,) = random.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


@jax.jit
def compute_loss(
    tts: list[TT], x: Float[Array, "B d"], y: Float[Array, "B d_o"]
) -> tuple[float, float]:
    y_pred = jax.vmap(ctt, in_axes=(None, 0))(tts, x)
    loss = jnp.mean(optax.l2_loss(y_pred, y))
    return loss


def regularization(
    xk: Float[Array, "d_lift"], control: TT, coefficient: float
) -> float:
    feats = _eval_bases(xk)
    contracted = tt_matvec(control, feats)  # (d, 1)
    return 0.5 * coefficient * jnp.sum(contracted[:, 0] ** 2)


def terminal_cost(xT: Float[Array, "d_lift"], y: Float[Array, "d_o"]) -> float:
    y_pred = retraction(xT)
    return 0.5 * jnp.sum((y_pred - y) ** 2)


def transition(xk: Float[Array, "d_lift"], control: TT) -> Float[Array, "d_lift"]:
    psi = _ftt(xk, control)
    return xk + psi


def _build_gram_op(features: list[Float[Array, "B m"]], ranks: list[int]) -> TTO:
    B = features[0].shape[0]
    d = len(features)
    G = jax.tree.map(jax.vmap(jnp.outer), features, features)
    cpo = [jnp.tile(jnp.eye(d), (B, 1, 1))] + G  # (B, d, d) + (B, m, m)*d
    # convert the CPO to TTO
    tt_op = cpo_to_tto_truncate(
        cpo, [1] + ranks
    )  # we have a d+1-order tensor operator now
    return tt_op


def _build_rhs(
    features: list[Float[Array, "B m"]],
    costates: Float[Array, "B d_lift"],
    ranks: list[int],
) -> TT:
    B = features[0].shape[0]
    d = costates.shape[1]
    first_factor = -costates
    rhs = [first_factor] + features
    # convert the CP to TT
    tt = cp_to_tt_truncate(rhs, [1] + ranks)
    return tt


def _lr_schedule(lr: float, k: int) -> float:
    return lr
    # return lr * (k + 1) ** (-0.1)


def _get_update(
    features: list[Float[Array, "B m"]],
    costates: Float[Array, "B d_lift"],
    control: TT,
    gamma: float,
    coefficient: float,
    ranks: list[int],
    k: int,
) -> TT:
    # build the TT operator
    gram_op = _build_gram_op(features, ranks)

    # build RHS
    rhs = _build_rhs(features, costates, ranks)
    # rhs = tt_orth_right(rhs)

    iters, stag, sol = als_linear_system(
        A=gram_op,
        b=rhs,
        x0=tt_zeros_like(rhs),
        # x0=tmp_control,
        max_iters=100,
        stagnation=1e-7,
        l2_regularization=1e-10,
    )
    core = jnp.einsum("abc,cde->bde", sol[0], sol[1])
    sol = [core] + sol[2:]

    # alpha = R/(gamma + R)
    alpha = coefficient / (gamma + coefficient)
    alpha = _lr_schedule(alpha, k)
    tt = tt_add(
        tt_mul_scalar(control, 1.0 - alpha),
        tt_mul_scalar(sol, alpha / coefficient),
    )
    # tt = tt_add(control, tt_mul_scalar(sol, alpha / coefficient))
    tt = tt_truncate(tt, ranks)
    return tt


def min_hamiltonian(
    states: Float[Array, "B d_lift"],
    costates: Float[Array, "B d_lift"],
    control: TT,
    k: int,
    coefficient: float,
    ranks: list[int],
    radius: Optional[float] = None,
    gamma: Optional[float] = None,
) -> TT:
    features = jax.vmap(_eval_bases)(states)

    squared_ranks = ranks[:1] + list(map(lambda x: x**2, ranks[1:]))
    tt = _get_update(features, costates, control, gamma, coefficient, squared_ranks, k)

    if radius:
        # project the TT to the closed ball of radius `radius`
        norm = tt_norm(tt)
        tt = jax.lax.cond(
            norm <= radius, lambda u: u, lambda u: tt_mul_scalar(u, radius / norm), tt
        )

    return tt


def pmp(
    train_dataset: tuple[Float[Array, "B d"], Float[Array, "B d_o"]],
    val_dataset: tuple[Float[Array, "B d"], Float[Array, "B d_o"]],
    params: list[TT],
    opt_steps: int,
    training_key: PRNGKeyArray,
    regularization_coeff: float,
    radius: Optional[float] = None,
    batch_size: int = 64,
    gamma: Optional[float] = None,
) -> tuple[list[TT], tuple[list[float], list[float]]]:
    X_train, y_train = train_dataset
    X_val, y_val = val_dataset

    X_train_lifted = jax.vmap(lift)(X_train)
    optimizer = mini_batch_pmp(
        regularization=jax.tree_util.Partial(
            regularization, coefficient=regularization_coeff
        ),
        terminal_cost=terminal_cost,
        transition=transition,
        min_hamiltonian=jax.tree_util.Partial(
            min_hamiltonian,
            coefficient=regularization_coeff,
            ranks=ranks,
            radius=radius,
            gamma=gamma,
        ),
        num_steps=num_steps,
    )
    optimizer = jax.jit(optimizer)

    train_losses = []
    val_losses = []
    for i, (x_lift, x, y) in zip(
        range(opt_steps),
        dataloader((X_train_lifted, X_train, y_train), batch_size, key=training_key),
    ):
        params = optimizer(params, x_lift, y, i)

        train_loss = compute_loss(params, x, y)
        val_loss = compute_loss(params, X_val, y_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Iter {i} | Train loss: {train_loss:.3e} | Val loss: {val_loss:.3e}")

    return params, (train_losses, val_losses)


key, *control_keys = random.split(key, num_steps + 1)

init_controls = [
    tt_randn(control_key, [n_features] * d_lift, ranks, cov=0.1)
    for control_key in control_keys
]
zero_init_controls = [tt_zeros([n_features] * d_lift, ranks) for _ in range(num_steps)]
print("Init controls norm:", list(map(tt_norm, init_controls)))

key, training_key = random.split(key)
# key, *perturbation_keys = random.split(key, 1 + num_steps)


# def _perturb_tt(tt: TT, key: PRNGKeyArray) -> TT:
#     perturbed = []
#     for i in range(len(tt)):
#         key, perturbation_key = random.split(key)
#         core = tt[i]
#         perturbed.append(
#             core + 1e-5 * random.normal(key=perturbation_key, shape=core.shape)
#         )
#     return perturbed


# perturbated_target_tts = list(
#     map(
#         _perturb_tt,
#         target_tts,
#         perturbation_keys,
#     )
# )
gamma = 1000
R = 1e-15
print("Alpha =", R / (gamma + R))
print("Learning rate =", 1.0 / (gamma + R))
# TODO: use linesearch for alpha
params_pmp, (train_losses_pmp, val_losses_pmp) = pmp(
    train_dataset=(X_train, y_train),
    val_dataset=(X_val, y_val),
    params=zero_init_controls,
    opt_steps=20000,
    regularization_coeff=R,
    radius=None,
    gamma=gamma,
    training_key=training_key,
    batch_size=128,
)

fig, ax = plt.subplots()

y_val_norm = jnp.mean(optax.l2_loss(y_val))

# plot PMP losses
val_losses_pmp = jnp.asarray(val_losses_pmp)

ax.semilogy(
    jnp.arange(len(val_losses_pmp)),
    jnp.sqrt(val_losses_pmp / y_val_norm),
    label="Relative L2 (PMP)",
    color="blue",
)
# ax.semilogy(jnp.arange(len(val_losses_pmp)), jnp.sqrt(val_losses_pmp), label="Absolute L2", color="blue", linestyle="dashed")

ax.grid()
ax.legend()
ax.set_xlabel("Iterations")
ax.set_ylabel("Error")
plt.show()

import jax
import jax.experimental
import jax.numpy as jnp
import jax.random as random
import jax.scipy.optimize
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float, PRNGKeyArray

from ctt.bases import make_canonical_polynomials
from ctt.model import make_ctt
from ctt.optimizer import MSAState, natural_msa
from ctt.tt import (
    TT,
    tt_matvec,
    tt_norm,
    tt_randn,
    tt_zeros,
    validate_ranks,
)

jax.config.update("jax_enable_x64", True)

d = 2
n_features = 2
basis = make_canonical_polynomials(dim=n_features)
# basis = make_legendre_polynomials(n_features)
# basis = make_chebyshev_polynomials(dim=n_features)
# basis = make_fourier(n_features)
# alpha = 1e-1
# soft_relu = lambda x: alpha*jax.nn.softplus(x/alpha)
# basis = lambda x: jnp.asarray([1.0, jax.nn.relu(x)])
key = random.PRNGKey(42)

key, cov_key = random.split(key)
Sigma = random.normal(key=cov_key, shape=(d, d))
Sigma = Sigma @ Sigma.T / d


def lift(x: Float[Array, "d"]) -> Float[Array, "m"]:
    return jnp.hstack((jnp.zeros((1,)), x))
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
    # val = jnp.exp(-jnp.sum(x**2) / 2)
    # val = 1.0 / (2.0 + jnp.prod(x))
    # val = jnp.mean(x**2)
    # val = jnp.sum(jnp.sin(2*jnp.pi*x))
    # val = jnp.prod((x >= 0.0) * 1.0)
    # val = jnp.mean(jnp.exp(x))
    # val = jnp.exp(jnp.sum(x))
    # val = target_ctt(target_tts, x)
    # val = jnp.log(1.0 + jnp.sum(x**2))
    # val = jax.scipy.stats.multivariate_normal.pdf(
    #     x=x, mean=jnp.zeros_like(x), cov=Sigma
    # )
    # A = jnp.array(
    #     [
    #         [jnp.cos(jnp.pi / 4), -jnp.sin(jnp.pi / 4)],
    #         [jnp.sin(jnp.pi / 4), jnp.cos(jnp.pi / 4)],
    #     ]
    # )
    # y = A @ x
    # val = jnp.exp(-jnp.sum(y**2) / 10)
    # Henon-Heiles potential
    # val = (
    #     0.5 * jnp.sum(x**2)
    #     + 0.2 * jnp.sum(x[:-1] * x[1:] ** 2 - x[:-1] ** 3)
    #     + 0.2**2 / 16 * jnp.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2)
    # )
    val = jnp.mean((x[::2] + x[1::2]) ** 2)
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

domain = (0.0, 1.0)
N_train = 2_000_000
X_train = random.uniform(
    key=train_key, shape=(N_train, d), minval=domain[0], maxval=domain[1]
)
y_train = jax.vmap(target)(X_train)

N_val = 2000
X_val = random.uniform(
    key=val_key, shape=(N_val, d), minval=domain[0], maxval=domain[1]
)
y_val = jax.vmap(target)(X_val)

num_steps = 2
ranks = [d_lift] + [2] * (d_lift - 1) + [1]
ranks = validate_ranks([n_features] * d_lift, ranks)

print(jnp.min(y_train), jnp.max(y_train))


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


def terminal_cost(xT: Float[Array, "d_lift"], y: Float[Array, "d_o"]) -> float:
    y_pred = retraction(xT)
    return 0.5 * jnp.sum((y_pred - y) ** 2)


def transition(xk: Float[Array, "d_lift"], control: TT) -> Float[Array, "d_lift"]:
    psi = _ftt(xk, control)
    return xk + psi


def pmp(
    train_dataset: tuple[Float[Array, "B d"], Float[Array, "B d_o"]],
    val_dataset: tuple[Float[Array, "B d"], Float[Array, "B d_o"]],
    params: list[TT],
    opt_steps: int,
    training_key: PRNGKeyArray,
    regularization_coeff: float,
    batch_size: int = 64,
) -> tuple[list[TT], tuple[list[float], list[float]]]:
    X_train, y_train = train_dataset
    X_val, y_val = val_dataset

    X_train_lifted = jax.vmap(lift)(X_train)
    optimizer = natural_msa(
        R=regularization_coeff,
        terminal_cost=terminal_cost,
        transition=transition,
        bases=bases,
        step_size=1e-3,
    )
    optimizer = jax.jit(optimizer)

    train_losses = []
    val_losses = []
    state = MSAState(iterations=0)
    for i, (x_lift, x, y) in zip(
        range(opt_steps),
        dataloader((X_train_lifted, X_train, y_train), batch_size, key=training_key),
    ):
        params, state = optimizer(params, x_lift, state, y)

        train_loss = compute_loss(params, x, y)
        val_loss = compute_loss(params, X_val, y_val)
        if jnp.isnan(train_loss):
            print("Panic!")
            break
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        relative_val_loss = jnp.sqrt(val_loss / jnp.mean(optax.l2_loss(y)))

        print(
            f"Iter {i} | Train loss: {train_loss:.3e} | Val loss: {val_loss:.3e} | Val relative L2: {relative_val_loss:.3e}"
        )

    return params, (train_losses, val_losses)


key, *control_keys = random.split(key, num_steps + 1)

init_controls = [
    tt_randn(control_key, [n_features] * d_lift, ranks, cov=1e-2)
    for control_key in control_keys
]
zero_init_controls = [tt_zeros([n_features] * d_lift, ranks) for _ in range(num_steps)]
print("Init controls norm:", list(map(tt_norm, init_controls)))

key, training_key = random.split(key)

R = 1e-12

params_pmp, (train_losses_pmp, val_losses_pmp) = pmp(
    train_dataset=(X_train, y_train),
    val_dataset=(X_val, y_val),
    params=zero_init_controls,
    opt_steps=10_000,
    regularization_coeff=R,
    training_key=training_key,
    batch_size=256,
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

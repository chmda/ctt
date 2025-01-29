from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from jaxtyping import Array, Float, PRNGKeyArray

from ctt.bases import make_canonical_polynomials
from ctt.optimizer import Control, batch_pmp
from ctt.tt import TT, tt_add, tt_matvec, tt_mul_scalar, tt_norm, tt_retract, tt_zeros

jax.config.update("jax_enable_x64", True)

d = 5
n_features = 2
# basis = make_fourier(dim=n_features, domain=(-3.0, 3.0))
basis = make_canonical_polynomials(n_features)
# basis = lambda x: jnp.array([1.0, x, jnp.abs(x)])
bases = [basis] * (d + 1)
key = random.PRNGKey(42)


def _random_cov_matrix(key: PRNGKeyArray, d: int, l: int = 2) -> Float[Array, "d d"]:
    A = random.uniform(key=key, shape=(d, d), minval=-1.0, maxval=1.0)
    prec = A @ A.T

    error = 1.0
    while error > 1e-4:
        for i in range(d - 1):
            sub_diag = prec[i + 1 :, : i + 1]
            u, s, vh = jnp.linalg.svd(sub_diag, full_matrices=False)
            s = jnp.zeros_like(s)
            s = s.at[: min(l, s.shape[0])].set(1.0)
            new_prec = prec.at[i + 1 :, : i + 1].set((u * s) @ vh)
            new_prec = new_prec.at[: i + 1, i + 1 :].set(new_prec[i + 1 :, : i + 1].T)
            error = jnp.linalg.norm(prec - new_prec) / jnp.linalg.norm(prec)
            prec = new_prec

    return jnp.linalg.inv(prec)


key, cov_key = random.split(key)
# cov = _random_cov_matrix(key=cov_key, d=d, l=2)


@jax.jit
def target(x: Float[Array, "d"]) -> float:
    # return jax.scipy.stats.multivariate_normal.pdf(x, mean=jnp.zeros_like(x), cov=cov)
    return 1.0 / (1.0 + jnp.mean(x))


def _ftt(x: Float[Array, "d"], tt: TT) -> Float[Array, "d"]:
    features = [bases[i](x[i]) for i in range(x.shape[0])]  # (m,)*d
    result = tt_matvec(tt, features)  # (d, 1)
    return result[:, 0]


def regularization(
    xk: Float[Array, "B d"], control: Control, coefficient: float
) -> float:
    return coefficient * tt_norm(control) ** 2


def terminal_cost(xT: Float[Array, "B d"], y: Float[Array, "B d"]) -> float:
    return 0.5 * jnp.mean((xT - y) ** 2)


def transition(xk: Float[Array, "B d"], control: Control) -> Float[Array, "B d"]:
    psi = jax.vmap(_ftt, in_axes=(0, None))(xk, control)
    return xk + psi


def _compute_B(
    features: list[Float[Array, "B m"]], costates: Float[Array, "B d"], ranks: list[int]
) -> TT:
    assert len(ranks) == len(features) + 1
    d = len(features)
    # compute the first component by multiplying the first factor by the costates `lambda`
    first_comp = jnp.einsum("bd,bm->dmb", costates, features[0])  # (d, m, B)

    def _factor_to_core(factor: Float[Array, "r n"]) -> Float[Array, "r n r"]:
        r, n = factor.shape
        return jax.vmap(jnp.diag, in_axes=1)(factor).transpose(2, 0, 1)

    # converting the factors to TT cores and projecting back to the TT manifold
    tt_cores = [None] * d
    tt_cores[0] = first_comp
    for mu in range(d - 1):
        core = tt_cores[mu]
        r1, m, r2 = core.shape

        # compute the SVD of the current core
        u, s, vh = jnp.linalg.svd(core.reshape(r1 * m, r2), full_matrices=False)
        new_rank = ranks[mu + 1]
        # truncate the SVD to project back to the TT manifold
        u, s, vh = u[:, :new_rank], s[:new_rank], vh[:new_rank, :]

        # replace the TT core
        tt_cores[mu] = u.reshape((r1, m, new_rank))
        # update the next core
        if mu == d - 2:
            next_core = features[-1][..., None]
        else:
            next_core = _factor_to_core(features[mu + 1])
        tt_cores[mu + 1] = jnp.einsum("ir,rkl->ikl", s[:, None] * vh, next_core)

    return tt_cores


def min_hamiltonian(
    states: Float[Array, "B d"],
    costates: Float[Array, "B d"],
    control: Control,
    coefficient: float,
    ranks: list[int],
    radius: Optional[float] = None,
) -> Control:
    B, d = states.shape
    features = [
        jax.vmap(bases[i])(states[:, i]) for i in range(states.shape[1])
    ]  # (B, m)*d
    tt = _compute_B(features, costates, ranks)
    tt = tt_mul_scalar(tt, -1.0 / coefficient)
    lr = 1e-5
    tt = tt_add(tt_mul_scalar(control, 1.0 - lr), tt_mul_scalar(tt, lr))
    tt = tt_retract(tt, ranks)
    if radius:
        # project the TT to the closed ball of radius `radius`
        norm = tt_norm(tt)
        tt = jax.lax.cond(
            norm <= radius, lambda u: u, lambda u: tt_mul_scalar(u, radius / norm), tt
        )

    return tt


@jax.jit
def compute_ctt(x: Float[Array, "B d"], controls: list[Control]) -> Float[Array, "B d"]:
    def _body_fn(idx: int, val: Float[Array, "B d"]) -> Float[Array, "B d"]:
        control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
        next_state = transition(val, control)
        return next_state

    result = jax.lax.fori_loop(0, len(controls), _body_fn, x)
    return result


key, train_key, test_key = random.split(key, 3)

N_train = 3000
# X_train = random.multivariate_normal(
#     key=train_key, mean=jnp.zeros(d), cov=cov, shape=(N_train,)
# )
X_train = random.uniform(key=train_key, shape=(N_train, d), minval=0.0, maxval=1.0)
y_train = jax.vmap(target)(X_train)
X_train = jnp.concatenate((jnp.ones((N_train, 1)), X_train), axis=1)
y_train = jnp.concatenate((y_train[:, None], X_train[:, 1:]), axis=1)

N_test = 1000
# X_test = random.multivariate_normal(
#     key=train_key, mean=jnp.zeros(d), cov=cov, shape=(N_test,)
# )
X_test = random.uniform(key=test_key, shape=(N_test, d), minval=0.0, maxval=1.0)
y_test = jax.vmap(target)(X_test)
X_test = jnp.concatenate((jnp.ones((N_test, 1)), X_test), axis=1)
y_test = jnp.concatenate((y_test[:, None], X_test[:, 1:]), axis=1)

num_steps = 5
regularization_coeff = 1e-4
radius = 1.0
ranks = [d + 1] + [2] * (d) + [1]
pmp = batch_pmp(
    regularization=jax.tree_util.Partial(
        regularization, coefficient=regularization_coeff
    ),
    terminal_cost=jax.tree_util.Partial(terminal_cost, y=y_train),
    transition=transition,
    min_hamiltonian=jax.tree_util.Partial(
        min_hamiltonian, coefficient=regularization_coeff, ranks=ranks, radius=radius
    ),
    x0=X_train,
    num_steps=num_steps,
)
pmp = jax.jit(pmp)

control_keys = random.split(key, num=num_steps)
init_controls = [
    # tt_randn(control_key, [n_features] * (d + 1), ranks, cov=2.0)
    # for control_key in control_keys
    tt_zeros([n_features] * (d + 1), ranks)
    for _ in control_keys
]

opt_steps = 100
controls = init_controls
losses = []
for i in range(opt_steps):
    controls = pmp(controls)
    xT = compute_ctt(X_test, controls)
    loss = terminal_cost(xT, y_test)
    losses.append(loss)
    print(f"Iter {i} | Loss: {loss:.3e}")

# plot the loss
plt.semilogy(jnp.arange(opt_steps), jnp.sqrt(jnp.asarray(losses)), label="Absolute L2")
plt.semilogy(
    jnp.arange(opt_steps),
    jnp.sqrt(jnp.asarray(losses) / terminal_cost(jnp.zeros_like(y_test), y_test)),
    label="Relative L2",
)
plt.xticks(jnp.arange(opt_steps))
plt.grid()
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("loss.jpg")
plt.show()

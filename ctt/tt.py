import math
from typing import Callable, Optional

import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray

TTCore = Float[Array, "r0 m r1"]
TT = list[TTCore]


def tt_size(tt: TT) -> int:
    return sum(core.size for core in tt)


def tt_dims(tt: TT) -> list[int]:
    return [core.shape[1] for core in tt]


def tt_ranks(tt: TT) -> list[int]:
    return [core.shape[0] for core in tt] + [tt[-1].shape[2]]


def tt_mul_scalar(tt: TT, val: float) -> TT:
    new_core = val * tt[0]
    return [new_core] + tt[1:]


def tt_dot(a: TT, b: TT) -> float:
    assert len(a) == len(b), "the two TTs must be of the same order"
    res = jnp.einsum("oab,oac->obc", a[0], b[0])
    for i in range(1, len(a)):
        res = jnp.einsum("obc,bnd,cnf->odf", res, a[i], b[i])
    return jnp.sum(res)


def tt_norm(tt: TT) -> float:
    return jnp.sqrt(tt_dot(tt, tt))


def tt_dot_rank_one(a: TT, b: list[Float[Array, "m"]]) -> float:
    cores = [v.reshape(1, -1, 1) for v in b]
    return tt_dot(a, cores)


def _tt_modify_ranks(
    tt: TT,
    rule: Callable[
        [Float[Array, "n*r0 r"], Float[Array, "r"], Float[Array, "r r1"], int], int
    ],
) -> TT:
    new_cores = [core.copy() for core in tt]
    for pos in range(len(tt) - 1):
        core = new_cores[pos]
        shape = core.shape

        u, s, vh = jnp.linalg.svd(
            core.reshape(shape[0] * shape[1], shape[2]), full_matrices=False
        )
        new_rank = rule(u, s, vh, pos)

        u, s, vh = u[:, :new_rank], s[:new_rank], vh[:new_rank, :]
        new_cores[pos] = u.reshape((shape[0], shape[1], new_rank))
        new_cores[pos + 1] = jnp.einsum("ir,rkl->ikl", s * vh, new_cores[pos + 1])

    return new_cores


def tt_round(tt: TT, epsilon: float) -> TT:
    delta = epsilon / math.sqrt(len(epsilon) - 1) * tt_norm(tt)

    def rule(u, s, vh, pos):
        return max(jnp.sum(s > delta).item(), 1)

    return _tt_modify_ranks(tt, rule)


def tt_retract(tt: TT, ranks: list[int]) -> TT:
    def rule(u, s, vh, pos):
        return ranks[pos]

    return _tt_modify_ranks(tt, rule)


def canonical_to_tt(cores: list[Float[Array, "r n"]]) -> TT:
    new_cores = []
    new_cores[0] = cores[0][None, ...]

    for mu in range(1, len(cores) - 1):
        factor = cores[mu]
        core = jnp.zeros(
            (factor.shape[0], factor.shape[0] + 1, factor.shape[1]), dtype=factor.dtype
        )
        core = core.at[..., 0, :].set(factor)
        core = core.reshape((factor.shape[0] + 1, factor.shape[0], factor.shape[1]))
        core = jnp.transpose(core, (0, 2, 1))
        new_cores.append(core[..., :-1, :, :])

    return new_cores


def _make_tt(
    dims: list[int],
    ranks: list[int],
    func: Callable[
        [tuple[int, int, int], Optional[PRNGKeyArray]],
        tuple[TTCore, Optional[PRNGKeyArray]],
    ],
    key: Optional[PRNGKeyArray] = None,
) -> TT:
    assert len(ranks) == len(dims) + 1
    cores = []
    for mu in range(len(dims)):
        shape = (ranks[mu], dims[mu], ranks[mu + 1])
        core, key = func(shape, key)
        cores.append(core)

    return cores


def tt_rand(
    key: PRNGKeyArray,
    dims: list[int],
    ranks: list[int],
    low: float = -1.0,
    high: float = 1.0,
) -> TT:
    def func(shape, key):
        key, sample_key = random.split(key)
        return random.uniform(key=sample_key, shape=shape, minval=low, maxval=high), key

    return _make_tt(dims, ranks, func, key)


def tt_randn(
    key: PRNGKeyArray,
    dims: list[int],
    ranks: list[int],
    mean: float = 0.0,
    cov: float = 1.0,
) -> TT:
    def func(shape, key):
        key, sample_key = random.split(key)
        return mean + cov * random.normal(key=sample_key, shape=shape), key

    return _make_tt(dims, ranks, func, key)


def tt_zeros(dims: list[int], ranks: list[int]) -> TT:
    def func(shape, key):
        return jnp.zeros(shape), None

    return _make_tt(dims, ranks, func, None)

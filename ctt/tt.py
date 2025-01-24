import math
from typing import Callable, Optional, Protocol

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray

__all__ = [
    "TTCore",
    "TT",
    "tt_size",
    "tt_dims",
    "tt_ranks",
    "tt_mul_scalar",
    "tt_dot",
    "tt_norm",
    "tt_dot_rank_one",
    "tt_round",
    "tt_retract",
    "canonical_to_tt",
    "tt_rand",
    "tt_randn",
    "tt_zeros",
    "tt_matvec",
]

TTCore = Float[Array, "r0 m r1"]
TT = list[TTCore]


class RankRule(Protocol):
    def __call__(
        self,
        u: Float[Array, "n*r0 r"],
        s: Float[Array, "r"],
        vh: Float[Array, "r r1"],
        pos: int,
    ) -> int: ...


def tt_size(tt: TT) -> int:
    return sum(core.size for core in tt)


def tt_dims(tt: TT) -> list[int]:
    return [core.shape[1] for core in tt]


def tt_ranks(tt: TT) -> list[int]:
    return [core.shape[0] for core in tt] + [tt[-1].shape[2]]


def tt_mul_scalar(tt: TT, val: float) -> TT:
    new_core = val * tt[0]
    return [new_core] + tt[1:]


def tt_add(a: TT, b: TT) -> TT:
    assert all(c1.shape[1] == c2.shape[1] for c1, c2 in zip(a, b))

    comps = [jnp.concatenate((a[0], b[0]), axis=2)]

    for mu in range(1, len(a) - 1):
        c1, c2 = a[mu], b[mu]
        rp1, _, rpp1 = c1.shape
        rp2, _, rpp2 = c2.shape
        data = jnp.zeros((rp1 + rp2, c1.shape[1], rpp1 + rpp2))
        data = data.at[:rp1, :, :rpp1].set(c1)
        data = data.at[rp1:, :, rpp1:].set(c2)
        comps.append(data)

    data = jnp.concatenate((a[-1][:, :, 0], b[-1][:, :, 0]), axis=0)
    data = data[..., None]
    comps.append(data)
    return comps


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


def tt_matvec(tt: TT, x: list[Float[Array, "m"]]) -> Float[Array, "r0 rd"]:
    res = jnp.einsum("oab,a->ob", tt[0], x[0])
    for i in range(1, len(tt)):
        res = jnp.einsum("ob,bnd,n->od", res, tt[i], x[i])
    return res


def _tt_modify_ranks(tt: TT, rule: RankRule) -> TT:
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
        new_cores[pos + 1] = jnp.einsum(
            "ir,rkl->ikl", s[:, None] * vh, new_cores[pos + 1]
        )

    return new_cores


def tt_round(tt: TT, epsilon: float) -> TT:
    delta = epsilon / math.sqrt(len(epsilon) - 1) * tt_norm(tt)

    def rule(u, s, vh, pos):
        return max(jnp.sum(s > delta).item(), 1)

    return _tt_modify_ranks(tt, rule)


def tt_retract(tt: TT, inner_ranks: list[int]) -> TT:
    def rule(u, s, vh, pos):
        return inner_ranks[pos]

    return _tt_modify_ranks(tt, rule)


def canonical_to_tt(cores: list[Float[Array, "r n"]]) -> TT:
    # new_cores = []
    # new_cores[0] = cores[0][None, ...]

    # for mu in range(1, len(cores) - 1):
    #     factor = cores[mu]
    #     core = jnp.zeros(
    #         (factor.shape[0], factor.shape[0] + 1, factor.shape[1]), dtype=factor.dtype
    #     )
    #     core = core.at[..., 0, :].set(factor)
    #     core = core.reshape((factor.shape[0] + 1, factor.shape[0], factor.shape[1]))
    #     core = jnp.transpose(core, (0, 2, 1))
    #     new_cores.append(core[..., :-1, :, :])

    # return new_cores

    # tt_cores = [None] * len(cores)
    # tt_cores[0] = cores[0].copy()[None, ...]
    # tt_cores[-1] = cores[-1].copy()[..., None]

    def _factor_to_core(factor: Float[Array, "r n"]) -> Float[Array, "r n r"]:
        r, n = factor.shape
        """
        core = jnp.zeros((r, n, r), dtype=factor.dtype)
        return core.at[jnp.arange(r), :, jnp.arange(r)].set(core)
        """
        return jax.vmap(jnp.diag, in_axes=1)(factor).transpose(2, 0, 1)

    tt_cores = (
        [cores[0].copy()[None, ...]]
        + jax.tree.map(_factor_to_core, cores[1:-1])
        + [cores[-1].copy()[..., None]]
    )

    # for mu in range(1, len(cores) - 1):
    #     factor = cores[mu]
    #     R, N = factor.shape
    #     core = jnp.zeros((R, N, R), dtype=factor.dtype)
    #     tt_cores[mu] = core.at[jnp.arange(R), :, jnp.arange(R)].set(factor)

    return tt_cores


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

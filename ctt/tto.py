import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ctt.tt import TT, TTCore

TTOCore = Float[Array, "r0 row col r1"]
TTO = list[TTOCore]
CPO = list[Float[Array, "r row col"]]


def cpo_to_tto_truncate(cpo: CPO, ranks: list[int]) -> TTO:
    d = len(cpo)
    assert len(ranks) == d + 1

    def _factor_to_core(factor: Float[Array, "r row col"]) -> TTOCore:
        unfolded = jnp.reshape(factor, (factor.shape[0], -1))
        core = jax.vmap(jnp.diag, in_axes=1)(unfolded)
        core = jnp.transpose(core, (2, 0, 1))
        core = jnp.reshape(core, factor.shape + (factor.shape[0],))
        return core

    def _truncate_factor(a: TTOCore, b: TTOCore, rank: int) -> tuple[TTOCore, TTOCore]:
        r1, m, n, r2 = a.shape
        # compute the SVD of the current core
        u, s, vh = jnp.linalg.svd(a.reshape(r1 * m * n, r2), full_matrices=False)
        # truncate the SVD to project back to the TT manifold
        u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]

        # replace the TT core
        return u.reshape((r1, m, n, rank)), jnp.einsum("i,ir,rkdl->ikdl", s, vh, b)

    # converting the factors to TT cores and right-orthogonalize
    tto_cores = [None] * d
    tto_cores[-1] = cpo[-1][..., None]
    for mu in range(d - 1, 0, -1):
        core = tto_cores[mu]
        if mu == 1:
            previous_core = jnp.transpose(cpo[0], (1, 2, 0))[None, ...]
        else:
            previous_core = _factor_to_core(cpo[mu - 1])

        unfolded = jnp.reshape(core, (core.shape[0], -1))
        q, r = jnp.linalg.qr(unfolded.T)
        qT = q.T
        comp = jnp.reshape(
            qT, (qT.shape[0], core.shape[1], core.shape[2], core.shape[3])
        )
        comp_previous = jnp.einsum("ijkl,lm->ijkm", previous_core, r.T)

        tto_cores[mu] = comp
        tto_cores[mu - 1] = comp_previous

    # truncate the cores
    for mu in range(d - 1):
        core = tto_cores[mu]
        next_core = tto_cores[mu + 1]
        tto_cores[mu], tto_cores[mu + 1] = _truncate_factor(
            core, next_core, ranks[mu + 1]
        )

    return tto_cores


def cpo_to_tto_rounding(cpo: CPO, epsilon: float) -> TTO:
    d = len(cpo)
    delta = epsilon * cpo_norm(cpo) / jnp.sqrt(d - 1)

    def _factor_to_core(factor: Float[Array, "r row col"]) -> TTOCore:
        unfolded = jnp.reshape(factor, (factor.shape[0], -1))
        core = jax.vmap(jnp.diag, in_axes=1)(unfolded)
        core = jnp.transpose(core, (2, 0, 1))
        core = jnp.reshape(core, factor.shape + (factor.shape[0],))
        return core

    def _truncate_factor(a: TTOCore, b: TTOCore) -> tuple[TTOCore, TTOCore]:
        r1, m, n, r2 = a.shape
        # compute the SVD of the current core
        u, s, vh = jnp.linalg.svd(a.reshape(r1 * m * n, r2), full_matrices=False)
        cumsum = jnp.sqrt(jnp.cumsum(s[::-1] ** 2))
        rank = max(s.shape[0] - jnp.sum(jnp.sqrt(cumsum) <= delta), 1)
        # truncate the SVD to project back to the TT manifold
        u, s, vh = u[:, :rank], s[:rank], vh[:rank, :]

        # replace the TT core
        return u.reshape((r1, m, n, rank)), jnp.einsum("i,ir,rkdl->ikdl", s, vh, b)

    # converting the factors to TT cores and right-orthogonalize
    tto_cores = [None] * d
    tto_cores[-1] = cpo[-1][..., None]
    for mu in range(d - 1, 0, -1):
        core = tto_cores[mu]
        if mu == 1:
            previous_core = jnp.transpose(cpo[0], (1, 2, 0))[None, ...]
        else:
            previous_core = _factor_to_core(cpo[mu - 1])

        unfolded = jnp.reshape(core, (core.shape[0], -1))
        q, r = jnp.linalg.qr(unfolded.T)
        qT = q.T
        comp = jnp.reshape(
            qT, (qT.shape[0], core.shape[1], core.shape[2], core.shape[3])
        )
        comp_previous = jnp.einsum("ijkl,lm->ijkm", previous_core, r.T)

        tto_cores[mu] = comp
        tto_cores[mu - 1] = comp_previous

    # round the cores
    for mu in range(d - 1):
        core = tto_cores[mu]
        next_core = tto_cores[mu + 1]
        tto_cores[mu], tto_cores[mu + 1] = _truncate_factor(core, next_core)

    return tto_cores


def tto_contract_tt(tto: TTO, tt: TT) -> TT:
    def _contract(op_core: TTOCore, tt_core: TTCore) -> TTCore:
        result = jnp.einsum("abcd,ecf->aebdf", op_core, tt_core)
        return jnp.reshape(
            result,
            (
                op_core.shape[0] * tt_core.shape[0],
                op_core.shape[1],
                op_core.shape[3] * tt_core.shape[2],
            ),
        )

    return list(map(_contract, tto, tt))


def cpo_norm(cpo: CPO) -> float:
    R = cpo[0].shape[0]

    result = jnp.ones((R, R))
    for factor in cpo:
        unfolded = factor.reshape(R, -1)
        gram = jnp.dot(unfolded, unfolded.T)
        result *= gram

    return jnp.sqrt(jnp.sum(result))

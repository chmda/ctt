import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

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

    # converting the factors to TT cores and projecting back to the TT manifold
    tto_cores = [None] * d
    tto_cores[0] = jnp.transpose(cpo[0], (1, 2, 0))[None, ...]
    for mu in range(d - 1):
        core = tto_cores[mu]
        if mu == d - 2:
            next_core = cpo[-1][..., None]
        else:
            next_core = _factor_to_core(cpo[mu + 1])
        tto_cores[mu], tto_cores[mu + 1] = _truncate_factor(
            core, next_core, ranks[mu + 1]
        )

    return tto_cores

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float

__all__ = ["FUNCTIONS", "benchmark_function"]

FUNCTIONS = ["gaussian", "henon-heiles", "toy", "sos"]


def benchmark_function(
    name: str, d: int
) -> Callable[[Float[Array, "d"]], Float[Array, "m"]]:
    if name.lower() not in FUNCTIONS:
        raise ValueError(
            f"Benchmark function '{name}' does not exist. Supported functions: {','.join(FUNCTIONS)}"
        )

    if name == "gaussian":
        sample_key = random.key(0)
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

    @jax.jit
    def target(x: Float[Array, "d"]) -> Float[Array, "m"]:
        """Compute target function value at x based on experiment type."""
        if name == "gaussian":
            val = jax.scipy.stats.multivariate_normal.pdf(
                x=x, mean=jnp.zeros_like(x), cov=Sigma
            )
        elif name == "henon-heiles":
            val = (
                0.5 * jnp.sum(x**2)
                + 0.2 * jnp.sum(x[:-1] * x[1:] ** 2 - x[:-1] ** 3)
                + 0.2**2 / 16 * jnp.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2)
            )
        elif name == "toy":
            # val = jnp.log(1e-1 + 3*jnp.sum(x)**2)
            val = jnp.exp(jnp.prod(x)) / jnp.exp(1)
        elif name == "sos":
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

    return target

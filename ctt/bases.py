from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["Basis", "make_fourier", "make_canonical_polynomials"]


class Basis(Protocol):
    def __call__(self, x: float) -> Float[Array, "m"]: ...


def make_fourier(dim: int, domain: tuple[float, float] = (0.0, 1.0)) -> Basis:
    length = domain[1] - domain[0]
    inv_norm = jnp.sqrt(2.0 / length)
    n_sin = dim // 2
    n_cos = (dim + 1) // 2

    def _basis(x: float) -> Float[Array, "m"]:
        eval_cos = jnp.cos(2 * jnp.arange(0, n_cos) * jnp.pi / length * x)
        eval_sin = jnp.sin(2 * jnp.arange(1, n_sin + 1) * jnp.pi / length * x)
        return jnp.concatenate((eval_cos, eval_sin)) * inv_norm

    return _basis


def make_canonical_polynomials(dim: int) -> Basis:
    def _basis(x: float) -> Float[Array, "m"]:
        return x ** jnp.arange(dim)

    return _basis

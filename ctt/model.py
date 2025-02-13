from typing import Callable, Protocol

import jax
from jaxtyping import Array, Float

from ctt.bases import Basis
from ctt.tt import TT, tt_matvec


class CTT(Protocol):
    def __call__(self, tts: list[TT], x: Float[Array, "d"]) -> Float[Array, "d_o"]: ...


def _ftt(bases: list[Basis], x: Float[Array, "d"], tt: TT) -> Float[Array, "d"]:
    features = [bases[i](x[i]) for i in range(x.shape[0])]  # (m,)*d
    result = tt_matvec(tt, features)  # (d, 1)
    return result[:, 0]


def make_ctt(
    lift: Callable[[Float[Array, "d"]], Float[Array, "d_lift"]],
    retraction: Callable[[Float[Array, "d_lift"]], Float[Array, "d_o"]],
    basis: Basis,
    dim: int,
) -> CTT:
    d_lift = jax.eval_shape(lift, jax.ShapeDtypeStruct((dim,), dtype=float)).shape[0]
    bases = [basis] * (d_lift)

    def _ctt(tts: list[TT], x: Float[Array, "d"]) -> Float[Array, "d_o"]:
        def _body_fn(i: int, val: Float[Array, "d"]) -> Float[Array, "d"]:
            control = jax.lax.switch(i, [lambda u=u: u for u in tts])
            return val + _ftt(bases, val, control)

        x = lift(x)
        val = jax.lax.fori_loop(0, len(tts), _body_fn, x)
        val = retraction(val)
        return val

    return _ctt

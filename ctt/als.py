from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ctt.tt import (
    TT,
    TTCore,
    tt_dot,
    tt_norm,
    tt_orth_right,
    tt_shift_left,
    tt_shift_right,
)
from ctt.tto import TTO, TTOCore


def _solve_core(
    A: Float[Array, "r0*row*r1 r0*col*r1"],
    b: Float[Array, "r0*row*r1"],
    *,
    l2_regularization: Optional[float] = None,
) -> Float[Array, "r0*row*r1"]:
    if l2_regularization is None:
        return jnp.linalg.lstsq(A, b)[0]
    else:
        AT = A.T
        return jnp.linalg.solve(
            jnp.dot(AT, A) + l2_regularization * jnp.eye(A.shape[1]), jnp.dot(AT, b)
        )


@jax.jit
def _compute_stagnation(A: TT, B: TT) -> float:
    EPS = 1e-8
    norm_a = tt_norm(A) ** 2
    norm_b = tt_norm(B) ** 2
    inner_product = tt_dot(A, B)
    return jnp.sqrt(abs((norm_a - 2.0 * inner_product + norm_b) / (norm_a + EPS)))


def _compute_left_right_stack(
    tt: TT, A: TTO, b: TT
) -> tuple[tuple[list, list], tuple[list, list]]:
    d = len(tt)
    left_op = [None] * d
    left_op[0] = jnp.array([1], ndmin=3)
    left_rhs = [None] * d
    left_rhs[0] = jnp.array([1], ndmin=2)
    right_op = [None] * d
    right_op[-1] = jnp.array([1], ndmin=3)
    right_rhs = [None] * d
    right_rhs[-1] = jnp.array([1], ndmin=2)
    for k in range(d - 1, 0, -1):
        right_op[k - 1] = jnp.einsum(
            "abc,dec,gbie,kid->kga", jnp.conj(tt[k]), right_op[k], A[k], tt[k]
        )
        right_rhs[k - 1] = jnp.einsum(
            "abc,dc,ebd->ea", jnp.conj(tt[k]), right_rhs[k], b[k]
        )

    return (left_op, right_op), (left_rhs, right_rhs)


def _construct_micro_op(
    left_op, right_op, core_op: TTOCore, core_sol: TTCore
) -> Float[Array, "r0*row*r1 r0*col*r1"]:
    micro_op = jnp.einsum("abc,bdef,gfh->acdegh", left_op, core_op, right_op)
    micro_op = jnp.transpose(micro_op, (1, 2, 5, 0, 3, 4))  # acdegh->cdhaeg
    return jnp.reshape(
        micro_op,
        (
            core_sol.shape[0] * core_op.shape[1] * core_sol.shape[2],
            core_sol.shape[0] * core_op.shape[2] * core_sol.shape[2],
        ),
    )


def _construct_micro_rhs(
    left_rhs, right_rhs, core_rhs: TTCore
) -> Float[Array, "r0*row*r1"]:
    micro_rhs = jnp.einsum("ab,acd,de->bce", left_rhs, core_rhs, right_rhs)
    micro_rhs = jnp.ravel(micro_rhs)
    return micro_rhs


def als(
    A: TTO,
    b: TT,
    x0: TT,
    *,
    max_iters: int = 30,
    stagnation: float = 1e-5,
    l2_regularization: Optional[float] = None,
) -> tuple[int, float, TT]:
    d = len(x0)

    def _cond(val: tuple[int, float, TT]) -> bool:
        iters, stag, _ = val
        return (iters < max_iters) & (stag > stagnation)

    def _body(val: tuple[int, float, TT]) -> tuple[int, float, TT]:
        iters, stag, x0 = val
        guess = tt_orth_right(x0)  # right-orthonormalize components

        (left_op, right_op), (left_rhs, right_rhs) = _compute_left_right_stack(
            guess, A, b
        )
        # forward sweep
        for mu in range(0, d - 1):
            # build micro operator
            micro_op = _construct_micro_op(left_op[mu], right_op[mu], A[mu], b[mu])
            # build micro rhs
            micro_rhs = _construct_micro_rhs(left_rhs[mu], right_rhs[mu], b[mu])

            core = _solve_core(micro_op, micro_rhs, l2_regularization=l2_regularization)
            core = jnp.reshape(core, guess[mu].shape)
            # orthogonalize
            guess[mu], guess[mu + 1] = tt_shift_right(core, guess[mu + 1])
            # update 'left'
            left_op[mu + 1] = jnp.einsum(
                "abc,ade,bfdg,cfh->egh",
                left_op[mu],
                guess[mu],
                A[mu],
                jnp.conj(guess[mu]),
            )
            left_rhs[mu + 1] = jnp.einsum(
                "ab,acd,bce->de", left_rhs[mu], b[mu], jnp.conj(guess[mu])
            )

        # backward sweep
        for mu in range(d - 1, -1, -1):
            # build micro operator
            micro_op = _construct_micro_op(left_op[mu], right_op[mu], A[mu], b[mu])
            # build micro rhs
            micro_rhs = _construct_micro_rhs(left_rhs[mu], right_rhs[mu], b[mu])
            core = _solve_core(micro_op, micro_rhs, l2_regularization=l2_regularization)
            core = jnp.reshape(core, guess[mu].shape)
            if mu > 0:
                # orthogonalize
                guess[mu - 1], guess[mu] = tt_shift_left(guess[mu - 1], core)
                # update 'right'
                right_op[mu - 1] = jnp.einsum(
                    "abc,dec,gbie,kid->kga",
                    jnp.conj(guess[mu]),
                    right_op[mu],
                    A[mu],
                    guess[mu],
                )
                right_rhs[mu - 1] = jnp.einsum(
                    "abc,dc,ebd->ea", jnp.conj(guess[mu]), right_rhs[mu], b[mu]
                )
            else:
                guess[mu] = core

        iters += 1
        stag = _compute_stagnation(x0, guess)
        # jax.debug.print("iters={iters}, stag={stag}", iters=iters, stag=stag)

        return iters, stag, guess

    iters, stag, guess = jax.lax.while_loop(_cond, _body, init_val=(0, jnp.inf, x0))
    return iters, stag, guess

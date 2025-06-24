from functools import partial, reduce
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
    tt_ranks,
    tt_shift_left,
    tt_shift_right,
)
from ctt.tto import TTO, TTOCore


def outer_products(*vectors: Float[Array, "m"]) -> Float[Array, "..."]:
    return reduce(partial(jnp.tensordot, axes=0), vectors)


@jax.jit
def _compute_stagnation(A: TT, B: TT) -> float:
    EPS = 1e-8
    norm_a = tt_norm(A) ** 2
    norm_b = tt_norm(B) ** 2
    inner_product = tt_dot(A, B)
    return jnp.sqrt(abs((norm_a - 2.0 * inner_product + norm_b) / (norm_a + EPS)))


def als_linear_system(
    A: TTO,
    b: TT,
    x0: TT,
    *,
    max_iters: int = 30,
    stagnation: float = 1e-5,
    l2_regularization: Optional[float] = None,
) -> tuple[int, float, TT]:
    """
    Solve a linear system in Tensor Train (TT) format using Alternating Least Squares (ALS).

    This function approximately solves the linear system ``A x = b`` where `A` is a tensor train operator (TTO),
    `b` is a tensor train (TT), and `x` is the TT solution initialized from `x0`. The ALS method
    alternates between optimizing TT cores in a forward and backward sweep, updating local solutions
    until convergence or a maximum number of iterations is reached.

    Parameters
    ----------
    A : TTO
        Linear operator in Tensor Train Operator format.
    b : TT
        Right-hand side tensor in Tensor Train format.
    x0 : TT
        Initial guess for the solution in TT format.
    max_iters : int, optional
        Maximum number of ALS iterations to perform (default is 30).
    stagnation : float, optional
        Convergence threshold. Iterations stop when the change between successive iterates
        is below this value (default is 1e-5).
    l2_regularization : float or None, optional
        L2 regularization strength for the local least squares problems. If None, no
        regularization is applied (default is None).

    Returns
    -------
    iters : int
        Number of iterations performed.
    stag : float
        Final stagnation value (a measure of relative change between last two iterates).
    guess : TT
        Final solution tensor in TT format.

    Notes
    -----
    The method performs forward and backward ALS sweeps over the TT cores. During each sweep, local
    least-squares subproblems are solved to update each core. Right and left orthogonalization
    of TT components is applied to maintain numerical stability and efficiency.

    """

    d = len(x0)

    def _solve_core(
        A: Float[Array, "r0*row*r1 r0*col*r1"],
        b: Float[Array, "r0*row*r1"],
        *,
        l2_regularization: Optional[float] = None,
    ) -> Float[Array, "r0*row*r1"]:
        if l2_regularization is None:
            # return jnp.linalg.lstsq(A, b)[0]
            return jnp.linalg.solve(A, b)
        else:
            AT = A.T
            return jnp.linalg.solve(
                jnp.dot(AT, A) + l2_regularization * jnp.eye(A.shape[1]), jnp.dot(AT, b)
            )

    def _compute_left_right_stack(
        tt: TT, A: TTO, b: TT
    ) -> tuple[tuple[list, list], tuple[list, list]]:
        d = len(tt)
        left_op = [None] * d
        left_op[0] = jnp.ones((1, 1, 1))
        left_rhs = [None] * d
        left_rhs[0] = jnp.ones((1, 1))
        right_op = [None] * d
        right_op[-1] = jnp.ones((1, 1, 1))  # (rk, rk_op, rk)
        right_rhs = [None] * d
        right_rhs[-1] = jnp.ones((1, 1))  # (rk, rk)
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
            micro_op = _construct_micro_op(left_op[mu], right_op[mu], A[mu], guess[mu])
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
            )  # (rank[k], rank_op[k], rank[k])
            left_rhs[mu + 1] = jnp.einsum(
                "ab,acd,bce->de", left_rhs[mu], b[mu], jnp.conj(guess[mu])
            )  # (rank_sol[k], rank[k])

        # backward sweep
        for mu in range(d - 1, -1, -1):
            # build micro operator
            micro_op = _construct_micro_op(left_op[mu], right_op[mu], A[mu], guess[mu])
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
                )  # (rank[k], rank_op[k], rank[k])
                right_rhs[mu - 1] = jnp.einsum(
                    "abc,dc,ebd->ea", jnp.conj(guess[mu]), right_rhs[mu], b[mu]
                )  # (rank_sol[k], rank[k])
            else:
                guess[mu] = core

        iters += 1
        stag = _compute_stagnation(x0, guess)
        # jax.debug.print("iters={iters}, stag={stag}", iters=iters, stag=stag)

        return iters, stag, guess

    iters, stag, guess = jax.lax.while_loop(_cond, _body, init_val=(0, jnp.inf, x0))
    return iters, stag, guess


def als_ls_vector(
    A: list[Float[Array, "B m"]],
    b: Float[Array, "B d_o"],
    x0: TT,
    weights: Optional[Float[Array, "B"]] = None,
    *,
    max_iters: int = 30,
    stagnation: float = 1e-5,
    l2_regularization: Optional[float] = None,
) -> tuple[int, float, TT]:
    """
    Solve a least-squares problem using Alternating Least Squares (ALS) for vector-valued tensor trains.

    Each batch input of `A` is assumed to be of the form :math:`(\psi_b \otimes \Phi_b)`, where :math:`\psi_b \in \mathbb{R}^{d_o \times d}`
    and :math:`\Phi_b \in \mathbb{R}^{m \times \dots \times m}` for each batch entry `b`.
    The tensor `x0` represents the initial guess in the form of a tensor train with shape (d, m, ..., m).
    The goal is to solve for the optimal tensor `x` in TT format that minimizes the regularized least-squares loss.

    Parameters
    ----------
    A : list of Float[Array, "B m"]
        A list of tensor factors, where `A[0]` has shape (B, d_o, d) and each `A[i]` for i > 0 has shape (B, m).
        The list should have length `order + 1`, where `order = len(x0)`.

    b : Float[Array, "B d_o"]
        Target output vectors of shape (B, d_o), where B is the batch size and d_o is the output dimension.

    x0 : TT
        Initial guess for the tensor in TT format. Should be a list of `order` cores with appropriate ranks and shapes
        to represent a tensor of shape (d, m, ..., m), with the first core shape (d, m, r1).

    weights : Optional[Float[Array, "B"]], optional
        Optional sample weights of shape (B,). If None, all samples are equally weighted.

    max_iters : int, default=30
        Maximum number of ALS sweeps (forward and backward passes).

    stagnation : float, default=1e-5
        Convergence criterion based on relative change in the solution across iterations.

    l2_regularization : Optional[float], optional
        If provided, L2 regularization is applied to the least-squares solution for each TT core.

    Returns
    -------
    iters : int
        Number of ALS iterations performed before convergence.

    stagnation : float
        Final stagnation value (relative difference between two successive tensor estimates).

    guess : TT
        The optimized tensor in TT format that approximates the least-squares solution.

    Notes
    -----
    - The ALS approach alternates between optimizing each TT core while keeping the others fixed.
    - Each subproblem is a linear least-squares problem, optionally regularized.
    - The function assumes the TT representation is right-orthogonalized before each forward sweep.
    - The input `A` must be constructed carefully so that the Kronecker structure aligns with the TT decomposition.
    """
    # one batch entry of A is of the form Psi ⊗ Phi, where Psi \in R^{d_o x d} and Phi \in R^{m x ... x m}
    # NOTE: x0 should be a tensor of shape (d, m, ..., m), i.e. it should be a vector-valued tensor
    # so the TT cores should be of the form (d, m, r1) (r1, m, r2) ... (rd-1, m, 1)

    assert A[0].ndim == 3

    B, d_o, d = A[0].shape
    order = len(x0)

    assert b.shape == (B, d_o)
    assert x0[0].shape[0] == d, (
        "The first rank of 'x0' should match the output dimension."
    )
    assert len(A) == order + 1

    if weights is None:
        weights = jnp.ones((B,))
    sqrt_weights = jnp.sqrt(weights)

    # we flatten sqrt(w)*b into the shape (B*d_o,)
    weighted_b = jnp.ravel(sqrt_weights[:, None] * b)

    def _solve_core(
        A: Float[Array, "B p"],
        b: Float[Array, "B"],
        *,
        l2_regularization: Optional[float] = None,
    ) -> Float[Array, "p"]:
        if l2_regularization is None:
            sol = jnp.linalg.lstsq(A, b)[0]
        else:
            AT = A.T
            sol = jnp.linalg.solve(
                jnp.dot(AT, A) + l2_regularization * jnp.eye(A.shape[1]), jnp.dot(AT, b)
            )
        return sol

    def _compute_left_right_stack(
        tt: TT, A: list[Float[Array, "B m"]]
    ) -> tuple[list[Float[Array, "B r"]], list[Float[Array, "B r"]]]:
        B = A[0].shape[0]
        ranks = tt_ranks(tt)
        left = [None] * len(tt)
        left[0] = A[0]  # (B, d_o, d)
        right = [jnp.ones((B, r)) for r in ranks[1:]]
        for k in range(len(tt) - 1, 0, -1):
            right[k - 1] = jnp.einsum(
                "bk,ijk,bj->bi", right[k], tt[k], A[k + 1]
            )  # NOTE: we access the `k+1`-th index because A has an additional dimension
        return left, right

    def _cond(val: tuple[int, float, TT]) -> bool:
        iters, stag, _ = val
        return (iters < max_iters) & (stag > stagnation)

    def _body(val: tuple[int, float, TT]) -> tuple[int, float, TT]:
        iters, stag, x0 = val
        guess = tt_orth_right(x0)  # right-orthonormalize components

        left, right = _compute_left_right_stack(guess, A)
        # forward sweep
        for mu in range(0, order - 1):
            # solve for the new core
            L = left[mu]  # (B, d_o, r_{mu-1})
            M = A[
                mu + 1
            ]  # NOTE: we access the `mu+1`-th index because A has an additional dimension. Shape: (B, m_\mu)
            R = right[mu]  # (B, r_\mu)
            features = jax.vmap(outer_products)(L, M, R).reshape(
                (B, d_o, -1)
            )  # (B, d_o, r_{mu-1} x m_\mu x r_\mu)
            weighted_features = sqrt_weights[:, None, None] * features
            weighted_features = jnp.reshape(
                weighted_features, (B * d_o, -1)
            )  # (B*d_o, r_{mu-1} x m_\mu x r_\mu)
            core = _solve_core(
                weighted_features,
                weighted_b,
                l2_regularization=l2_regularization,
            )
            core = jnp.reshape(core, guess[mu].shape)
            # orthogonalize
            guess[mu], guess[mu + 1] = tt_shift_right(core, guess[mu + 1])
            # update 'left'
            left[mu + 1] = jnp.einsum("bdi,ijk,bj->bdk", L, guess[mu], M)

        # backward sweep
        for mu in range(d - 1, -1, -1):
            # solve for the new core
            L = left[mu]
            M = A[mu + 1]
            R = right[mu]
            features = jax.vmap(outer_products)(L, M, R).reshape(
                (B, d_o, -1)
            )  # (B, d_o, r_{mu-1} x m_\mu x r_\mu)
            weighted_features = sqrt_weights[:, None, None] * features
            weighted_features = jnp.reshape(
                weighted_features, (B * d_o, -1)
            )  # (B*d_o, r_{mu-1} x m_\mu x r_\mu)
            core = _solve_core(
                weighted_features,
                weighted_b,
                l2_regularization=l2_regularization,
            )
            core = jnp.reshape(core, guess[mu].shape)
            # orthogonalize
            if mu > 0:
                guess[mu - 1], guess[mu] = tt_shift_left(guess[mu - 1], core)
                # update 'right'
                right[mu - 1] = jnp.einsum("bk,ijk,bj->bi", R, guess[mu], M)
            else:
                guess[mu] = core

        iters += 1
        stag = _compute_stagnation(x0, guess)
        # jax.debug.print("iters={iters}, stag={stag}", iters=iters, stag=stag)

        return iters, stag, guess

    iters, stag, guess = jax.lax.while_loop(_cond, _body, init_val=(0, jnp.inf, x0))
    return iters, stag, guess

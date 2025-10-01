from functools import partial
from typing import Any, Callable, Literal, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray
from scikit_tt import TT as sk_TT
from scikit_tt.solvers.sle import mals
from scikit_tt.tensor_train import residual_error

from ctt.bases import Basis
from ctt.model import _eval_bases, eval_ftt
from ctt.solvers import als_linear_system
from ctt.tt import (
    CP,
    TT,
    cp_to_tt_truncate,
    tt_add,
    tt_dims,
    tt_dot,
    tt_mul_scalar,
    tt_orth_right,
    tt_randn,
    tt_ranks,
    tt_truncate,
    validate_ranks,
)
from ctt.tto import cpo_to_tto_truncate


def ctt_natural_grad(
    bases: list[Basis],
    lift: Callable[[Float[Array, "d"]], Float[Array, "d_lift"]],
    retraction: Callable[[Float[Array, "d_lift"]], Float[Array, "d_o"]],
    grad_loss: Callable[..., Float[Array, "B d_o"]],
):
    r"""Returns a function that computes the natural gradients,
    and a function that computes the functional gradient of the loss.

    Parameters
    ----------
    bases : list[Basis]
        Bases that define the CTT.
    lift : Callable
        Lift operator that defines the CTT.
    retraction : Callable
        Retraction operator that defines the CTT.
    grad_loss : Callable
        Functional gradient of the loss e.g. if :math:`\mathcal{L}(u) = \frac{1}{2} \|u-v\|^2`,
        then :math:`\mathcal{L}(u)(x) = u(x)-v(x)`.

    Returns
    -------
    tuple[Callable, Callable]
        - the first function computes the natural gradient at some parameters `params`, using `points`
        to estimate the integrals.
        - the second function computes the functional gradient :math:`\nabla_{\theta} \mathcal{L}(u)`.
    """

    def _compute_jacobians(
        params: list[TT], x: Float[Array, "d"]
    ) -> tuple[Float[Array, "L+1 d_o"], Float[Array, "L d_o d_lift"]]:
        L = len(params)
        # lift the point
        X = lift(x)  # (d_lift,)

        # computes the intermediates values u_k(x) = f_k o ... o f_1 o L(x)
        # def _fwd(
        #     carry: Float[Array, "d_lift"], k: int
        # ) -> tuple[Float[Array, "d_lift"], Float[Array, "d_lift"]]:
        #     u = jax.lax.switch(k, [lambda u=u: u for u in params])
        #     y = carry + _ftt(bases, carry, u)
        #     return y, y

        # _, dynamics = jax.lax.scan(_fwd, X, jnp.arange(len(params)))  # (L, d_lift)
        # dynamics = jnp.concatenate([X[None], dynamics], axis=0)  # (L+1, d_lift)
        def _layer(x: Float[Array, "d"], params: TT) -> Float[Array, "d"]:
            return x + eval_ftt(bases, x, params)

        dynamics = [X]
        h = X
        for k in range(len(params)):
            h = _layer(h, params[k])
            dynamics.append(h)
        dynamics = jnp.stack(dynamics, axis=0)

        # computes the Jacobian du/du_k
        jacobians = []
        for k in range(1, len(params) + 1):

            def u_k(
                y: Float[Array, "d_lift"], tts: list[TT] = params[k:]
            ) -> Float[Array, "d_o"]:
                h = y
                for u in tts:
                    h = h + eval_ftt(bases, h, u)
                return retraction(h)

            J = jax.jacrev(u_k)(dynamics[k])  # (d_o, d_lift)
            jacobians.append(J)

        # we have that `jacobians[k] = du/dx^{k+1}`
        jacobians = jnp.stack(jacobians, axis=0)  # (L, d_o, d_lift)
        return dynamics, jacobians

    def _compute_jacobians_fast(
        params: list[TT], x: Float[Array, "d"]
    ) -> tuple[Float[Array, "L+1 d_o"], Float[Array, "L d_o d_lift"]]:
        L = len(params)
        X = lift(x)

        def _layer(x: Float[Array, "d"], params: TT) -> Float[Array, "d"]:
            return x + eval_ftt(bases, x, params)

        # forward
        dynamics = [X]
        h = X
        for k in range(L):
            h = _layer(h, params[k])
            dynamics.append(h)
        dynamics = jnp.stack(dynamics, axis=0)

        # computes the Jacobian du/du_k
        jacobians = []
        for k in range(1, L + 1):

            def u_k(
                y: Float[Array, "d_lift"], tts: list[TT] = params[k:]
            ) -> Float[Array, "d_o"]:
                h = y
                for u in tts:
                    h = h + eval_ftt(bases, h, u)
                return retraction(h)

            ans, vjp_py = jax.vjp(
                u_k, dynamics[k], params
            )  # dynamics[k] or dynamics[k-1]
            g = vjp_py(jnp.ones_like(ans))[0]
            g = jnp.atleast_2d(g)
            jacobians.append(g)

        # we have that `jacobians[k] = du/dx^{k+1}`
        jacobians = jnp.stack(jacobians, axis=0)  # (L, d_o, d_lift)
        return dynamics, jacobians

    def _compute_functional_grad(
        params: list[TT],
        X: Float[Array, "B d"],
        weights: Optional[Float[Array, "B"]] = None,
        intermediate_values: Optional[Float[Array, "B L+1 d_lift"]] = None,
        jacobians: Optional[Float[Array, "B L d_o d_lift"]] = None,
        **extra_args: Any,
    ) -> list[CP]:
        if intermediate_values is None or jacobians is None:
            # intermediate_values, jacobians = jax.vmap(
            #     _compute_jacobians, in_axes=(None, 0)
            # )(params, X)
            intermediate_values, jacobians = jax.vmap(
                _compute_jacobians_fast, in_axes=(None, 0)
            )(params, X)

        B, d = X.shape
        L = len(params)

        if weights is None:
            weights = jnp.ones((B,))

        # normalize the weights
        weights /= jnp.sum(weights)

        # compute ∇L(u)(x) for each x
        last_layer = intermediate_values[:, -1, :]  # (B, d_lift)
        outputs = jax.vmap(retraction)(last_layer)  # (B, d_o)
        grad_loss_ = partial(grad_loss, **extra_args)
        grad_losses = grad_loss_(outputs)  # (B, d_o)

        def _compute_tensor_grad(
            tensor: TT, jac: Float[Array, "B d_o d_lift"], u_k: Float[Array, "B d_lift"]
        ) -> CP:
            # we have ∇_{\theta_k} L(u) = \int <∇L(u)(x), du/d\theta_k(x)> d\mu(x) \in R^{d_lift x n x ... x n}
            # so we have first to contract ∇L(u)(x) with the leg of du/d\theta_k that corresponds to `d_o`
            first_leg = jax.vmap(jnp.matmul)(grad_losses, jac)  # (B, d_lift)

            # now, <∇L(u)(x), du/d\theta_k(x)> is of the form
            # A_k(x) ⊗ \Phi(u_{k-1}(x))
            # which is in the CP format
            # the integral is discretized using Monte-Carlo, therefore the ranks would be upper bounded by `B`.
            phi = jax.vmap(_eval_bases, in_axes=(None, 0))(bases, u_k)  # (B, m)*d_lift
            factors = [weights[:, None] * first_leg] + phi  # (d_lift+1)-order tensor

            # we convert the tensor in the CP format to a tensor in the TT format
            # the ranks of the TT are however high (`B`), so we have to do truncation or rounding.
            # the best method would be `rounding`, however, due to JAX, the output ranks will not be predictable.
            # therefore, we decide to use `truncation`, where the ranks are the same as `tensor`.
            # ranks = tt_ranks(tensor)
            # ranks.insert(0, 1)
            # # ranks = [1] + ranks[1:]
            # grad = cp_to_tt_truncate(factors, ranks)
            return factors

        grads = [
            _compute_tensor_grad(
                params[i], jacobians[:, i, :, :], intermediate_values[:, i, :]
            )
            for i in range(L)
        ]
        return grads

    def _compute_natural_grad(
        key: PRNGKeyArray,
        params: list[TT],
        X: Float[Array, "B d"],
        weights: Optional[Float[Array, "B"]] = None,
        l2_regularization: Optional[float] = None,
        max_iters: int = 100,
        stagnation: float = 1e-8,
        **extra_args: dict[str, Any],
    ) -> list[TT]:
        B, d = X.shape
        L = len(params)
        if weights is None:
            weights = jnp.ones((B,))
        weights = weights / jnp.sum(weights)

        # intermediate_values, jacobians = jax.vmap(
        #     _compute_jacobians, in_axes=(None, 0)
        # )(params, X)
        intermediate_values, jacobians = jax.vmap(
            _compute_jacobians_fast, in_axes=(None, 0)
        )(params, X)
        outputs = jax.vmap(retraction)(intermediate_values[:, -1, :])  # (B, d_o)
        grad_loss_ = partial(grad_loss, **extra_args)
        grad_losses = grad_loss_(outputs)  # (B, d_o)

        def _assemble_normal_equation(
            tt: TT,
            jac: Float[Array, "B d_o d_lift"],
            u_k: Float[Array, "B d_lift"],
            op_ranks: list[int],
        ) -> tuple[TT, TT]:
            # first, compute the Gram matrix `G`
            def _compute_rank_one_tensor(
                w: float, J: Float[Array, "d_o d_lift"], h: Float[Array, "d_lift"]
            ) -> CP:
                # the first leg is J.T @ J
                first_leg = J.T @ J  # (d_lift, d_lift)
                # the second leg is (\phi^1(h_1) \phi^1(h_1)^T) ⊗ ... ⊗ (\phi^d(h_d)\phi^d(h_d)^T)
                phi = _eval_bases(bases, h)  # (m,)*d_lift
                tmp = jax.tree.map(jnp.outer, phi, phi)  # (m, m)*d_lift
                return [w * first_leg] + tmp

            factors = jax.vmap(_compute_rank_one_tensor)(weights, jac, u_k)
            # the Gram operator is converted to the TT format using truncation
            gram = cpo_to_tto_truncate(factors, op_ranks)

            # we have ∇_{\theta_k} L(u) = \int <∇L(u)(x), du/d\theta_k(x)> d\mu(x) \in R^{d_lift x n x ... x n}
            # so we have first to contract ∇L(u)(x) with the leg of du/d\theta_k that corresponds to `d_o`
            first_leg = jax.vmap(jnp.matmul)(grad_losses, jac)  # (B, d_lift)

            # now, <∇L(u)(x), du/d\theta_k(x)> is of the form
            # A_k(x) ⊗ \Phi(u_{k-1}(x))
            # which is in the CP format
            # the integral is discretized using Monte-Carlo, therefore the ranks would be upper bounded by `B`.
            phi = jax.vmap(_eval_bases, in_axes=(None, 0))(bases, u_k)  # (B, m)*d_lift
            factors = [weights[:, None] * first_leg] + phi  # (d_lift+1)-order tensor
            # we convert the tensor in the CP format to a tensor in the TT format
            # the ranks of the TT are however high (`B`), so we have to do truncation or rounding.
            # the best method would be `rounding`, however, due to JAX, the output ranks will not be predictable.
            # therefore, we decide to use `truncation`, where the ranks are the same as `tensor`.
            ranks = tt_ranks(tt)
            ranks.insert(0, 1)
            # rhs = cp_to_tt_truncate(factors, ranks)
            rhs = cp_to_tt_truncate(factors, op_ranks)

            return gram, rhs

        def _natural_grad(
            subkey: PRNGKeyArray,
            tt: TT,
            jac: Float[Array, "B d_o d_lift"],
            u_k: Float[Array, "B d_lift"],
            grad: Float[Array, "B d_o"],
        ) -> TT:
            # du_dthetak = [jac] + jax.vmap(_eval_bases, in_axes=(None, 0))(bases, u_k)
            ranks = tt_ranks(tt)
            dims = tt_dims(tt)
            # we might need higher ranks for converging
            # op_ranks = [ranks[0]] + list(map(lambda x: x**2, ranks[1:]))
            op_ranks = [1] + [ranks[0]] + list(map(lambda x: x**2, ranks[1:]))
            # op_ranks = [1] + ranks
            # op_ranks = validate_ranks(dims, op_ranks)
            op_ranks = validate_ranks([ranks[0]] + dims, op_ranks)
            init_tt = tt_randn(subkey, [ranks[0]] + dims, op_ranks)

            # solve the least-squares problem \sum_j ||<d u^j/d theta_k, d> - ∇L(u)(x)_j||^2
            # NOTE: in our case, the natural metric g_\theta is exactly the metric on H,
            # because D^2H = Id
            # iterations, stag, residual, dir = als_ls_vector(
            #     du_dthetak,
            #     grad,
            #     init_tt,
            #     weights=weights,
            #     max_iters=max_iters,
            #     stagnation=stagnation,
            #     l2_regularization=l2_regularization,
            #     # cutoff=1e-5,
            # )

            # or instead solve the normal equation G @ d = b
            # compute the tensor operator that corresponds to the Gram matrix
            gram, rhs = _assemble_normal_equation(tt, jac, u_k, op_ranks)

            # TODO: change to solving least-squares problem
            # solve the linear system G @ d = b
            iterations, stag, dir = als_linear_system(
                gram,
                rhs,
                init_tt,
                max_iters=max_iters,
                stagnation=stagnation,
                l2_regularization=l2_regularization,
            )
            # NOTE: we are using `scikit_tt` to check if our ALS implementation is correct
            gram_sk = sk_TT(gram)
            rhs_sk = sk_TT([core[:, :, None, :] for core in rhs])
            init_sk = sk_TT([core[:, :, None, :] for core in init_tt])
            jax.debug.print(
                "Our ALS residual: {residual:.3e}",
                residual=residual_error(
                    gram_sk, sk_TT([core[:, :, None, :] for core in dir]), rhs_sk
                ),
            )
            dir = mals(gram_sk, init_sk, rhs_sk, repeats=1, threshold=1e-8, solver="lu")
            jax.debug.print(
                "MALS residual: {residual:.3e}",
                residual=residual_error(gram_sk, dir, rhs_sk),
            )
            # `dir` is of order d_lift+1, we contract the first two cores to have a vector output TT of order d_lift
            dir = [core[:, :, 0, :] for core in dir.cores]
            dir = [jnp.einsum("ijk,kmn->jmn", dir[0], dir[1])] + dir[2:]
            return dir

        subkeys = random.split(key, num=L)
        natural_grads = [
            _natural_grad(
                subkeys[i],
                params[i],
                jacobians[:, i, :, :],
                intermediate_values[:, i, :],
                grad_losses,
            )
            for i in range(L)
        ]
        return natural_grads

    return _compute_natural_grad, _compute_functional_grad


def ctt_apply_updates(x: list[TT], y: list[TT]) -> list[TT]:
    """Apply updates `y` to the TTs `x`.

    Parameters
    ----------
    x : list[TT]
        TTs to be updated.
    y : list[TT]
        Updates to apply.

    Returns
    -------
    list[TT]
        Updated TTs.
    """

    def _update(x: TT, y: TT) -> TT:
        ranks = tt_ranks(x)
        u = tt_add(x, y)
        return tt_truncate(u, ranks)

    # fused TT-add+truncate
    def _update_fused(a: TT, b: TT) -> TT:
        # ranks to preserve from the original
        target_ranks = tt_ranks(a)
        target_ranks = validate_ranks(tt_dims(a), target_ranks)

        # orthogonalize to the right for stable SVD truncation
        cores = tt_orth_right(a)
        updates = tt_orth_right(b)

        new_cores = []

        # first core
        core0 = jnp.concatenate((cores[0], updates[0]), axis=2)
        new_cores.append(core0)

        # middle cores: block structure addition + local svd truncation
        cur = core0
        for pos, (c1, c2) in enumerate(zip(cores[1:-1], updates[1:-1]), start=1):
            # build block core inline
            upper = jnp.concatenate(
                (c1, jnp.zeros((c1.shape[0], c2.shape[1], c2.shape[2]))), axis=2
            )
            lower = jnp.concatenate(
                (jnp.zeros((c2.shape[0], c1.shape[1], c1.shape[2])), c2), axis=2
            )
            block = jnp.concatenate((upper, lower), axis=0)

            # truncate this block
            shape = cur.shape
            u, s, vh = jnp.linalg.svd(
                cur.reshape(shape[0] * shape[1], shape[2]), full_matrices=False
            )
            r_new = target_ranks[pos]
            u, s, vh = u[:, :r_new], s[:r_new], vh[:r_new, :]
            new_cores[-1] = u.reshape((shape[0], shape[1], r_new))
            cur = jnp.einsum("i,ir,rkl->ikl", s, vh, block)
            new_cores.append(cur)
        # last core
        last = jnp.concatenate((cores[-1], updates[-1]), axis=0)
        new_cores.append(last)
        return new_cores

    return list(map(_update, x, y))
    # return list(map(_update_fused, x, y))


class LinesearchState(NamedTuple):
    learning_rate: Union[float, Array]
    value: Union[float, Array]


def ctt_linesearch(
    max_backtracking_steps: int,
    condition: Literal["armijo", "goldstein", "strong-wolfe", "wolfe"] = "armijo",
    slope_rtol: float = 1e-4,
    curvature_rtol: float = 0.9,
    decrease_factor: float = 0.8,
    increase_factor: float = 1.5,
    max_learning_rate: float = 1.0,
    atol: float = 0.0,
    rtol: float = 0.0,
):
    if condition not in ["armijo", "goldstein", "strong-wolfe", "wolfe"]:
        raise ValueError(
            "'condition' should be one of "
            "'armijo', 'goldstein', 'strong-wolfe', 'wolfe'."
        )

    def _compute_slope(x: list[TT], y: list[TT]) -> float:
        return sum(list(map(tt_dot, x, y)))

    def init_fn(params: list[TT]) -> LinesearchState:
        return LinesearchState(
            learning_rate=jnp.asarray(1.0), value=jnp.asarray(jnp.inf)
        )

    def update_fn(
        updates: list[TT],
        state: LinesearchState,
        params: list[TT],
        slope_fn: Callable[[list[TT], list[TT]], float],
        *,
        value: float,
        value_fn: Callable[..., Union[Array, float]],
        **extra_args: dict[str, Any],
    ) -> tuple[list[TT], LinesearchState]:
        # TODO: we should instead for a function that computes the slope <d_k, \nabla f(x_k)>
        # because the way we are doing right now requires the gradient to be in the TT format
        # whereas we have shown that it is in the CP format.
        # slope = _compute_slope(updates, grad)
        slope = slope_fn(
            updates, params
        )  # computes <d_k, \nabla f(x_k)>, where d_k is the direction

        class _LoopState(NamedTuple):
            learning_rate: Union[float, Array]
            new_value: Union[float, Array]
            decrease_error: Union[float, Array]
            iter_num: int

        def cond_fn(search_state: _LoopState):
            decrease_error = search_state.decrease_error
            iter_num = search_state.iter_num
            return (decrease_error > atol) & (iter_num <= max_backtracking_steps)

        def body_fn(search_state: _LoopState) -> _LoopState:
            learning_rate = search_state.learning_rate
            iter_num = search_state.iter_num
            # We start decreasing the learning rate after the first iteration
            # and up until the criterion is satisfied.
            learning_rate = jnp.where(
                iter_num > 0, decrease_factor * learning_rate, learning_rate
            )
            new_updates = list(map(partial(tt_mul_scalar, val=learning_rate), updates))
            new_params = ctt_apply_updates(params, new_updates)

            value_fn_ = partial(value_fn, **extra_args)
            # grad_fn_ = partial(grad_fn, **extra_args)

            new_value = value_fn_(new_params)
            # new_grads = None
            if condition == "strong-wolfe" or condition == "wolfe":
                # new_grads = grad_fn_(new_params)
                # new_slope = _compute_slope(updates, new_grads)
                new_slope = slope_fn(
                    updates, new_params
                )  # computes <d_k, \nabla f(x_k + \alpha_k d_k)>

            # Armijo condition (upper bound on admissible step size)
            # f(x+alpha*d) <= (1+delta)f(x) + c_1 alpha <d, df(x)> + eps
            error_cond1 = jnp.maximum(
                0.0,
                new_value - ((1.0 + rtol) * value + slope_rtol * learning_rate * slope),
            )
            error = error_cond1

            if condition == "armijo":
                pass
            elif condition == "strong-wolfe":
                # |<df(x+alpha*d), d>| <= c_2 |<d, df(x)>| + eps
                error_cond2 = jnp.maximum(
                    0.0, jnp.abs(new_slope) - curvature_rtol * jnp.abs(slope)
                )
                error = jnp.maximum(error_cond1, error_cond2)
            elif condition == "wolfe":
                # <d, df(x+alpha*d)> >= c_2 <d, df(x)>
                error_cond2 = jnp.maximum(0.0, curvature_rtol * slope - new_slope)
                error = jnp.maximum(error_cond1, error_cond2)
            elif condition == "goldstein":
                # f(x+alpha*d) >= f(x) + (1-c_1)*alpha*<d, df(x)>
                error_cond2 = jnp.maximum(
                    0.0, value + (1.0 - slope_rtol) * learning_rate * slope - new_value
                )
                error = jnp.maximum(error_cond1, error_cond2)

            search_state = _LoopState(
                learning_rate=learning_rate,
                new_value=new_value,
                decrease_error=error,
                iter_num=iter_num + 1,
            )
            return search_state

        # We start with a guess candidate learning rate that may be larger than
        # the current one but no larger than the maximum one.
        # learning_rate = jnp.minimum(
        #     increase_factor * state.learning_rate, max_learning_rate
        # )
        learning_rate = 1.0
        search_state = _LoopState(
            learning_rate=learning_rate,
            new_value=value,
            decrease_error=jnp.array(jnp.inf),
            iter_num=0,
        )
        search_state = jax.lax.while_loop(cond_fn, body_fn, search_state)

        new_value = search_state.new_value
        # If the decrease error is infinite, we avoid making any step (which would
        # result in nan or infinite values): we set the learning rate to 0.
        new_learning_rate = jnp.where(
            jnp.isinf(search_state.decrease_error), 0.0, search_state.learning_rate
        )
        # At the end, we just scale the updates with the learning rate found.
        new_updates = list(map(partial(tt_mul_scalar, val=new_learning_rate), updates))
        new_state = LinesearchState(learning_rate=new_learning_rate, value=new_value)
        return new_updates, new_state

    return init_fn, update_fn

from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Union,
)

import jax
import jax.numpy as jnp
import jax.scipy.optimize
from jaxtyping import Array, Float

from ctt.bases import Basis
from ctt.solvers import als_linear_system
from ctt.tt import (
    TT,
    cp_to_tt_truncate,
    tt_add,
    tt_dims,
    tt_matvec,
    tt_mul_scalar,
    tt_orth_right,
    tt_ranks,
    tt_truncate,
    tt_zeros_like,
    validate_ranks,
)
from ctt.tto import TTO, cpo_to_tto_truncate

__all__ = ["Control", "discrete_pmp", "batch_pmp"]

ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
Control = ArrayTree


class PMP(Protocol):
    def __call__(
        self, controls: list[Control], *args: Any, **kwargs: Any
    ) -> list[Control]: ...


def discrete_pmp(
    regularization: Callable[[Float[Array, "N"], Control], float],
    terminal_cost: Callable[[Float[Array, "N"]], float],
    transition: Callable[[Float[Array, "N"], Control], Float[Array, "N"]],
    min_hamiltonian: Callable[[Float[Array, "N"], Float[Array, "N"], Control], Control],
    x0: Float[Array, "N"],
    num_steps: int,
) -> PMP:
    def _solve_state(controls: list[Control]) -> Float[Array, "steps N"]:
        def _body_fn(xk, idx):
            control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
            next_state = transition(xk, control)
            return next_state, next_state

        _, states = jax.lax.scan(_body_fn, x0, jnp.arange(0, num_steps))
        states = jnp.concatenate([x0[None], states], axis=0)
        return states

    def _solve_costate(
        controls: list[Control], states: Float[Array, "steps N"]
    ) -> Float[Array, "steps"]:
        def _body_fn(costate, idx):
            control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
            xk = states[idx, :]  # x_k
            previous_costate = (
                jax.grad(regularization, argnums=0)(xk, control)
                # + jax.jacfwd(transition, argnums=0)(xk, control) @ costatae
                # + jax.jacfwd(transition, argnums=0)(xk, control) @ costatae
                + jax.jvp(lambda x: transition(x, control), (xk,), (costate,))[1]
            )
            return previous_costate, previous_costate

        terminal_point = jax.grad(terminal_cost)(states[-1, :])
        _, costates = jax.lax.scan(
            _body_fn, terminal_point, jnp.arange(0, num_steps), reverse=True
        )
        costates = jnp.concatenate([costates, terminal_point[None]], axis=0)
        return costates

    def _solve_pmp(controls: list[Control]) -> list[Control]:
        states = _solve_state(controls)
        costates = _solve_costate(controls, states)
        new_controls = [
            min_hamiltonian(states[i, :], costates[i + 1, :], controls[i])
            for i in range(num_steps)
        ]
        return new_controls

    return _solve_pmp


def batch_pmp(
    regularization: Callable[[Float[Array, "B d"], Control], float],
    terminal_cost: Callable[[Float[Array, "B d"]], float],
    transition: Callable[[Float[Array, "B d"], Control], Float[Array, "B d"]],
    min_hamiltonian: Callable[
        [Float[Array, "B d"], Float[Array, "B d"], Control], Control
    ],
    x0: Float[Array, "B d"],
    num_steps: int,
) -> PMP:
    B, d = x0.shape

    def _regularization(xk: Float[Array, "N"], control: Control) -> float:
        xk = jnp.reshape(xk, (B, d))
        val = regularization(xk, control)
        return val

    def _terminal_cost(xT: Float[Array, "N"]) -> float:
        xT = jnp.reshape(xT, (B, d))
        val = terminal_cost(xT)
        return val

    def _transition(xk: Float[Array, "N"], control: Control) -> Float[Array, "N"]:
        xk = jnp.reshape(xk, (B, d))
        val = transition(xk, control)
        val = jnp.ravel(val)
        return val

    def _min_hamiltoninan(
        state: Float[Array, "N"], costate: Float[Array, "N"], control: Control
    ) -> Control:
        state = jnp.reshape(state, (B, d))
        costate = jnp.reshape(costate, (B, d))
        val = min_hamiltonian(state, costate, control)
        return val

    return discrete_pmp(
        regularization=_regularization,
        terminal_cost=_terminal_cost,
        transition=_transition,
        min_hamiltonian=_min_hamiltoninan,
        x0=jnp.ravel(x0),
        num_steps=num_steps,
    )


def mini_batch_pmp(
    regularization: Callable[[Float[Array, "d_lift"], Control], float],
    terminal_cost: Callable[
        [Float[Array, "d_lift"], Optional[Float[Array, "d_o"]]], float
    ],
    transition: Callable[[Float[Array, "d_lift"], Control], Float[Array, "d_lift"]],
    min_hamiltonian: Callable[
        [Float[Array, "B d_lift"], Float[Array, "B d_lift"], Control, int], Control
    ],
    num_steps: int,
) -> PMP:
    def _solve_state(
        controls: list[Control], x0: Float[Array, "B d_lift"]
    ) -> Float[Array, "steps B d_lift"]:
        def _body_fn(xk, idx):
            control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
            next_state = jax.vmap(transition, in_axes=(0, None))(xk, control)
            return next_state, next_state

        _, states = jax.lax.scan(_body_fn, x0, jnp.arange(0, num_steps))
        states = jnp.concatenate([x0[None], states], axis=0)
        return states

    def _hamiltonian(
        state: Float[Array, "d_lift"],
        costate: Float[Array, "d_lift"],
        control: Control,
    ) -> float:
        return regularization(state, control) + jnp.dot(
            costate, transition(state, control)
        )

    def _solve_costate(
        controls: list[Control],
        states: Float[Array, "steps B d_lift"],
        yN: Optional[Float[Array, "B d_o"]] = None,
    ) -> Float[Array, "steps"]:
        def _body_fn(costate, idx):
            control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
            xk = states[idx, ...]  # x_k
            # previous_costate = (
            #     jax.grad(regularization, argnums=0)(xk, control)
            #     # + jax.jacfwd(transition, argnums=0)(xk, control) @ costatae
            #     # + jax.jacfwd(transition, argnums=0)(xk, control) @ costatae
            #     + jax.jvp(lambda x: transition(x, control), (xk,), (costate,))[1]
            # )
            previous_costate = jax.vmap(jax.grad(_hamiltonian), in_axes=(0, 0, None))(
                xk, costate, control
            )
            return previous_costate, previous_costate

        terminal_point = jax.vmap(jax.grad(terminal_cost))(states[-1, :], yN)
        _, costates = jax.lax.scan(
            _body_fn, terminal_point, jnp.arange(0, num_steps), reverse=True
        )
        costates = jnp.concatenate([costates, terminal_point[None]], axis=0)
        return costates

    def _solve_pmp(
        controls: list[Control],
        x0: Float[Array, "B d_lift"],
        yN: Optional[Float[Array, "B d_o"]] = None,
        k: Optional[int] = None,
    ) -> list[Control]:
        states = _solve_state(controls, x0)
        costates = _solve_costate(controls, states, yN)
        new_controls = [
            min_hamiltonian(states[i, :], costates[i + 1, :], controls[i], k)
            for i in range(num_steps)
        ]
        return new_controls

    return _solve_pmp


class MSAState(NamedTuple):
    iterations: int


def natural_msa(
    R: float,
    terminal_cost: Callable[
        [Float[Array, "d_lift"], Optional[Float[Array, "d_o"]]], float
    ],
    transition: Callable[[Float[Array, "d_lift"], Control], Float[Array, "d_lift"]],
    bases: list[Basis],
    step_size: float,
):
    def _eval_bases(x: Float[Array, "d"]) -> list[Float[Array, "m"]]:
        return list(map(lambda b, y: b(y), bases, x))

    def regularization(
        xk: Float[Array, "d_lift"], control: Control, coefficient: float
    ) -> float:
        feats = _eval_bases(xk)
        contracted = tt_matvec(control, feats)  # (d, 1)
        return 0.5 * coefficient * jnp.sum(contracted[:, 0] ** 2)

    def _solve_state(
        controls: list[Control], x0: Float[Array, "B d_lift"]
    ) -> Float[Array, "steps B d_lift"]:
        def _body_fn(xk, idx):
            control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
            next_state = jax.vmap(transition, in_axes=(0, None))(xk, control)
            return next_state, next_state

        num_steps = len(controls)
        _, states = jax.lax.scan(_body_fn, x0, jnp.arange(0, num_steps))
        states = jnp.concatenate([x0[None], states], axis=0)
        return states

    def _hamiltonian(
        state: Float[Array, "d_lift"],
        costate: Float[Array, "d_lift"],
        control: Control,
    ) -> float:
        return regularization(state, control, R) + jnp.dot(
            costate, transition(state, control)
        )

    def _solve_costate(
        controls: list[Control],
        states: Float[Array, "steps B d_lift"],
        yN: Optional[Float[Array, "B d_o"]] = None,
    ) -> Float[Array, "steps"]:
        def _body_fn(costate, idx):
            control = jax.lax.switch(idx, [lambda u=u: u for u in controls])
            xk = states[idx, ...]  # x_k
            previous_costate = jax.vmap(jax.grad(_hamiltonian), in_axes=(0, 0, None))(
                xk, costate, control
            )
            return previous_costate, previous_costate

        num_steps = len(controls)
        terminal_point = jax.vmap(jax.grad(terminal_cost))(states[-1, :], yN)
        _, costates = jax.lax.scan(
            _body_fn, terminal_point, jnp.arange(0, num_steps), reverse=True
        )
        costates = jnp.concatenate([costates, terminal_point[None]], axis=0)
        return costates

    def _build_gram_op(features: list[Float[Array, "B m"]], ranks: list[int]) -> TTO:
        B = features[0].shape[0]
        d = len(features)
        G = jax.tree.map(jax.vmap(jnp.outer), features, features)
        cpo = [jnp.tile(jnp.eye(d), (B, 1, 1))] + G  # (B, d, d) + (B, m, m)*d
        # convert the CPO to TTO
        tt_op = cpo_to_tto_truncate(
            cpo, [1] + ranks
        )  # we have a d+1-order tensor operator now
        return tt_op

    def _build_rhs(
        features: list[Float[Array, "B m"]],
        costates: Float[Array, "B d_lift"],
        ranks: list[int],
    ) -> TT:
        first_factor = -costates
        rhs = [first_factor] + features
        # convert the CP to TT
        tt = cp_to_tt_truncate(rhs, [1] + ranks)
        return tt

    def _get_update(
        states: list[Float[Array, "B d_lift"]],
        costates: Float[Array, "B d_lift"],
        controls: list[TT],
        state: MSAState,
        idx: int,
        step_size: float,
        x0: Float[Array, "B d_lift"],
        yN: Optional[Float[Array, "B d_o"]] = None,
    ) -> TT:
        features = jax.vmap(_eval_bases)(states)
        control = controls[idx]
        ranks = tt_ranks(control)
        squared_ranks = ranks[:1] + list(map(lambda x: x**2, ranks[1:]))
        squared_ranks = validate_ranks(tt_dims(control), squared_ranks)
        # build the TT operator
        gram_op = _build_gram_op(features, squared_ranks)

        # build RHS
        # rhs = _build_rhs(features, costates, squared_ranks)
        rhs = _build_rhs(
            features, costates, validate_ranks(tt_dims(control), squared_ranks)
        )
        rhs = tt_orth_right(rhs)

        iters, stag, sol = als_linear_system(
            A=gram_op,
            b=rhs,
            x0=tt_zeros_like(rhs),
            # x0=tmp_control,
            max_iters=100,
            stagnation=1e-7,
            l2_regularization=1e-10,
        )
        core = jnp.einsum("abc,cde->bde", sol[0], sol[1])
        sol = [core] + sol[2:]
        return sol

    def _solve_pmp(
        controls: list[Control],
        x0: Float[Array, "B d_lift"],
        state: MSAState,
        yN: Optional[Float[Array, "B d_o"]] = None,
    ) -> tuple[list[Control], MSAState]:
        states = _solve_state(controls, x0)
        costates = _solve_costate(controls, states, yN=yN)

        # for k in range(len(controls)):
        #     controls[k] = _get_update(
        #         states[k], costates[k], controls, state, k, step_size, x0, yN
        #     )
        updates = []
        for k in range(len(controls)):
            updates.append(
                _get_update(
                    states[k], costates[k], controls, state, k, step_size, x0, yN
                )
            )

        # find the learning rates that ensure descent
        def _objective(alpha: Float[Array, "L"]) -> float:
            new_controls = []
            for i, lr in enumerate(alpha):
                tt = tt_add(
                    tt_mul_scalar(controls[i], 1.0 - lr * R),
                    tt_mul_scalar(updates[i], lr),
                )
                tt = tt_truncate(tt, tt_ranks(controls[i]))
                new_controls.append(tt)

            states = _solve_state(new_controls, x0)
            loss = jax.vmap(terminal_cost)(states[-1, :], yN)
            return jnp.mean(loss)

        # def _objective(alpha: Float[Array, "L"]) -> float:
        #     lr = alpha[0]
        #     new_controls = []
        #     for i in range(len(controls)):
        #         tt = tt_add(
        #             tt_mul_scalar(controls[i], 1.0 - lr * R),
        #             tt_mul_scalar(updates[i], lr),
        #         )
        #         tt = tt_truncate(tt, tt_ranks(controls[i]))
        #         new_controls.append(tt)

        #     states = _solve_state(new_controls, x0)
        #     loss = jax.vmap(terminal_cost)(states[-1, :], yN)
        #     return jnp.mean(loss)

        # result = jax.scipy.optimize.minimize(
        #     _objective,
        #     x0=jnp.asarray([1e-2] * len(controls)),
        #     method="BFGS",
        # )
        # learning_rates = result.x
        # alpha = step_size * (state.iterations + 1) ** (-0.5)
        alpha = step_size
        learning_rates = [alpha] * len(controls)
        # solver = jaxopt.BFGS(fun=_objective, maxiter=500, jit=False)
        # res = solver.run([1e-4])
        # alpha = res.params[0]
        # learning_rates = [alpha] * len(controls)
        # jax.debug.print(
        #     "Learning rates: {lr} | Iter: {iter}",
        #     lr=learning_rates,
        #     iter=res.state.iter_num,
        # )
        for i, lr in enumerate(learning_rates):
            tt = tt_add(
                tt_mul_scalar(controls[i], 1.0 - lr * R), tt_mul_scalar(updates[i], lr)
            )
            tt = tt_truncate(tt, tt_ranks(controls[i]))
            controls[i] = tt

        new_state = MSAState(iterations=state.iterations + 1)
        return controls, new_state

    return _solve_pmp

from typing import Any, Callable, Iterable, Mapping, Protocol, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["Control", "discrete_pmp", "batch_pmp"]

ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
Control = ArrayTree


class PMP(Protocol):
    def __call__(self, controls: list[Control]) -> list[Control]: ...


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

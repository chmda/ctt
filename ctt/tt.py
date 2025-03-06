import math
from typing import Callable, Optional, Protocol

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import Array, Float, PRNGKeyArray

__all__ = [
    "TTCore",
    "TT",
    "tt_size",
    "tt_dims",
    "tt_ranks",
    "tt_mul_scalar",
    "tt_dot",
    "tt_norm",
    "tt_dot_rank_one",
    "tt_round",
    "tt_retract",
    "canonical_to_tt",
    "tt_rand",
    "tt_randn",
    "tt_zeros",
    "tt_matvec",
]

TTCore = Float[Array, "r0 m r1"]
TT = list[TTCore]


class RankRule(Protocol):
    def __call__(
        self,
        u: Float[Array, "n*r0 r"],
        s: Float[Array, "r"],
        vh: Float[Array, "r r1"],
        pos: int,
    ) -> int: ...


def tt_size(tt: TT) -> int:
    """
    Calculates the total number of elements in the TT representation.

    Parameters
    ----------
    tt : TT
        Tensor Train object represented as a list of jax.numpy.ndarray
        cores.

    Returns
    -------
    int
        Total number of elements in the TT cores.
    """
    return sum(core.size for core in tt)


def tt_dims(tt: TT) -> list[int]:
    """
    Extracts the dimensions of the tensor represented by the TT.

    The dimensions are inferred from the shape of the TT cores. Specifically,
    it returns a list of the second dimension (mode size) of each core.

    Parameters
    ----------
    tt : TT
        Tensor Train object represented as a list of jax.numpy.ndarray
        cores.

    Returns
    -------
    list[int]
        List of dimensions of the tensor represented by the TT.
    """
    return [core.shape[1] for core in tt]


def tt_ranks(tt: TT) -> list[int]:
    """
    Extracts the TT-ranks of the TT representation.

    Parameters
    ----------
    tt : TT
        Tensor Train object represented as a list of jax.numpy.ndarray
        cores.

    Returns
    -------
    list[int]
        List of TT-ranks.
    """
    return [core.shape[0] for core in tt] + [tt[-1].shape[2]]


def tt_mul_scalar(tt: TT, val: float) -> TT:
    """
    Multiplies a TT object by a scalar value.

    This function scales the first core of the TT object by the given scalar
    value.

    Parameters
    ----------
    tt : TT
        Tensor Train object represented as a list of jax.numpy.ndarray
        cores.
    val : float
        Scalar value to multiply the TT object by.

    Returns
    -------
    TT
        New TT object representing the scaled tensor.
    """
    new_core = val * tt[0]
    return [new_core] + tt[1:]


def tt_add(a: TT, b: TT) -> TT:
    """
    Adds two TT objects together.

    This function performs element-wise addition of two tensors represented
    in the Tensor Train format. It assumes that the two TT objects have
    compatible dimensions.

    Parameters
    ----------
    a : TT
        First Tensor Train object.
    b : TT
        Second Tensor Train object.

    Returns
    -------
    TT
        New TT object representing the sum of the two input TT objects.

    Raises
    ------
    AssertionError
        If the dimensions of the input TT objects are not compatible
        (i.e., if their mode sizes do not match).
    """
    assert all(c1.shape[1] == c2.shape[1] for c1, c2 in zip(a, b))

    comps = [jnp.concatenate((a[0], b[0]), axis=2)]

    for mu in range(1, len(a) - 1):
        c1, c2 = a[mu], b[mu]
        rp1, _, rpp1 = c1.shape
        rp2, _, rpp2 = c2.shape
        data = jnp.zeros((rp1 + rp2, c1.shape[1], rpp1 + rpp2))
        data = data.at[:rp1, :, :rpp1].set(c1)
        data = data.at[rp1:, :, rpp1:].set(c2)
        comps.append(data)

    data = jnp.concatenate((a[-1][:, :, 0], b[-1][:, :, 0]), axis=0)
    data = data[..., None]
    comps.append(data)
    return comps


def tt_dot(a: TT, b: TT) -> float:
    """
    Computes the dot product of two TT objects.

    This function calculates the inner product of two tensors represented
    in the Tensor Train format. It assumes that the two TT objects have
    the same order and compatible dimensions.

    Parameters
    ----------
    a : TT
        First Tensor Train object.
    b : TT
        Second Tensor Train object.

    Returns
    -------
    float
        The dot product of the two TT objects.

    Raises
    ------
    AssertionError
        If the two TT objects are not of the same order (i.e., do not
        have the same number of cores).
    """
    assert len(a) == len(b), "the two TTs must be of the same order"
    res = jnp.einsum("oab,oac->obc", a[0], b[0])
    for i in range(1, len(a)):
        res = jnp.einsum("obc,bnd,cnf->odf", res, a[i], b[i])
    return jnp.sum(res)


def tt_norm(tt: TT) -> float:
    """
    Computes the Frobenius norm of a TT object.

    This function calculates the Frobenius norm of a tensor
    represented in the Tensor Train format.

    Parameters
    ----------
    tt : TT
        Tensor Train object.

    Returns
    -------
    float
        The Frobenius norm of the TT object.
    """
    return jnp.sqrt(tt_dot(tt, tt))


def tt_dot_rank_one(a: TT, b: list[Float[Array, "m"]]) -> float:
    """
    Computes the dot product of a TT object and a rank-one TT.

    The rank-one TT is given in factorized form as a list of vectors.
    This is an efficient way to compute the dot product when one of the
    tensors is rank-one.

    Parameters
    ----------
    a : TT
        Tensor Train object.
    b : list[Float[Array, "m"]]
        List of vectors representing the rank-one TT in factorized form.
        Each element in the list is a jax.numpy.ndarray of shape (m_i,),
        where m_i is the dimension of the i-th mode.

    Returns
    -------
    float
        The dot product of the TT object and the rank-one TT.
    """
    cores = [v.reshape(1, -1, 1) for v in b]
    return tt_dot(a, cores)


def tt_matvec(tt: TT, x: list[Float[Array, "m"]]) -> Float[Array, "r0 rd"]:
    """
    Performs matrix-vector multiplication in TT format.

    This function computes the product of a matrix in Tensor Train format
    and a vector, where the vector is also given in a factorized form
    suitable for TT operations (list of vectors for each mode).

    Parameters
    ----------
    tt : TT
        Tensor Train object representing the matrix.
    x : list[Float[Array, "m"]]
        List of vectors representing the input vector in factorized form.
        Each element in the list is a jax.numpy.ndarray of shape (m_i,),
        where m_i is the dimension of the i-th mode.

    Returns
    -------
    Float[Array, "r0 rd"]
        The resulting vector from the matrix-vector multiplication,
        represented as a jax.numpy.ndarray.
    """
    res = jnp.einsum("oab,a->ob", tt[0], x[0])
    for i in range(1, len(tt)):
        res = jnp.einsum("ob,bnd,n->od", res, tt[i], x[i])
    return res


def _tt_modify_ranks(tt: TT, rule: RankRule) -> TT:
    """
    Modifies the ranks of a TT object according to a given rule.

    This is a helper function that applies a rank modification rule to each
    core of a Tensor Train object. It iterates through the cores, performs
    an SVD on each core (reshaped as a matrix), applies the provided `rule`
    function to determine the new rank, truncates the SVD components to
    the new rank, and updates the TT cores accordingly.

    Parameters
    ----------
    tt : TT
        Tensor Train object to modify.
    rule : RankRule
        A function that determines the new rank at each position.
        The function should accept four arguments:
          - u: Left singular vectors (jax.numpy.ndarray)
          - s: Singular values (jax.numpy.ndarray)
          - vh: Right singular vectors (jax.numpy.ndarray)
          - pos: The current core position (int, starting from 0).
        It should return an integer representing the new rank.

    Returns
    -------
    TT
        A new TT object with modified ranks according to the provided rule.

    Notes
    -----
    This function modifies the ranks of the TT object by performing
    rank truncation based on SVD. It is used internally by functions like
    `tt_round` and `tt_retract`.
    """
    # new_cores = [core.copy() for core in tt]
    # new_cores = [None] * len(tt)
    # new_cores[0] = tt[0]
    new_cores = tt_orth_right(tt)

    def _modify_cores(a: TTCore, b: TTCore, pos: int) -> tuple[TTCore, TTCore]:
        shape = a.shape
        u, s, vh = jnp.linalg.svd(
            a.reshape(shape[0] * shape[1], shape[2]), full_matrices=False
        )
        new_rank = rule(u, s, vh, pos)

        u, s, vh = u[:, :new_rank], s[:new_rank], vh[:new_rank, :]
        core = u.reshape((shape[0], shape[1], new_rank))
        next_core = jnp.einsum("i,ir,rkl->ikl", s, vh, b)
        return core, next_core

    for pos in range(len(tt) - 1):
        new_cores[pos], new_cores[pos + 1] = _modify_cores(
            new_cores[pos], new_cores[pos + 1], pos
        )
    # shape = core.shape

    # u, s, vh = jnp.linalg.svd(
    #     core.reshape(shape[0] * shape[1], shape[2]), full_matrices=False
    # )
    # new_rank = rule(u, s, vh, pos)

    # u, s, vh = u[:, :new_rank], s[:new_rank], vh[:new_rank, :]
    # new_cores[pos] = u.reshape((shape[0], shape[1], new_rank))
    # new_cores[pos + 1] = jnp.einsum("ir,rkl->ikl", s[:, None] * vh, tt[pos + 1])

    # new_cores = tt_orth_left(tt)
    # d = len(tt)
    # for mu in range(d - 1, 0, -1):
    #     core = new_cores[mu]
    #     u, s, vh = jnp.linalg.svd(
    #         core.reshape((core.shape[0], -1)), full_matrices=False
    #     )
    #     new_rank = rule(u, s, vh, mu)
    #     u, s, vh = u[:, :new_rank], s[:new_rank], vh[:new_rank, :]
    #     u = u @ jnp.diag(s)

    #     new_cores[mu] = vh.reshape((new_rank, core.shape[1], core.shape[2]))
    #     new_cores[mu - 1] = jnp.einsum("ijk,kl->ijl", new_cores[mu - 1], u)

    return new_cores


def tt_round(tt: TT, epsilon: float) -> TT:
    """
    Rounds a TT object to a specified accuracy using SVD-based rounding.

    This function reduces the ranks of a Tensor Train object while attempting
    to maintain a specified accuracy level `epsilon`. It uses an SVD-based
    rounding procedure where singular values below a threshold are discarded.
    The threshold is determined based on the desired accuracy `epsilon` and
    the norm of the TT object.

    Parameters
    ----------
    tt : TT
        Tensor Train object to round.
    epsilon : float
        Desired relative accuracy of the rounding.

    Returns
    -------
    TT
        A new TT object with reduced ranks, rounded to the specified accuracy.
    """
    delta = epsilon / math.sqrt(len(epsilon) - 1) * tt_norm(tt)

    def rule(u, s, vh, pos):
        return max(jnp.sum(s > delta).item(), 1)

    return _tt_modify_ranks(tt, rule)


def tt_retract(tt: TT, ranks: list[int]) -> TT:
    """
    Retracts (truncates ranks of) a TT object to specified ranks.

    This function explicitly sets the TT-ranks of a Tensor Train object to
    the provided ranks. It truncates the SVD components of each core to
    achieve the desired ranks.

    Parameters
    ----------
    tt : TT
        Tensor Train object to retract.
    ranks : list[int]
        List of desired TT-ranks. The length of this list should be one
        greater than the number of cores in the TT object (or the number of
        dimensions of the represented tensor). The first and last ranks are
        typically 1.

    Returns
    -------
    TT
        A new TT object with the specified TT-ranks.
    """
    ranks = validate_ranks(tt_dims(tt), ranks)

    def rule(u, s, vh, pos):
        return ranks[pos + 1]

    return _tt_modify_ranks(tt, rule)


def canonical_to_tt(cores: list[Float[Array, "r n"]]) -> TT:
    """
    Converts a list of canonical factors to Tensor Train cores.

    This function transforms a list of factors from a canonical tensor
    decomposition into the Tensor Train format. It assumes the input
    factors are in a specific canonical form suitable for conversion to TT.

    Parameters
    ----------
    cores : list[Float[Array, "r n"]]
        List of canonical factors, where each factor is a jax.numpy.ndarray
        of shape (r_i, n_i).

    Returns
    -------
    TT
        Tensor Train object representing the tensor in TT format.
    """
    # new_cores = []
    # new_cores[0] = cores[0][None, ...]

    # for mu in range(1, len(cores) - 1):
    #     factor = cores[mu]
    #     core = jnp.zeros(
    #         (factor.shape[0], factor.shape[0] + 1, factor.shape[1]), dtype=factor.dtype
    #     )
    #     core = core.at[..., 0, :].set(factor)
    #     core = core.reshape((factor.shape[0] + 1, factor.shape[0], factor.shape[1]))
    #     core = jnp.transpose(core, (0, 2, 1))
    #     new_cores.append(core[..., :-1, :, :])

    # return new_cores

    # tt_cores = [None] * len(cores)
    # tt_cores[0] = cores[0].copy()[None, ...]
    # tt_cores[-1] = cores[-1].copy()[..., None]

    def _factor_to_core(factor: Float[Array, "r n"]) -> Float[Array, "r n r"]:
        r, n = factor.shape
        """
        core = jnp.zeros((r, n, r), dtype=factor.dtype)
        return core.at[jnp.arange(r), :, jnp.arange(r)].set(core)
        """
        return jax.vmap(jnp.diag, in_axes=1)(factor).transpose(2, 0, 1)

    tt_cores = (
        [cores[0].copy()[None, ...]]
        + jax.tree.map(_factor_to_core, cores[1:-1])
        + [cores[-1].copy()[..., None]]
    )

    # for mu in range(1, len(cores) - 1):
    #     factor = cores[mu]
    #     R, N = factor.shape
    #     core = jnp.zeros((R, N, R), dtype=factor.dtype)
    #     tt_cores[mu] = core.at[jnp.arange(R), :, jnp.arange(R)].set(factor)

    return tt_cores


def _make_tt(
    dims: list[int],
    ranks: list[int],
    func: Callable[
        [tuple[int, int, int], Optional[PRNGKeyArray]],
        tuple[TTCore, Optional[PRNGKeyArray]],
    ],
    key: Optional[PRNGKeyArray] = None,
) -> TT:
    assert len(ranks) == len(dims) + 1
    ranks = validate_ranks(dims, ranks)
    cores = []
    for mu in range(len(dims)):
        shape = (ranks[mu], dims[mu], ranks[mu + 1])
        core, key = func(shape, key)
        cores.append(core)

    return cores


def tt_rand(
    key: PRNGKeyArray,
    dims: list[int],
    ranks: list[int],
    low: float = -1.0,
    high: float = 1.0,
) -> TT:
    def func(shape, key):
        key, sample_key = random.split(key)
        return random.uniform(key=sample_key, shape=shape, minval=low, maxval=high), key

    return _make_tt(dims, ranks, func, key)


def tt_randn(
    key: PRNGKeyArray,
    dims: list[int],
    ranks: list[int],
    mean: float = 0.0,
    cov: float = 1.0,
) -> TT:
    def func(shape, key):
        key, sample_key = random.split(key)
        return mean + cov * random.normal(key=sample_key, shape=shape) / jnp.prod(
            jnp.asarray(shape)
        ), key

    return _make_tt(dims, ranks, func, key)


def tt_zeros(dims: list[int], ranks: list[int]) -> TT:
    def func(shape, key):
        return jnp.zeros(shape), None

    return _make_tt(dims, ranks, func, None)


def validate_ranks(dims: list[int], ranks: list[int]) -> list[int]:
    return [ranks[0]] + [
        min(r, min(math.prod(dims[: k + 1]), math.prod(dims[k + 1 :])))
        for k, r in enumerate(ranks[1:])
    ]


@jax.jit
def tt_shift_right(a: TTCore, b: TTCore) -> tuple[TTCore, TTCore]:
    c = jnp.reshape(a, (-1, a.shape[2]))
    q, r = jnp.linalg.qr(c)
    comp = jnp.reshape(q, (a.shape[0], a.shape[1], q.shape[1]))
    comp_next = jnp.einsum("ij,jkl->ikl", r, b)
    return comp, comp_next


@jax.jit
def tt_shift_left(a: TTCore, b: TTCore) -> tuple[TTCore, TTCore]:
    c = jnp.reshape(b, (b.shape[0], -1))
    q, r = jnp.linalg.qr(c.T)
    qT = q.T
    comp = jnp.reshape(qT, (qT.shape[0], b.shape[1], b.shape[2]))
    comp_previous = jnp.einsum("ijk,kl->ijl", a, r.T)
    return comp_previous, comp


@jax.jit
def tt_orth_right(x: TT) -> TT:
    d = len(x)

    V = [None] * (d - 1)
    comp = x[-1]
    for mu in range(d - 2, -1, -1):
        comp, v = tt_shift_left(x[mu], comp)
        V[mu] = v
    return [comp] + V


@jax.jit
def tt_orth_left(x: TT) -> TT:
    d = len(x)

    U = [None] * (d - 1)
    comp = x[0]
    for mu in range(d - 1):
        u, comp = tt_shift_right(comp, x[mu + 1])
        U[mu] = u
    return U + [comp]


def tt_zeros_like(x: TT) -> TT:
    dims = tt_dims(x)
    ranks = tt_ranks(x)
    return tt_zeros(dims, ranks)


def tt_orthogonalize(x: TT) -> tuple[TT, TT, TT]:
    d = len(x)
    U = [None] * (d - 1)
    V = [None] * (d - 1)
    S = [None] * d

    # first compute U_1, ..., U_d-1
    comp = x[0]
    for k in range(d - 1):
        u, comp = tt_shift_right(comp, x[k + 1])
        U[k] = u
    # we also get S_d
    S[-1] = comp
    # then, compute V_2,...,V_d
    for k in range(d - 2, -1, -1):
        comp, v = tt_shift_left(U[k], comp)
        V[k] = v
        S[k] = comp

    return U, V, S

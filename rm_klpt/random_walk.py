from random import choice
from sage.arith.misc import is_square

from rm_klpt.quaternion_matrix import QuaternionMatrix


def _check_rm_isogeny(a, g, gamma, order):
    """
    Check if ``gamma`` defines an RM-isogeny for the RM-pair ``(a, g)``, with respect to
    ``order``.

    INPUT:

    -a, g, gamma: QuaternionMatrix
    -order: Maximal order in quaternion algebra
    """
    gamma_in_order = gamma.has_integral_coefficients(order)
    gamma_norm_square, N = is_square(gamma.reduced_norm(), root=True)
    if not (gamma_in_order and gamma_norm_square):
        return False
    new_g = N * gamma.conjugate_transpose().inverse() * g * gamma.inverse()
    new_a = gamma * a * gamma.inverse()
    check_g = new_g.has_integral_coefficients(order)
    check_a = new_a.has_integral_coefficients(order)
    return check_g and check_a, new_a, new_g


def _outward_edges(a, g, order, excluded=[]):
    """
    Compute outward edges in the (2,2)-RM-isogeny graph from the RM surface
    represented by (a, b) and not in ``excluded``.

    WARNING:
    Requires that ``p = 3 mod 4`` and that ``order``
    is the maximal order with basis (1, i, (i + j)/2, (1 + k)/2).

    INPUT:

    -a, g: QuaternionMatrix
    -order: maximal order as above
    -excluded: list of QuaternionMatrix

    OUTPUT:
    -list of ``RM_endo_data`` if step if False
        list of ``tuple[RM_endo_data, QuaternionMatrix]`` if step is True
    """

    def check_excluded(gamma):
        gammai = gamma.inverse()
        return any((delta * gammai).is_integral_unit(order) for delta in excluded)

    B = order.quaternion_algebra()
    potentials = [
        (gamma, _check_rm_isogeny(a, g, gamma, order))
        for gamma in potential_gammas(B)
        if not check_excluded(gamma)
    ]
    return [(t[0], t[1][1], t[1][2]) for t in potentials if t[1][0]]


def _random_step(a, g, order, previous_step=None):
    """
    Take a random step from ``self`` on the (2,2) RM-isogeny graph.

    Avoid walking back if ``previous_step`` is not None, and instead
    is a tuple ``(prev, gamma)`` where ``prev`` is another RM_endo_data
    and ``gamma`` is a quaternion matrix such that``self = prev.codomain(gamma)``.

    INPUT:

    -previous_step: ``None``, or a ``tuple[RM_endo_data, QuaternionMatrix]``
    """
    if previous_step is None:
        excluded = []
    else:
        gamma, old_a, old_g = previous_step
        gamma_dual = gamma.rosati(old_g, g)
        excluded = [gamma_dual]
    edges = _outward_edges(a, g, order, excluded=excluded)
    return choice(edges)


def random_walk(a, g, order, length):
    """
    Perform a random walk of length ``length`` on the RM (2,2)-isogeny graph
    starting from ``self``.

    INPUT:

    -length: nonnegative integer
    """
    previous_step = None
    new_a = a
    new_g = g
    for _ in range(length):
        old_a = new_a
        old_g = new_g
        step, new_a, new_g = _random_step(
            old_a, old_g, order, previous_step=previous_step
        )
        previous_step = (step, old_a, old_g)
    return new_a, new_g


def potential_gammas(B):
    r"""
    Enumerate quaternion matrices representing all potential (2,2)-isogenies.

    The matrices are drawn from the paper
    KLPT2 : Algebraic pathfinding in dimension two and applications,
    by Wouter Castryck, Thomas Décru, Péter Kutas, Abel Laval, Christophe Petit
    and Yan Bo Ti.

    WARNING:
    Only valid if using the order with basis `(1, i, (i + j)/2, (1 + k)/2)`.

    INPUT:

    -Quaternion algebra with discriminant `p = 3 \mod 4`
    """
    p = B.discriminant()
    i, j, k = B.gens()
    om3 = (i + j) / 2
    om4 = (1 + k) / 2
    if p % 8 == 3:
        return [
            QuaternionMatrix(B, [2, 0, 0, 1]),
            QuaternionMatrix(B, [1, 0, 0, 2]),
            QuaternionMatrix(B, [1, 1, -1, 1]),
            QuaternionMatrix(B, [1, i, -1, i]),
            QuaternionMatrix(B, [1 + i, 0, 0, 1 + i]),
            QuaternionMatrix(B, [0, 1 + i, 1 + i, i]),
            QuaternionMatrix(B, [om4, i, i + om3, 1]),
            QuaternionMatrix(B, [om4, -1, i + om3, i]),
            QuaternionMatrix(B, [i, 1 + i, 1 + i, 0]),
            QuaternionMatrix(B, [om3, 1, 1 + om4, i]),
            QuaternionMatrix(B, [1, om4, -i, i + om3]),
            QuaternionMatrix(B, [1, i - om4, -1, i + om4]),
            QuaternionMatrix(B, [i, i - om4, -i, i + om4]),
            QuaternionMatrix(B, [i, 1 - om3, -i, 1 + om3]),
            QuaternionMatrix(B, [1, 1 - om3, -1, 1 + om3]),
            QuaternionMatrix(B, [1 + om3, i, -1 + om3, i]),
            QuaternionMatrix(B, [i + om4, i, -i + om4, i]),
            QuaternionMatrix(B, [i + om4, 1, -i + om4, 1]),
            QuaternionMatrix(B, [1 + om3, 1, -1 + om3, 1]),
            QuaternionMatrix(B, [1, om3 + om4, -i, 1 + i + om3 - om4]),
            QuaternionMatrix(B, [1, 1 + i + om3 - om4, i, om3 + om4]),
            QuaternionMatrix(B, [1, i - om3 + om4, -i, i + om3 + om4]),
            QuaternionMatrix(B, [i, 1 + om3 - om4, -1, 1 + om3 + om4]),
            QuaternionMatrix(B, [1 + i + om3 - om4, 1, om3 + om4, i]),
            QuaternionMatrix(B, [1 + i + om3 - om4, -i, om3 + om4, 1]),
            QuaternionMatrix(B, [1 + om3 - om4, 1, 1 + om3 + om4, i]),
            QuaternionMatrix(B, [i + om3 - om4, -i, -i + om3 + om4, 1]),
            QuaternionMatrix(B, [1 + i + om3, i - om4, i + om4, -1 + i + om3]),
            QuaternionMatrix(B, [i + om4, 1 + i + om4, 1 + i + om3, 1 + om3]),
            QuaternionMatrix(B, [1 + om3, 1 + i - om4, -1 - i + om4, 1 + om3]),
            QuaternionMatrix(B, [1 + i + om4, -i - om4, 1 - i + om4, i - om4]),
            QuaternionMatrix(
                B, [1 + om3 + om4, 1 + i + om3 - om4, 1 + om3 - om4, -om3 - om4]
            ),
            QuaternionMatrix(
                B, [i + om3 + om4, om3 + om4, i - om3 + om4, 1 + i - om3 + om4]
            ),
            QuaternionMatrix(
                B, [1 + i + om3 - om4, i + om3 - om4, om3 + om4, -i + om3 + om4]
            ),
            QuaternionMatrix(
                B, [om3 + om4, 1 + om3 + om4, 1 + i + om3 - om4, 1 + om3 - om4]
            ),
        ]
    if p % 8 == 7:
        return [
            QuaternionMatrix(B, [2, 0, 0, 1]),
            QuaternionMatrix(B, [1, 0, 0, 2]),
            QuaternionMatrix(B, [1, 1, -1, 1]),
            QuaternionMatrix(B, [1, i, -1, i]),
            QuaternionMatrix(B, [1 + i, 0, 0, 1 + i]),
            QuaternionMatrix(B, [0, 1 + i, 1 + i, i]),
            QuaternionMatrix(B, [i, i - om4, -i, i + om4]),
            QuaternionMatrix(B, [i, 1 - om3, -i, 1 + om3]),
            QuaternionMatrix(B, [i, 1 + i, 1 + i, 0]),
            QuaternionMatrix(B, [1, 1 - om3, -1, 1 + om3]),
            QuaternionMatrix(B, [1, i - om4, -1, i + om4]),
            QuaternionMatrix(B, [1, om4, -i, i + om3]),
            QuaternionMatrix(B, [i, om4, 1, i + om3]),
            QuaternionMatrix(B, [1, 1 + om4, -i, om3]),
            QuaternionMatrix(B, [1, om3, i, 1 + om4]),
            QuaternionMatrix(B, [1 + om4, -1, om3, i]),
            QuaternionMatrix(B, [om4, i, i + om3, 1]),
            QuaternionMatrix(B, [om4, -1, i + om3, i]),
            QuaternionMatrix(B, [om3, 1, 1 + om4, i]),
            QuaternionMatrix(B, [1, 1 + i + om3 - om4, i, om3 + om4]),
            QuaternionMatrix(B, [1, om3 + om4, -i, 1 + i + om3 - om4]),
            QuaternionMatrix(B, [i, 1 + om3 - om4, -1, 1 + om3 + om4]),
            QuaternionMatrix(B, [1, i - om3 + om4, -i, i + om3 + om4]),
            QuaternionMatrix(B, [1 + i + om3 - om4, -i, om3 + om4, 1]),
            QuaternionMatrix(B, [1 + i + om3 - om4, 1, om3 + om4, i]),
            QuaternionMatrix(B, [i + om3 - om4, -i, -i + om3 + om4, 1]),
            QuaternionMatrix(B, [1 + om3 - om4, 1, 1 + om3 + om4, i]),
            QuaternionMatrix(B, [i + om3, -om4, om4, i + om3]),
            QuaternionMatrix(B, [i + om3, om3, om4, 1 + om4]),
            QuaternionMatrix(B, [om3, -1 - om4, 1 + om4, om3]),
            QuaternionMatrix(B, [1 + om4, om4, om3, i + om3]),
            QuaternionMatrix(
                B, [i + om3 + om4, om3 + om4, i - om3 + om4, 1 + i - om3 + om4]
            ),
            QuaternionMatrix(
                B, [1 + om3 + om4, 1 + i + om3 - om4, 1 + om3 - om4, -om3 - om4]
            ),
            QuaternionMatrix(
                B, [om3 + om4, 1 + om3 + om4, 1 + i + om3 - om4, 1 + om3 - om4]
            ),
            QuaternionMatrix(
                B, [1 + i + om3 - om4, i + om3 - om4, om3 + om4, -i + om3 + om4]
            ),
        ]

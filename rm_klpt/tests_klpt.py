from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.algebras.quatalg.quaternion_algebra import QuaternionAlgebra
from sage.functions.other import floor
from sage.misc.prandom import randint

from quaternion_matrix import QuaternionMatrix, lattice_subspace_intersection
from rm_data import random_rm
from represent_integer import RepresentIntegerSolver


def random_quaternion_matrix(B):
    return QuaternionMatrix(B, [B.random_element() for _ in range(4)])


def lattice_subspace_intersection_test():
    B = QuaternionAlgebra(101)
    subspace = [QuaternionMatrix(B, [a, 0, 0, a]) for a in B.basis()]
    lattice = [
        QuaternionMatrix(a * b)
        for b in subspace[0].matrix_space().basis()
        for a in B.maximal_order().basis()
    ]
    cap = lattice_subspace_intersection(lattice, subspace)
    cap = [vector(v.rational_coefficients()) for v in cap]
    lattice_mat = matrix([v.rational_coefficients() for v in lattice])
    subspace_mat = matrix([v.rational_coefficients() for v in subspace])
    assert all(v in subspace_mat.row_module(QQ) for v in cap)
    assert all(v in lattice_mat.row_module(ZZ) for v in cap)


def principal_ideal_Z_basis_test():
    B = QuaternionAlgebra(101)
    order = B.maximal_order()
    for _ in range(10):
        M = random_quaternion_matrix(B)
        ideal = M.principal_ideal_Z_basis(order=order)
        N = random_quaternion_matrix(B).numerator()
        v = vector((M * N).rational_coefficients())
        lattice_mat = matrix([v.rational_coefficients() for v in ideal])
        sol = lattice_mat.solve_left(v)
        assert all(c in ZZ for c in sol)


def inverse_test():
    B = QuaternionAlgebra(101)
    for _ in range(100):
        while True:
            M = random_quaternion_matrix(B)
            if not M.reduced_norm().is_zero():
                break
        iden_mat = M * M.inverse()
        assert iden_mat.list() == [B(1), B(0), B(0), B(1)]


def klpt_test():
    d = ZZ(5)
    p = ZZ(2**10).next_prime()
    while p % 4 != 3:
        p = p.next_prime()
    left = random_rm(d, p)
    right = random_rm(d, p)
    gamma = right.klpt(left)
    assert right == left.codomain(gamma)
    N = right.K_degree(left, gamma)
    assert N in ZZ
    ell, d = ZZ(N).is_prime_power(get_data=True)
    assert ell == 2


def represent_integer_test():
    """
    Minimal runnable example.
    """
    d = 5
    p = ZZ(2) ** 128
    p = p.next_prime()
    while p % 4 != 3:
        p = p.next_prime()

    solver = RepresentIntegerSolver(d, p)
    scale = max(1, floor(solver.threshold.sqrt()))
    best_gap = None
    best_data = None
    for _ in range(2000):
        z1 = randint(scale // 2, 2 * scale)
        z2 = randint(-scale, scale)
        M = z1 + z2 * solver.omega
        if not M.is_totally_positive():
            continue
        gap = M.norm() - solver.threshold
        if gap <= 0:
            continue
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_data = (z1, z2, M)

    if best_data is None:
        raise ValueError("Could not sample a totally positive M above the threshold.")

    z1, z2, M = best_data

    found, solution = solver.solve_four_squares(M, verbose=True)
    assert found
    assert (
        solution[0] ** 2 + solution[1] ** 2 + p * (solution[2] ** 2 + solution[3] ** 2)
        == M
    )


lattice_subspace_intersection_test()
principal_ideal_Z_basis_test()
inverse_test()
represent_integer_test()
klpt_test()

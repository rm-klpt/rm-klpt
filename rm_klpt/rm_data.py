from itertools import batched

from sage.functions.other import ceil
from sage.rings.number_field.number_field import QuadraticField
from sage.algebras.quatalg.quaternion_algebra import QuaternionAlgebra
from sage.misc.functional import log
from sage.misc.functional import sqrt
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.quadratic_forms.binary_qf import BinaryQF
from sage.arith.misc import is_square

from quaternion_matrix import (
    QuaternionMatrix,
    lattice_subspace_intersection,
)
from quaternion_ideal import QuaternionFractionalIdeal_nf

from represent_integer import RepresentIntegerSolver
from random_walk import random_walk


class RM_endo_data:
    r"""
    Endomorphism ring data for a principally polarised superspecial abelian
    surface with real multiplication

    WARNING: So far, this is only implemented if the following conditions are
    satisfied:
        - d is a sum of two squares.
        - The real quadratic field `\Q(\sqrt{d})` has class number one.

    INPUT:

    - a: A 2x2 matrix over a rational quaternion algebra `B` with prime
        discriminant, that furthermore satisfies `a^2 = d`, with `d` a positive
        integer
    - g: An IKO matrix such that `a = g^-1 a* g`
    - O0: A maximal order of `B` which contains the standard quaternion basis
        of `B`
    """

    def __init__(self, a, g, O0=None):
        self.B = a.base_ring()
        check_a, self.d = a.is_a_matrix(O0)
        check = a == a.rosati(g) and g.is_iko_matrix(O0) and check_a
        if not check:
            raise ValueError("The input is not a valid RM pair.")

        self.K = QuadraticField(self.d)
        if self.d % 4 == 1:
            self.aK = (1 + self.K.gen()) / 2
        else:
            self.aK = self.K.gen()
        mq, mp = self.B.invariants()
        self.q = ZZ(-mq)
        self.p = ZZ(-mp)
        self.rep_int_solver = RepresentIntegerSolver(self.d, self.p)
        self.BK = QuaternionAlgebra(self.K, mq, mp)
        self.a = a
        self.g = g
        self.matrix_space = a.matrix_space()
        self.O0 = O0
        if O0 is None:
            self.O0 = self.B.maximal_order()

        self.BK_matrix_K_basis = centraliser(a, O0)
        self.BK_matrix_Q_basis = sum([[M, a * M] for M in self.BK_matrix_K_basis], [])

        m_2_O_basis = [
            QuaternionMatrix(x * y)
            for y in self.matrix_space.basis()
            for x in self.O0.basis()
        ]
        m_2_O_cap_BK = lattice_subspace_intersection(
            m_2_O_basis, self.BK_matrix_Q_basis
        )
        BK_order_gen = [vector(self.matrix_to_BK(x)) for x in m_2_O_cap_BK]
        gen_matrix = matrix(BK_order_gen)
        BK_order_basis = gen_matrix.row_module(self.K.maximal_order()).basis()
        BK_order_basis = [self.BK(tuple(v)) for v in BK_order_basis]
        self.O_BK = self.BK.quaternion_order(BK_order_basis, check=False)

    def __eq__(self, other):
        return self.a == other.a and self.g == other.g and self.O0 == other.O0

    def matrix_to_BK(self, mat):
        """
        Convert a matrix that commutes with ``self.a`` into a quaternion in
        ``self.BK``.

        INPUT:

        -mat: QuaternionMatrix
        """
        v = mat.coordinates_in_rational_subspace(self.BK_matrix_Q_basis)
        K_coordinates = [v[0] + self.aK * v[1] for v in batched(v, n=2)]
        return self.BK(K_coordinates)

    def matrix_to_K(self, mat):
        """
        Convert a matrix lying in `span(I_2, self.a)` into an element of
        ``self.K``.

        INPUT:

        -mat: QuaternionMatrix
        """
        v = mat.coordinates_in_rational_subspace([mat.identity_matrix(), self.a])
        return v[0] + v[1] * self.aK

    def K_to_matrix(self, x):
        """
        Convert an element of ``self.K`` into a matrix lying in
        `span(I_2, self.a)`.
        """
        mat = matrix([vector(self.K(1)), vector(self.aK)])
        x = self.K(vector(x) * ~mat)
        return x[0] * self.a.identity_matrix() + x[1] * self.a

    def BK_to_matrix(self, x):
        """
        Convert an element of ``self.BK`` into a matrix that commutes with
        ``self.a``.
        """
        return sum(
            [
                self.K_to_matrix(c) * mat
                for c, mat in zip(x.coefficient_tuple(), self.BK_matrix_K_basis)
            ],
            self.a.new_matrix(),
        )

    def ideal(self, gamma):
        """
        Computes the OK-basis of the intersection between gamma * M_2(O) and
        Cent(a) embedded as an ideal in B ⊗ K.

        INPUT:

        -gamma: QuaternionMatrix
        """
        lattice_basis = gamma.principal_ideal_Z_basis(order=self.O0)
        ideal_matrix_basis = lattice_subspace_intersection(
            lattice_basis, self.BK_matrix_Q_basis
        )
        ideal_quat_gens = [self.matrix_to_BK(x) for x in ideal_matrix_basis]
        return QuaternionFractionalIdeal_nf(ideal_quat_gens, right_order=self.O_BK)

    def K_degree(self, other, gamma):
        """
        Compute the K-degree of RM-isogeny ``gamma`` from ``other`` to ``self``.

        INPUT:
        -other: RM_endo_data
        -gamma: QuaternionMatrix
        """
        return self.matrix_to_K(gamma * gamma.rosati(self.g, other.g))

    def initial_solution(self, other):
        """
        Outputs a quaternion matrix ``delta`` satisfying the following:
            - ``delta.conjugate_transpose() * self.g * delta == N * other.g``
            for some rational integer ``N``
            - ``delta * other.a = self.a * delta``
            - The coefficients of ``delta`` lie in ``self.O0``

        INPUT:

        -other: RM_endo_data
        """
        delta = other.a.conjugating_matrix(self.a)
        return delta.denominator(self.O0) * delta

    def check_isogeny(self, other, gamma, N_prime_power=False):
        """
        Check whether ``gamma`` represents an RM-preserving polarised isogeny
        from ``other`` to ``self``.

        If check_N is True, it will be checked whether the norm of gamma is a
        power of two. If verbose is True, prints results of the
        tests.

        INPUT:

        -other: RM_endo_data
        -gamma: QuaternionMatrix, the purported output of self.klpt(other)
        """
        if not gamma.has_integral_coefficients(self.O0):
            print("Gamma has non-integral coefficients.")
            return None
        if gamma * other.a != self.a * gamma:
            print("Not an RM-isogeny.")
            return None
        k_degree = self.K_degree(other, gamma)
        print(
            f"This is an RM-isogeny of K-degree {k_degree}. Its norm is about p^{log(k_degree.norm(), self.p).n(digits=3)}"
        )
        if N_prime_power:
            if k_degree not in ZZ:
                print("The K-degree is not rational")
                print(f"It factors as {k_degree.factor()}")
                return None
            k_degree = ZZ(k_degree)
            ell, d = k_degree.is_prime_power(get_data=True)
            if d != 0:
                print(f"The K-degree is {ell}^{d}.")
                print(f"This is about p^{log(k_degree, self.p).n()}")
            else:
                print(
                    f"The K-degree is not a prime power. It factors as {k_degree.factor()}"
                )

    def _O_BK_contains(self, x):
        """
        Check if ``self.O_BK`` contains ``x``.

        INPUT:

        -x: element of ``self.BK``
        """
        O_basis_matrix = matrix([c.coefficient_tuple() for c in self.O_BK.basis()])
        v = O_basis_matrix.solve_left(vector(x))
        ZK = self.K.maximal_order()
        return all(c in ZK for c in v)

    def is_standard(self, elements=False):
        """
        Check if ``self.O_BK`` contains a standard quaternion basis of
        ``self.BK`` with respect to invariants -self.q and -self.p.
        """
        if hasattr(self, "_is_standard"):
            if elements:
                return self._is_standard, self._OBK_quat_basis
            else:
                return self._is_standard
        is_standard, quat_basis = self._is_standard_computation()
        self._is_standard = is_standard
        self._OBK_quat_basis = quat_basis
        return self.is_standard(elements)

    def from_standard_coordinates(self, coordinates):
        check, (i, j) = self.is_standard(elements=True)
        if not check:
            raise ValueError("self is not standard")
        return (
            coordinates[0]
            + coordinates[1] * i
            + coordinates[2] * j
            + coordinates[3] * i * j
        )

    def _is_standard_computation(self):
        if all(self._O_BK_contains(e) for e in self.BK.gens()):
            return True, (self.BK.gen(0), self.BK.gen(1))
        basis = [
            a * e for e in self.O_BK.basis() for a in self.K.maximal_order().basis()
        ]
        gram = matrix(ZZ, [[(a.pair(b)).trace() for b in basis] for a in basis])
        U = gram.LLL_gram().transpose()
        red = [sum(c * g for c, g in zip(row, basis)) for row in U]
        i_candidates = [e for e in red if e**2 == -1]
        if not i_candidates:
            return False, None
        i_candidate = i_candidates[0]
        found = False
        for n in range(3**8):
            coeffs = ZZ(n).digits(base=3, padto=8)
            coeffs
            j_candidate = sum([c * e for c, e in zip(coeffs, red)])
            if (
                j_candidate**2 == -self.p
                and i_candidate * j_candidate == -j_candidate * i_candidate
            ):
                found = True
                break
        if found:
            return True, (i_candidate, j_candidate)
        return False, None

    def represent_integer(self, M):
        """
        Return an element of ``self.O_BK`` with reduced_norm M.

        WARNING: Only implemented if self is standard.

        INPUT:

        -M: element of ``self.K``
        """
        if not self.is_standard():
            raise NotImplementedError("Only implemented for standard rm data")
        if M.norm() < self.p**2:
            raise ValueError("The norm of M is too small.")
        _, sol = self.rep_int_solver.solve_four_squares(M)
        return self.from_standard_coordinates(sol)

    def strong_approximation(self, N, C, D, ell):
        r"""
        Return an element of ``self.O_BK`` of reduced norm a power of 2, and
        congruent to `Cj + Dk \mod N`

        INPUT:
            - N: a prime integral element of ``self.K``
            - C, D: integral elements of ``self.K``

        OUTPUT:
            - Quaternion
        """
        ideal = self.K.ideal(N)
        F = ideal.residue_field()
        ZK = self.K.maximal_order()

        def QF(x, y):
            return x**2 + ZZ(self.q) * y**2

        Fp = F(ZZ(self.p))
        FC = F(C)
        FD = F(D)
        e = ceil(log(self.p * N.norm() ** 4, ell))
        lam_sq = ell**e / (self.p * QF(C, D))
        if ideal.residue_symbol(lam_sq, 2) != 1:
            lam_sq *= ell
            e += 1
        Flam_sq = F(lam_sq)
        Flam = sqrt(Flam_sq)
        lam = ideal.small_residue(ZK(Flam))
        form = 2 * Fp * Flam * matrix([[FC], [F(ZZ(self.q)) * FD]])
        target = vector([F((ell**e - self.p * lam**2 * QF(C, D)) / N)])
        sol = form.solve_left(target, extend=False)
        ker = form.left_kernel()
        done = False
        while not done:
            z, t = sol + ker.random_element()
            z = ideal.small_residue(ZK(z))
            t = ideal.small_residue(ZK(t))
            M = (ell**e - self.p * QF(lam * C + N * z, lam * D + N * t)) / N**2
            if M.is_prime():
                done, lift = self.rep_int_solver.solve_two_squares(M)
        x, y = lift
        return self.from_standard_coordinates(
            [N * x, N * y, lam * C + N * z, lam * D + N * t]
        )

    def rescale(self, other, delta, gamma):
        r"""
        Updates a solution `\delta` of the RM_KLPT problem from other to self
        with an element of the intersection of `self.BK` and the ideal
        `\delta M_2(self.O0)`

        INPUT:

        -delta: A solution to the RM_KLPT problem from ``other`` to ``self``.
        -gamma: An element of the intersection of ``self.BK`` and the right
        generated by the matrix ``delta``.
        """
        return (delta.inverse() * self.BK_to_matrix(gamma)).rosati(other.g, self.g)

    def _klpt_standard(self, other, ell, check=False):
        """
        Computation of the solution of the RM_KLPT problem (see the
        documentation of the ``klpt`` method).

        Requires that ``self`` be standard (i.e ``self.is_standard()`` outputs
        ``True``).
        """
        assert self.is_standard()
        delta = self.initial_solution(other)
        if check:
            self.check_isogeny(other, delta)
        IK = self.ideal(delta)
        JK, N, eta = IK.equivalent_ideal_prime_norm(ell)
        delta = self.rescale(other, delta, eta)
        if check:
            self.check_isogeny(other, delta)
        e0 = max(
            [
                0,
                ceil(
                    log(self.p, ell)
                    + log(log(self.p, ell), ell)
                    - log(N.norm(), ell) / 2
                ),
            ]
        )
        M = N * ell**e0
        gamma = self.represent_integer(M)
        C, D = JK.ideal_mod_constraint(gamma)
        nu = self.strong_approximation(N, C, D, ell)
        gamma = self.rescale(other, delta, nu * gamma)
        n_gamma = self.K_degree(other, gamma)
        u = n_gamma.factor().unit()
        isq, s = u.is_square(root=True)
        if isq:
            gamma = gamma * other.K_to_matrix(~s)
        if check:
            self.check_isogeny(other, gamma, N_prime_power=True)
        return gamma

    def klpt(self, other, ell=2, check=False):
        """
        Return gamma in ``self.matrix_space()`` such that the following is true
            - ``gamma.conjugate_transpose() * self.g * gamma == N * other.g``
            - ``self.a * gamma == other.a * gamma``
            - N is a power of two.
        """
        if self.is_standard():
            return self._klpt_standard(other, ell, check)
        if other.is_standard():
            gamma = other._klpt_standard(self, ell, check)
            gamma = gamma.rosati(other.g, self.g)
            return gamma
        standard = initial_rm(self.d, self.p)
        gamma_1 = standard.klpt(other, ell, check)
        gamma_2 = self.klpt(standard, ell, check)
        return gamma_2 * gamma_1

    def codomain(self, gamma):
        """
        If gamma represents an RM-compatible isogeny, compute the codomain of
        ``self`` by ``gamma``

        INPUT:

        -gamma: QuaternionMatrix

        OUTPUT:
        -RM_endo_data
        """
        gamma_in_order = gamma.has_integral_coefficients(self.O0)
        gamma_norm_square, N = is_square(gamma.reduced_norm(), root=True)
        if not (gamma_in_order and gamma_norm_square):
            raise ValueError(
                "gamma must lie in M_2(self.O0) and have square reduced norm."
            )
        new_g = N * gamma.conjugate_transpose().inverse() * self.g * gamma.inverse()
        new_a = gamma * self.a * gamma.inverse()
        try:
            res = RM_endo_data(new_a, new_g, self.O0)
        except ValueError:
            raise ValueError(
                "gamma does not represent a polarised RM-oriented isogeny."
            )
        return res

    def random_walk(self, length):
        new_a, new_g = random_walk(self.a, self.g, self.O0, length)
        return RM_endo_data(new_a, new_g, self.O0)


def centraliser(a, order=None):
    r"""
    Outputs a `\Q(a)`-basis of the centraliser of quaternion matrix ``a``.

    Requires that either `a^2 = d`, with d a squarefree positive integer not
    congruent to 1 mod 4, or that `(2a-1)^2=d`, with d a squarefree positive
    integer congruent to 1 mod 4, and that the coefficients of ``a`` lie in
    ``order``. The basis is guaranteed to be a standard quaternion basis.
    If ``order`` is ``None``, ``a.base_ring().maximal_order()`` is used
    instead.

    INPUT:

    -a: A quaternion matrix
    -order: A maximal order in ``a.base_ring()`` or None
    """
    B = a.base_ring()
    check, d = a.is_a_matrix(order)
    if not check:
        raise ValueError("a should satisfy the condition of an a-matrix.")
    a_nice = standard_a(d, -B.invariants()[1])
    P = a_nice.conjugating_matrix(a)
    B = a.base_ring()
    Pinv = P.inverse()
    return [a.scalar_matrix(b).conjugate(Pinv) for b in B.basis()]


def standard_a(d, p):
    """
    Compute a symmetric matrix in `M_2(order)` with
    minmal polynomial `x^2 - x - (d-1)/4` if `d = 1 mod 4` or `x^2 - d`
    otherwise.

    INPUT:

    -d: An integer
    -order: A maximal order in a rational quaternion algebra

    OUTPUT:

    -QuaternionMatrix
    """
    if d % 4 != 1:
        raise NotImplementedError(
            "The KLPT computation is currently only supported for d = 1 mod 4."
        )
    qf = BinaryQF(1, 0, 1)
    x, y = qf.solve_integer(d)
    if ZZ(2).divides(x):
        u = (y - 1) // 2
        v = x // 2
    else:
        u = (x - 1) // 2
        v = y // 2
    B = QuaternionAlgebra(p)
    return QuaternionMatrix(B, [u + 1, v, v, -u])


def initial_rm(d, p):
    """
    Compute an ``RM_endo_data`` object with respect to ``d`` and ``p``
    whose attribute ``g`` is the identity matrix.

    INPUT:

    -d: A squarefree positive integer
    -p: A prime congruent to 3 mod 4

    OUTPUT:

    -RM_endo_data
    """
    if p % 4 != 3 or not p.is_prime():
        raise ValueError(f"p should be a prime congruent to 3 mod 4: {p}")
    B = QuaternionAlgebra(p)
    i, j, k = B.gens()
    order = B.maximal_order(order_basis=(B(1), i, (i + j) / 2, (1 + k) / 2))
    a = standard_a(d, p)
    g = a.identity_matrix()
    return RM_endo_data(a, g, order)


def random_rm(d, p):
    r"""
    Output a random RM_endo_data object with respect to ``d`` and ``p``.

    INPUT:

    -d: squarefree positive integer
    -p: prime equal to p mod 4, unramified in `\Q(\sqrt(d))`.
    """
    start = initial_rm(d, p)
    return start.random_walk(10)


def _special_order_small_norm_eq(order, target):
    """
    Solves a norm equation with small target in a special maximal order
    of the rational quaternion algebra of prime discriminant `p`.

    By small, we mean that the target is negligible with respect to `p`.

    INPUT:

    -order: maximal order in a rational quaternion algebra of prime discriminant
    -target: small integer

    OUTPUT:

    -quaternion
    """
    B = order.quaternion_algebra()
    i = B.gen(0)
    if i not in order:
        raise ValueError("order is not special")
    q, _ = B.invariants()
    qf = BinaryQF(1, 0, -q)
    sol = qf.solve_integer(target)
    if sol is None:
        return None
    return sol[0] + sol[1] * i

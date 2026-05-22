"""
Several portions of this code are directly adapted or copied from
the quaternion_algebra module of Sagemath.
"""

from sage.misc.cachefunc import cached_method, cached_function
from sage.algebras.quatalg.quaternion_algebra import (
    QuaternionFractionalIdeal,
    QuaternionOrder,
)
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.rings.ideal import Ideal_fractional
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.arith.misc import is_square
from sage.misc.prandom import randint
from sage.rings.number_field.unit_group import UnitGroup
from sage.matrix.special import identity_matrix, block_matrix
from sage.matrix.matrix_space import MatrixSpace


class QuaternionFractionalIdeal_nf(QuaternionFractionalIdeal):
    r"""
    A fractional ideal in a quaternion algebra over a number field.
    The field must have trivial class number.

    WARNING:

    - Currently only implemented for number fields with class number 1.

    INPUT:

    - ``left_order`` -- a quaternion order or ``None``

    - ``right_order`` -- a quaternion order or ``None``

    - ``gens`` -- tuple or vectors of elements of ambient
      quaternion algebra whose `\mathcal{O}_K`-span is an ideal

    - ``check`` -- boolean (default: ``True``); if ``False``, do no type
      checking.
    """

    def __init__(self, gens, left_order=None, right_order=None, check=True):
        Q = gens[0].parent()
        K = Q.base_ring()
        self.__base_ring = K.maximal_order()
        if check:
            if left_order is not None and not isinstance(left_order, QuaternionOrder):
                raise TypeError("left_order must be a quaternion order or None")
            if right_order is not None and not isinstance(right_order, QuaternionOrder):
                raise TypeError("right_order must be a quaternion order or None")
            if not isinstance(gens, (list, tuple)):
                raise TypeError("gens must be a list or tuple")
            if K.class_number() != 1:
                raise NotImplementedError("The base field must have class number 1")
            basis = [
                Q(tuple(v))
                for v in (K**4)
                .span([Q(tuple(v)).coefficient_tuple() for v in gens], self.__base_ring)
                .basis()
            ]
            if len(basis) != 4:
                raise ValueError("fractional ideal must have rank 4")
        gens_matrix = matrix([vector(gen) for gen in gens])
        basis = gens_matrix.row_module(self.__base_ring).basis()
        basis = [Q(tuple(v)) for v in basis]
        self.__left_order = left_order
        self.__right_order = right_order
        Ideal_fractional.__init__(self, Q, basis)

    def scale(self, alpha, left=False):
        """
        Scale the fractional ideal ``self`` by multiplying the basis
        by ``alpha``.

        INPUT:

        - `\alpha` -- nonzero element of quaternion algebra

        - ``left`` -- boolean (default: ``False``); if ``True`` multiply
          `\alpha` on the left, otherwise multiply `\alpha` on the right

        OUTPUT: a new fractional ideal
        """
        Q = self.quaternion_algebra()
        alpha = Q(alpha)
        if alpha == 0:
            raise ValueError("the scaling factor must be nonzero")
        if left:
            gens = [alpha * b for b in self.basis()]
        else:
            gens = [b * alpha for b in self.basis()]
        left_order = self.__left_order if alpha in QQ or not left else None
        right_order = self.__right_order if alpha in QQ or left else None
        return QuaternionFractionalIdeal_nf(
            gens, check=False, left_order=left_order, right_order=right_order
        )

    def quaternion_algebra(self):
        """
        Return the ambient quaternion algebra that contains this fractional
        ideal.

        This is an alias for `self.ring()`.
        """
        return self.ring()

    def base_ring(self):
        """
        Return the maximal order of the base field of the ambient algebra of
        ``self``.

        Equivalent to ``self.quaternion_algebra().base_ring().maximal_order()``
        """
        return self.__base_ring

    def _compute_order(self, side="left"):
        r"""
        Used internally to compute either the left or right order
        associated to an ideal in a quaternion algebra.

        INPUT:

        - ``side`` -- ``'left'`` or ``'right'``

        OUTPUT: the left order if ``side='left'``; the right order if
        ``side='right'``

        ALGORITHM: Let `b_1, b_2, b_3, b_3` be a basis for this
        fractional ideal `I`, and assume we want to compute the left
        order of `I` in the quaternion algebra `Q`.  Then
        multiplication by `b_i` on the right defines a map `B_i:Q \to
        Q`.  We have

        .. MATH::

           R = B_1^{-1}(I) \cap B_2^{-1}(I) \cap B_3^{-1}(I)\cap B_4^{-1}(I).

        This is because

        .. MATH::

           B_n^{-1}(I) = \{\alpha \in Q : \alpha b_n \in I \},

        and

        .. MATH::

           R = \{\alpha \in Q : \alpha b_n \in I, n=1,2,3,4\}.
        """
        if side == "left":
            action = "right"
        elif side == "right":
            action = "left"
        else:
            raise ValueError("side must be 'left' or 'right'")
        Q = self.quaternion_algebra()

        M = [(~b).matrix(action=action) for b in self.basis()]
        B = self.basis_matrix()
        invs = block_matrix([[B * m] for m in M])
        basis_vectors = invs.row_module(self.__base_ring)
        return Q.quaternion_order([Q(tuple(v)) for v in basis_vectors])

    def left_order(self):
        """
        Return the left order associated to this fractional ideal.

        OUTPUT: an order in a quaternion algebra
        """
        if self.__left_order is None:
            self.__left_order = self._compute_order(side="left")
        return self.__left_order

    def right_order(self):
        """
        Return the right order associated to this fractional ideal.

        OUTPUT: an order in a quaternion algebra
        """
        if self.__right_order is None:
            self.__right_order = self._compute_order(side="right")
        return self.__right_order

    def __repr__(self) -> str:
        """
        Return string representation of this quaternion fractional ideal.
        """
        return f"Fractional ideal {self.gens()}"

    def random_element(self, *args, **kwds):
        """
        Return a random element in the rational fractional ideal ``self``.

        The ``args`` and ``kwds`` are passed to the ``random_element`` method
        of the base ring.
        """
        return sum(
            self.__base_ring.random_element(*args, **kwds) * g for g in self.gens()
        )

    def basis(self):
        """
        Return a basis for this fractional ideal.

        OUTPUT: tuple
        """
        return self.gens()

    def __hash__(self) -> int:
        """
        Return the hash of ``self``.
        """
        return hash(self.gens())

    @cached_method
    def basis_matrix(self):
        r"""
        Return basis matrix `M` in Hermite normal form for ``self`` as a
        matrix with rational entries.

        If `Q` is the ambient quaternion algebra, then the `\ZZ`-span of
        the rows of `M` viewed as linear combinations of Q.basis() =
        `[1,i,j,k]` is the fractional ideal ``self``.  Also,
        ``M * M.denominator()`` is an integer matrix in Hermite normal form.

        OUTPUT: matrix over `\QQ`
        """
        B = matrix([vector(e.coefficient_tuple()) for e in self.gens()])
        d = B.denominator()
        C = B * d
        return C.hermite_form() / d

    def _Z_basis(self):
        """
        Return a basis of self as a Z-module.

        OUTPUT: tuple
        """
        return tuple(a * g for g in self.basis() for a in self.__base_ring.basis())

    def _Z_gram_matrix(self):
        r"""
        Return the gram matrix of self as a `\ZZ`-module with respect
        to the quadratic form defined as trace of reduced norm.

        OUTPUT: matrix over `\ZZ`
        """
        basis = self._Z_basis()
        return matrix(ZZ, [[(a.pair(b)).trace() for b in basis] for a in basis])

    @cached_method
    def reduced_Z_basis(self):
        r"""
        Return a basis of the ideal as a `\ZZ`-module, which is LLL-reduced
        with respect to the quadratic form defined as the trace of the
        reduced norm

        OUTPUT: tuple
        """
        if not self.quaternion_algebra().is_totally_definite():
            raise TypeError("the quaternion algebra must be totally definite")

        U = self._Z_gram_matrix().LLL_gram().transpose()
        return tuple(sum(c * g for c, g in zip(row, self._Z_basis())) for row in U)

    @cached_method
    def gram_matrix(self):
        """
        Return the Gram matrix of this fractional ideal.

        OUTPUT: `4 \times 4` matrix over the base field of the ambient algebra
        """
        A = self.gens()
        K = self.quaternion_algebra().base_ring()
        two = K(2)
        m = [two * a.pair(b) for b in A for a in A]
        M44 = MatrixSpace(K, 4, 4)
        return M44(m, coerce=False)

    def norm(self):
        """
        Return the reduced norm of this fractional ideal.

        OUTPUT: element of the base field of the ambient algebra
        """
        K = self.quaternion_algebra().base_ring()
        R = self.__left_order or self.__right_order or self.left_order()
        R_bm = matrix([c.coefficient_tuple() for c in R.basis()])
        basis_wrt_order = R_bm.solve_left(self.basis_matrix())
        r = basis_wrt_order.det()
        check_1, s = _generates_square_ideal(r, root=True)
        assert check_1, "absolute norm is not a square."
        return K.ideal(s)

    def __contains__(self, x):
        """
        Checks if x is an element of self.
        """
        v = vector(x.coefficient_tuple())
        coords = self.basis_matrix().solve_left(v)
        return all(c in self.__base_ring for c in coords)

    def equivalent_ideal_prime_norm(self, ell, left=False):
        r"""
        Return a fractional ideal of prime norm with the same right order as
        ``self``, such that ell is not a square modulo the output.

        If left is True, the left order is preserved instead of the right.
        order.

        INPUT:

        -ell: integer
        -left: boolean

        OUTPUT

        - The right-equivalent fractional ideal of prime norm
        - The norm of the new ideal, as a totally positive number
        - The element used to generate the output ideal by rescaling.
        """
        basis = self.reduced_Z_basis()
        K = self.ring().base_ring()
        N = self.norm().gens_reduced()[0]
        m = 1
        while True:
            delta = sum(randint(-m, m) * b for b in basis)
            dnorm = delta.reduced_norm() / N
            dnorm_id = K.ideal(dnorm)
            check_tot_pos, dnorm = _has_totally_positive_associate(dnorm)
            if (
                check_tot_pos
                and dnorm.is_prime()
                and ell not in dnorm_id
                and K.ideal(dnorm).residue_symbol(ell, 2) != 1
            ):
                break
        return self.scale(delta.conjugate() / N, left=not left), dnorm, delta

    def ideal_mod_constraint(self, gamma):
        r"""
        Return ``C, D`` in `\ZZ` such that ``(j * C + k * D) * gamma in I``,
        where ``j`` and ``k`` are elements of the standard basis of
        ``self.ring()``.

        Input:
        - quaternion
        """
        N = self.norm()
        _, j, k = self.quaternion_algebra().gens()
        ZK = self.base_ring()
        mat = matrix([_quat_mod_I(N, e).list() for e in self.basis()])
        system = (
            matrix([_quat_mod_I(N, x * gamma).list() for x in [j, k]])
            * mat.right_kernel_matrix().transpose()
        )
        C, D = system.kernel().basis()[0]
        return N.small_residue(ZK(C)), N.small_residue(ZK(D))


def _has_totally_positive_associate(x):
    """
    Test if x is totally positive, up to a multiplication by a unit.

    INPUT:

    -x: an element of the fraction field of self._Base_ring
    """
    K = x.parent()
    assert K.is_totally_real()
    if x.is_totally_positive():
        return True, x
    if (-x).is_totally_positive():
        return True, -x
    mus = [
        mu for mu in UnitGroup(K).fundamental_units() if not mu.is_totally_positive()
    ]
    if not mus:
        return False, None
    g = x * mus[0]
    if g.is_totally_positive():
        return True, g
    return True, -g


@cached_function
def _quat_mod_I_basis(B, ideal):
    r"""
    Return matrices over a finite field which correspond to elements of the
    basis of ``B`` modulo ``ideal``.

    INPUT:
    -B: A quaternion algebra over a number field K.
    -ideal: A fractional ideal in K.

    OUTPUT:
    -Elements of `M_2(\Z_K / ideal)` that are images of the quaternion basis of
    ``B`` by the projection to `O / ideal`, with `O` any maximal order
    containing the basis.
    """
    if not ideal.is_prime() or ideal in B.ramified_primes():
        raise ValueError("The ideal must be a prime that does not ramify in B")
    a, b = B.invariants()
    if a in ideal:
        raise ValueError("The first invariant of the quaternion algebra may\
        not be contained in the ideal")
    F = ideal.residue_field()
    i2 = F(a)
    j2 = F(b)
    i2inv = ~i2
    imod = matrix(F, 2, 2, [0, i2, 1, 0])
    alpha = None
    for z in F:
        if z.is_zero():
            pass
        c = j2 + i2inv * z**2
        if c.is_square():
            alpha = c.sqrt()
            break

    jmod = matrix(F, 2, 2, [alpha, z, -z * i2inv, -alpha])
    kmod = imod * jmod
    return [identity_matrix(F, 2), imod, jmod, kmod]


def _quat_mod_I(ideal, x):
    r"""
    Return the image of `x mod ideal` as a matrix over a finite field.

    INPUT:
    -ideal: fractional ideal in the base field of the algebra of ``x``
    -x: quaternion

    OUTPUT:
    -A matrix in `M_2(\Z_K / ideal)`
    """
    B = x.parent()
    red_basis = _quat_mod_I_basis(B, ideal)
    F = ideal.residue_field()
    return sum([F(c) * e for c, e in zip(x.coefficient_tuple(), red_basis)])


def _generates_square_ideal(x, root=False):
    r"""
    Checks if an element of a real quadratic field generates a
    principal ideal that is a square.

    INPUT:
    - x: Element of a real quadratic field.

    OUTPUT:
    -A boolean
    -A square root of x up to a unit if it exists. (Only if root flag is set)
    """
    K = x.parent()
    eta = UnitGroup(K).fundamental_units()[0]
    units = [1, -1, eta, -eta]
    res = [is_square(u * x, root=True) for u in units]
    res = [t for t in res if t[0]]
    if res:
        return res[0]
    return False, None

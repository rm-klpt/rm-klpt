from itertools import batched

from sage.algebras.quatalg.quaternion_algebra import QuaternionAlgebra_ab
from sage.structure.element import Matrix
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix
from sage.matrix.special import zero_matrix
from sage.matrix.special import block_matrix
from sage.arith.functions import lcm
from sage.modules.free_module_element import vector
from sage.modules.free_quadratic_module_integer_symmetric import IntegralLattice


class QuaternionMatrix:
    """
    A wrapper class for 2x2 matrices over a rational quaternion algebra.

    TODO: It would make more sense to have a class that inherits the Sage
    matrix classes, but this is simpler for a first implementation.

    INPUT:

    -arg0: a matrix with base ring a rational quaternion algebra
            or a rational quaternion algebra

    -arg1: if ``arg0`` is not a matrix, a tuple or list of either 4 quaternions
            or 16 rational numbers
    """

    def __init__(self, arg0, arg1=None):
        if isinstance(arg0, Matrix):
            self._from_matrix(arg0)
        elif isinstance(arg0, QuaternionAlgebra_ab):
            if arg0.base_ring() != QQ:
                raise NotImplementedError("The base ring should be a rational\
                quaternion algebra.")
            if not isinstance(arg1, (list, tuple)):
                raise TypeError("The second argument should be a list or a\
                tuple.")
            if len(arg1) == 4:
                self._from_matrix(matrix(arg0, 2, 2, arg1))
            elif len(arg1) == 16:
                self._from_rational_coefficients(arg0, arg1)
            else:
                raise ValueError("The coefficients should either be 4\
                quaternions or 16 rationals.")
        else:
            raise ValueError("Invalid input.")

    def __repr__(self):
        return self._data.__repr__()

    def _from_matrix(self, matrix):
        """
        Ancillary function for generating an object form a matrix of
        quaternions
        """
        if not isinstance(matrix.base_ring(), QuaternionAlgebra_ab):
            raise TypeError("The matrix's base ring must be a quaternion\
            algebra.")
        if matrix.base_ring().base_ring() != QQ:
            raise NotImplementedError(
                "The base ring must be defined over the rationals."
            )
        if matrix.nrows() != 2 or matrix.ncols() != 2:
            raise NotImplementedError("Only 2x2 matrices are currently suppoerted.")
        self._data = matrix
        self._base_ring = matrix.base_ring()

    def _from_rational_coefficients(self, ring, coefficients):
        """
        Ancillary function for generating an object form its base ring and a
        list of 16 rational coefficients.
        """
        if any(x not in QQ for x in coefficients):
            raise ValueError("If 16 coefficients are given, they should be\
            rational numbers.")
        mat = matrix(ring, 2, 2, [ring(c) for c in batched(coefficients, 4)])
        self._from_matrix(mat)

    def __getitem__(self, index):
        return self._data[index]

    def __mul__(self, other):
        if isinstance(other, QuaternionMatrix):
            return QuaternionMatrix(self._data * other._data)
        return QuaternionMatrix(self._data * other)

    def __rmul__(self, other):
        if isinstance(other, QuaternionMatrix):
            return QuaternionMatrix(other._data * self._data)
        return QuaternionMatrix(other * self._data)

    def __add__(self, other):
        return QuaternionMatrix(self._data + other._data)

    def __radd__(self, other):
        if isinstance(other, QuaternionMatrix):
            return self + other
        if other == 0:
            return self
        raise TypeError(f"Cannot add QuaternionMatrix with nonzero {type(other)}")

    def __neg__(self):
        return QuaternionMatrix(-self._data)

    def __sub__(self, other):
        return QuaternionMatrix(self._data - other._data)

    def __eq__(self, other):
        return self._data == other._data

    def base_ring(self):
        """
        Return the base ring of ``self``.
        """
        return self._base_ring

    def list(self):
        """
        Return a list of the coefficients of ``self``.
        """
        return self._data.list()

    def matrix_space(self):
        """
        Return the matrix space of 2x2 matrices over ``self.base_ring()``
        Equivalent to ``MatrixSpace(self.base_ring(), 2, 2)``.
        """
        return self._data.matrix_space()

    def new_matrix(self):
        """
        Return a zero QuaternionMatrix.
        """
        return QuaternionMatrix(self._data.new_matrix())

    def identity_matrix(self):
        """
        Return the identity matrix in the same ring as ``self``.
        """
        return QuaternionMatrix(self.base_ring(), [1, 0, 0, 1])

    def conjugate_transpose(self):
        """
        Return the conjugate-transpose of ``self``.
        """
        return QuaternionMatrix(self._data.conjugate_transpose())

    def reduced_norm(self):
        """
        Return the reduced norm of self.
        """
        x = self[0, 0]
        y = self[0, 1]
        z = self[1, 0]
        w = self[1, 1]
        return (
            x.reduced_norm() * w.reduced_norm()
            + y.reduced_norm() * z.reduced_norm()
            - (y.conjugate() * x * z.conjugate() * w).reduced_trace()
        )

    def reduced_trace(self):
        """
        Return the reduced trace of self.
        """
        return self._data.trace().reduced_trace()

    def inverse(self):
        """
        Return the inverse of self.
        """
        N = self.reduced_norm()
        x = self[0, 0]
        y = self[0, 1]
        z = self[1, 0]
        w = self[1, 1]
        return N.inverse() * QuaternionMatrix(
            self.base_ring(),
            [
                w.reduced_norm() * x.conjugate() - z.conjugate() * w * y.conjugate(),
                y.reduced_norm() * z.conjugate() - x.conjugate() * y * w.conjugate(),
                z.reduced_norm() * y.conjugate() - w.conjugate() * z * x.conjugate(),
                x.reduced_norm() * w.conjugate() - y.conjugate() * x * z.conjugate(),
            ],
        )

    def is_rational_sqrt(self):
        """
        Check if ``self**2`` is a rational number
        """
        M2 = self._data**2
        d = M2[0, 0]
        return (M2.is_scalar() and d in QQ), QQ(d)

    def denominator(self, order=None):
        """
        Return the smallest rational integer ``d`` such that ``d * self`` lies
        in `M_2(order)`.

        Uses ``self.base_ring().maximal_order()`` if ``order`` is ``None``.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        if order is None:
            B = self.base_ring()
            order = B.maximal_order()
        coefs = [order.basis_matrix().solve_left(vector(c)) for c in self.list()]
        return lcm([c.denominator() for c in sum(list(map(list, coefs)), [])])

    def numerator(self, order=None):
        """
        Return ``self.denominator(order=order) * self``

        INPUT:

        -order: a maximal order in ``self.base_ring()``.
        """
        return self.denominator(order) * self

    def is_iko_matrix(self, order=None):
        """
        Check is ``self`` is an IKO-matrix.

        That is, check if ``self`` is symmetric, lies in `M_2(order)`, has
        reduced norm equal to 1 and positive diagonal coefficients.
        Uses ``self.base_ring().maximal_order()`` if ``order`` is ``None``.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        if order is None:
            order = self.base_ring().maximal_order()
        sym = self == self.conjugate_transpose()
        diag = self[0, 0] in ZZ and self[1, 1] in ZZ
        pos = diag and ZZ(self[0, 0]) > 0 and ZZ(self[1, 1]) > 0
        antidiag = self[0, 1] in order
        norm = self.reduced_norm() == 1
        return sym and diag and pos and antidiag and norm

    def rosati(self, g1, g2=None):
        """
        Return the image of ``self`` by the Rosati map with respect to ``g1``
        and ``g2`` (a generalisation of the Rosati involution)

        Uses ``g1`` for ``g2`` if ``g2`` is ``None``.

        INPUT:

        -g1, g2 : quaternion matrices that are IKO matrices
        """
        if g2 is None:
            g2 = g1
        return g2.inverse() * self.conjugate_transpose() * g1

    def conjugating_matrix(self, other):
        """
        Return a matrix P such that `P * self * P^-1 = other` if it exists.

        WARNING: Doesn't check that the input matrices are conjugated, and may
        loop forever if they aren't.

        INPUT:

        -other: a quaternion matrix that is conjugated to ``self``
        """
        if self == other:
            return self.identity_matrix()
        id = self.identity_matrix()
        tensor = [(id, self), (-other, id)]
        rows = [
            b.apply_tensor_endomorphism(tensor) for b in self.rational_space_basis()
        ]
        system = matrix([mat.rational_coefficients() for mat in rows])
        ker = system.left_kernel()
        while True:
            P = QuaternionMatrix(self.base_ring(), tuple(ker.random_element()))
            if P.reduced_norm() != 0:
                return P

    def apply_tensor_endomorphism(self, tensor):
        """
        Return ``tensor(self)``, where ``tensor`` is a list of pairs of
        matrices defining an element in the envelopping algebra of quaternion
        matrices.

        INPUT:
        - tensor: a list of pairs of QuaternionMatrices
        """
        return sum([t[0] * self * t[1] for t in tensor], self.new_matrix())

    def rational_space_basis(self):
        r"""
        Return a `\Q`-basis of the matrix space of ``self``.
        """
        return [
            QuaternionMatrix(a * b)
            for a in self.matrix_space().basis()
            for b in self.base_ring().basis()
        ]

    def rational_coefficients(self):
        """
        Return a list of rational coefficients of self.
        The output is coherent with the basis returned by
        ``self.rational_space_basis()``.

        OUTPUT: list
        """
        return [a for x in self.list() for a in x.coefficient_tuple()]

    def coordinates_in_rational_subspace(self, basis):
        r"""
        Return the coordinates of ``self`` with respect to `\Q`-basis ``basis``
        of a subspace of ``self.matrix_space()``.

        INPUT:

        -basis: iterable of quaternion matrices

        OUTPUT:

        -vector or None
        """
        mat = matrix([b.rational_coefficients() for b in basis])
        if mat.nullity() != 0:
            raise ValueError("Input should be a basis of a subspace.")
        target = vector(self.rational_coefficients())
        try:
            return mat.solve_left(target)
        except ValueError:
            return None

    def matrix_space_order_Z_basis(self, order=None):
        r"""
        Return a `\Z`-basis of the matrix space of degree 2
        over ``order``.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        if order is None:
            B = self.base_ring()
            order = B.maximal_order()
        Obasis = [
            QuaternionMatrix(a * M)
            for M in self.matrix_space().basis()
            for a in order.basis()
        ]
        return Obasis

    def principal_ideal_Z_basis(self, order=None):
        r"""
        Return a `\Z`-basis of the principal right `M_2(order)`-ideal generated
        by ``self``.

        If ``order`` is ``None``, ``self.base_ring().maximal_order()`` is used
        instead.

        INPUT:

        -order: maximal order in ``self.base_ring()``
        """
        Obasis = self.matrix_space_order_Z_basis(order)
        return [self * b for b in Obasis]

    def is_integer(self):
        r"""
        Check if ``self`` is an integer.

        That is, a scalar matrix with coefficients in `\Z`.
        """
        check = self._data.is_scalar() and self[0, 0] in ZZ
        N = None
        if check:
            N = ZZ(self[0, 0])
        return check, N

    def conjugate(self, P):
        """
        Return the image of ``self`` by conjugation by ``P``

        INPUT:

        - P: an invertible quaternion matrix
        """
        return P.inverse() * self * P

    def scalar_matrix(self, b):
        """
        Return the scalar matrix ``b`` lying in the same space as self.

        INPUT:

        -b: element of ``self.base_ring()``
        """
        return QuaternionMatrix(self.base_ring(), [b, 0, 0, b])

    def is_a_matrix(self, order=None):
        r"""
        Check that self lies in `M_2(order)` and that the minimal polynomial
        of ``self`` over `\Q` is quadratic and that it is the minimal
        polynomial of a standard generator of the ring of integers of a real
        quadratic field.

        By standard generator, we mean that the minimal polynomial must be:
            - `x^2 - d` if `d % 4 != 1`
            - `x^2 - x - (d - 1) / 4` if `d % 4 = 1`
        If order is None, ``self.base_ring().maximal_order()`` is used instead.

        INPUT:

        -order: a maximal order of ``self.base_ring()`` or None

        OUTPUT:

        -a boolean
        -a squarefree integer d whose square root generates the field `Q(self)`,
            or `0` if ``self`` is not an "a-matrix".
        """
        if order is None:
            order = self.base_ring().maximal_order()
        if not self.has_integral_coefficients(order):
            return False, 0
        v = (self * self).coordinates_in_rational_subspace(
            [self.identity_matrix(), self]
        )
        if v is None:
            return False, 0
        elif v[1] == 0:
            d = v[0]
            return d > 0, ZZ(d)
        elif v[1] == 1:
            d = 4 * v[0] + 1
            return d > 0, ZZ(d)
        else:
            return False, 0

    def has_integral_coefficients(self, order=None):
        """
        Checks if ``self`` lies in `M_2(order)`.

        If ``order`` is ``None``, ``self.base_ring().maximal_order()``
        is used instead.

        INPUT:

        -QuaternionOrder
        """
        if order is None:
            order = self.base_ring().maximal_order()
        return all(c in order for c in self.list())

    def is_integral_unit(self, order=None):
        """
        Check if ``self`` is a unit in ``M_2(order)``.

        If ``order`` is ``None``, use ``self.base_ring().maximal_order()`` instead.

        INPUT:

        -QuaternionOrder or None
        """
        return self.has_integral_coefficients(order) and self.reduced_norm().abs() == 1

    def hermitian_lattice(self, order):
        """
        Return a reduced basis of the lattice of hermitian matrices with
        respect to the ``self.rosatti`` involution.

        ``self`` must be an iko_matrix.

        INPUT:

        -order: A maximal order in ``self.base_ring()``
        """
        if not self.is_iko_matrix(order):
            raise ValueError("self should be an iko matrix.")
        B = self.base_ring()
        symmetric_matrices = [
            QuaternionMatrix(B, [1, 0, 0, 0]),
            QuaternionMatrix(B, [0, 0, 0, 1]),
        ] + [QuaternionMatrix(B, [0, b, b.conjugate(), 0]) for b in order.basis()]
        lattice_basis = [e * self for e in symmetric_matrices]
        gram = matrix(
            ZZ,
            [
                [(a.rosati(self) * b).reduced_trace() for b in lattice_basis]
                for a in lattice_basis
            ],
        )
        L = IntegralLattice(gram)
        return L.LLL(), lattice_basis

    def compatible_a_matrices(self, d, order):
        """
        Return the list of RM-matrices in `M_2(order)` that are
        compatible with IKO matrix ``self``.

        INPUT:

        -d: squarefree positive integer
        -order: Maximal order in ``self.base_ring()``
        """
        hermitian_lattice, lattice_basis = self.hermitian_lattice(order)
        if d % 4 == 1:
            tr = 1
            det = (1 - d) // 4
        elif d % 4 in [2, 3]:
            tr = 0
            det = -d
        else:
            raise ValueError(f"d should not be a multple of 4: {d}")
        norm_target = 2 * tr**2 - 4 * det
        short_vectors = hermitian_lattice.short_vectors(norm_target + 1)
        short_vectors = [
            sum([c * e for c, e in zip(list(v), lattice_basis)])
            for v in short_vectors[norm_target]
        ]
        a_matrices = [(a, a.is_a_matrix(order)) for a in short_vectors]
        return [a for (a, (check, d_a)) in a_matrices if check and d_a == d]

    def iko_matrix_automorphisms(self, order):
        """
        Output the list polarised automorphisms of the polarised surface represented
        by IKO matrix ``self``.

        That is, output elements `u` in `GL_2(order)` such that
        ``self = u.conjugate_transpose() * self * u``

        INPUT:
        -order: maximal order of ``self.base_ring()``
        """
        if not self.is_iko_matrix(order):
            raise ValueError("self should be an iko matrix.")
        Obasis = self.matrix_space_order_Z_basis(order)
        gram = matrix(
            [[ZZ((a.rosati(self) * b).reduced_trace()) for b in Obasis] for a in Obasis]
        )
        L = IntegralLattice(gram)
        L = L.LLL()
        short_vectors = L.short_vectors(3)
        short_matrices = [
            sum([c * e for c, e in zip(list(v), Obasis)]) for v in short_vectors[2]
        ]
        return [
            gamma
            for gamma in short_matrices
            if gamma.rosati(self) * gamma == self.identity_matrix()
        ]

    def a_matrices_equivalence_classes(self, d, order):
        """
        Compute a list of equivalence classes of RM matrices compatible with ``self``

        ``self`` must be an IKO matrix.

        INPUT:
        -d: Squarefree positive integer
        -order: Maximal order of ``self.base_ring()``.

        OUTPUT:
        List of lists: each internal list represent an equivalence class of RM-matrices.
        """
        autos = self.iko_matrix_automorphisms(order)
        a_matrices = self.compatible_a_matrices(d, order)
        classes = []

        def are_equivalent(left, right):
            return any([gamma * left == right * gamma for gamma in autos])

        for a in a_matrices:
            found = False
            for c in classes:
                if are_equivalent(c[0], a):
                    c.append(a)
                    found = True
                    break
            if not found:
                classes.append([a])
        return classes


def quaternion_order_norm_equation(order, N):
    """
    Return an element of reduced norm ``N`` in quaternion order ``order``.

    INPUT:

    -order: a maximal order in a rational quaternion algebra
    -N: A positive integer
    """
    s = order.quadratic_form().solve(N)
    if all(c in ZZ for c in s):
        return True, order(s * order.basis_matrix())
    else:
        return False, None


def _col_padding(mat):
    """
    Pads a matrix with more rows than collumns into a square matrix.

    Uses zeroes for the padding.

    INPUT:

    -mat: A matrix
    """
    pad = zero_matrix(mat.base_ring(), mat.nrows(), mat.nrows() - mat.ncols())
    return block_matrix([[mat, pad]], subdivide=False)


def lattice_subspace_intersection(lattice_basis, subspace_basis):
    r"""
    Compute a `\Z`-basis of the intersection of a `\Z`-lattice and a
    `\Q`-subspace of 2x2 quaternion matrices.

    INPUT:

    -lattice_basis: an iterable of a basis of the `\Z`-lattice.
    -subspace_basis: an iterable of a basis of the `\Q`-subspace.

    OUTPUT:

    -list

    ALGORITHM:

    See Lemma 3.1 in Eﬃcient reductions among lattice problems by
    Daniele Micciancio.
    """
    L = matrix([v.rational_coefficients() for v in lattice_basis])
    G = matrix([v.rational_coefficients() for v in subspace_basis])
    H = G.right_kernel_matrix().transpose()
    H *= H.denominator()
    M = _col_padding(L * H)
    _, U, _ = M.smith_form(transformation=True, integral=True)
    n = len(lattice_basis)
    m = len(subspace_basis)
    res_vectors = (U * L).rows()[n - m :]
    B = lattice_basis[0].base_ring()
    return [QuaternionMatrix(B, tuple(v)) for v in res_vectors]

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


class Domain:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.x = None
        self.y = None
        self.is_bc = None
        self.mask = None
        self.cells = None
        self._allocate()
        self._set_bc_bool()
        self._set_mask_bool()
        self._set_cell_indices()
        self.dx = self.x[1, 0] - self.x[0, 0]
        self.dy = self.y[0, 1] - self.y[0, 0]

    @property
    def shape(self):
        return self.x.shape

    def is_compatible(self, domain):
        try:
            assert self.shape == domain.shape
            assert np.all(self.is_bc == domain.is_bc)
            assert np.all(self.mask == domain.mask)
            assert isinstance(self, type(domain))
        except:
            return False
        return True

    def index(self, i, j):
        ny = self.m + 1
        return i*ny + j

    # TODO: change _set_ to _initialize_
    def _set_bc_bool(self):
        raise NotImplementedError()

    def _set_mask_bool(self):
        raise NotImplementedError()

    def _set_cell_indices(self):
        raise NotImplementedError()

    def _allocate(self):
        raise NotImplementedError()

    def _get_cells(self):
        raise NotImplementedError()

    def is_valid_gridpoint(self, i, j):
        return not self.mask[i, j]

    def is_on_bc(self, i, j):
        return self.is_bc[i, j]

    def is_interior(self, i, j):
        return self.is_valid_gridpoint(i, j) and not self.is_on_bc(i, j)

    def _indices_boundary(self, condition):
        for j in range(self.m+1):
            for i in range(self.n+1):
                if condition(i, j):
                    assert self.is_on_bc(i, j)
                    yield i, j

    def indices_boundary_left(self):
        raise NotImplementedError()

    def indices_boundary_right(self):
        raise NotImplementedError()

    def indices_boundary_top(self):
        raise NotImplementedError()

    def indices_boundary_bottom(self):
        raise NotImplementedError()


class UnitSquare(Domain):
    def _allocate(self):
        x = np.linspace(0, 1, self.n+1)
        y = np.linspace(0, 1, self.m+1)
        self.x, self.y = np.meshgrid(x, y, indexing='ij')

    def _set_bc_bool(self):
        self.is_bc = np.zeros(self.x.shape, dtype=bool)
        self.is_bc[:] = False
        self.is_bc[:, 0] = True
        self.is_bc[:, -1] = True
        self.is_bc[0, :] = True
        self.is_bc[-1, :] = True

    def _set_mask_bool(self):
        self.mask = np.zeros(self.x.shape, dtype=bool)

    def _set_cell_indices(self):
        self.cells = []
        for j in range(self.m):
            for i in range(self.n):
                assert self.is_valid_gridpoint(i, j)
                assert self.is_valid_gridpoint(i+1, j)
                assert self.is_valid_gridpoint(i, j+1)
                assert self.is_valid_gridpoint(i+1, j+1)
                self.cells.append((
                    (i, j),
                    (i+1, j),
                    (i+1, j+1),
                    (i, j+1),
                ))

    def indices_boundary_left(self):
        return self._indices_boundary(lambda i, j: i == 0)

    def indices_boundary_right(self):
        return self._indices_boundary(lambda i, j: i == self.n)

    def indices_boundary_bottom(self):
        return self._indices_boundary(lambda i, j: j == 0)

    def indices_boundary_top(self):
        return self._indices_boundary(lambda i, j: j == self.m)

    def ddx(self, u, i, j):
        if i == 0:
            return (-3*u[i,j] + 4*u[i+1,j] - u[i+2, j]) / 2 / self.dx
        if i == self.n:
            return (3*u[i,j] - 4*u[i-1,j] + u[i-2, j]) / 2 / self.dx
        return (u[i+1,j] - u[i-1, j]) / self.dx / 2

    def ddy(self, u, i, j):
        if j == 0:
            return (-3*u[i,j] + 4*u[i,j+1] - u[i, j+2]) / 2 / self.dx
        if j == self.m:
            return (3*u[i,j] - 4*u[i,j-1] + u[i, j-2]) / 2 / self.dx
        return (u[i,j+1] - u[i, j-1]) / self.dy / 2

    def d2dx2(self, u, i, j):
        if i == 0 or i == self.n:
            raise NotImplementedError()
        return (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / self.dx / self.dx

    def d2dy2(self, u, i, j):
        if j == 0 or j == self.m:
            raise NotImplementedError()
        return (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / self.dy / self.dy

    def d2dxdy(self, u, i, j):
        if i == 0 or i == self.n:
            raise NotImplementedError()
        if j == 0 or j == self.m:
            raise NotImplementedError()
        return (u[i+1, j+1] - u[i-1, j+1] - u[i+1, j-1] + u[i-1, j-1]) / self.dx / self.dy / 4.0

    def fill_implicit_matrix_d2dx2(self, A, u, dt):
        for j in range(self.m+1):
            for i in range(self.n+1):
                k = self.index(i, j)
                A[k, :] = 0
                if not self.is_interior(i, j):
                    A[k, k] = 1
                else:
                    a = dt * u[i, j] / self.dx / self.dx
                    km1 = self.index(i-1, j)
                    kp1 = self.index(i+1, j)
                    A[k, km1] = -a
                    A[k, k  ] = 1 + 2*a
                    A[k, kp1] = -a

    def fill_implicit_matrix_d2dy2(self, A, u, dt):
        for j in range(self.m+1):
            for i in range(self.n+1):
                k = self.index(i, j)
                A[k, :] = 0
                if not self.is_interior(i, j):
                    A[k, k] = 1
                else:
                    a = dt * u[i, j] / self.dy / self.dy
                    km1 = self.index(i, j-1)
                    kp1 = self.index(i, j+1)
                    A[k, km1] = -a
                    A[k, k  ] = 1 + 2*a
                    A[k, kp1] = -a


class UnitTrapezoid(UnitSquare):
    def _allocate(self):
        x = np.linspace(0, 1, self.n+1)
        dx = x[1] - x[0]
        dy = dx * np.sqrt(3) / 2
        self.x = np.zeros((self.n+1, self.m+1))
        self.y = np.zeros((self.n+1, self.m+1))
        for j in range(self.m+1):
            n_max = self.n - j
            self.x[:n_max+1, j] = j*dx/2 + x[:n_max+1]
            y = j * dy
            self.y[:n_max+1, j] = y

    def indices_boundary_right(self):
        return self._indices_boundary(lambda i, j: i == self.n - j)

    def _set_bc_bool(self):
        self.is_bc = np.zeros(self.x.shape, dtype=bool)
        self.is_bc[:, 0] = True # bottom
        self.is_bc[0, :] = True # left
        for j in range(self.m+1):
            for i in range(self.n+1):
                end_idx = self.n - j
                # top
                if i <= end_idx:
                    if j == self.m:
                        self.is_bc[i, j] = True
                # right
                if i == end_idx:
                    self.is_bc[i, j] = True

    def _set_mask_bool(self):
        self.mask = np.zeros(self.x.shape, dtype=bool)
        for j in range(self.m+1):
            for i in range(self.n+1):
                if i > self.n - j:
                    self.mask[i, j] = True

    def _set_cell_indices(self):
        self.cells = []
        for j in range(self.m):
            for i in range(self.n):
                if not self.is_valid_gridpoint(i, j):
                    continue

                if self.is_valid_gridpoint(i+1, j) and self.is_valid_gridpoint(i, j+1):
                    self.cells.append((
                        (i, j),
                        (i+1, j),
                        (i, j+1),
                    ))

                    if self.is_valid_gridpoint(i+1, j+1):
                        self.cells.append((
                            (i+1, j+1),
                            (i, j+1),
                            (i+1, j),
                        ))

    def ddy_left(self, u, i, j):
        assert j > 0
        assert j < self.m
        assert i == 0
        if j < 4:
            a = -u[i, j-1] - 2*u[i+1, j-1] + u[i+2, j-1]
            b = 2 * (u[i, j] - u[i+1, j])
            c = 2 * u[i, j+1]
            return (a + b + c) / self.dy / 4

        # same as ddy_top
        assert not self.mask[i+2, j-4]
        assert not self.mask[i+1, j-2]
        return (3*u[i, j] - 4*u[i+1, j-2] + u[i+2, j-4]) / 4 / self.dy

    def ddy_right(self, u, i, j):
        assert j > 0
        assert j < self.m
        assert i == self.n - j
        if j < 4:
            a = -u[i+1, j-1] - 2*u[i, j-1] + u[i-1, j-1]
            b = 2 * (u[i, j] - u[i-1, j])
            c = 2 * u[i-1, j+1]
            return (a + b + c) / self.dy / 4

        # same as ddy_top
        assert not self.mask[i+2, j-4]
        assert not self.mask[i+1, j-2]
        return (3*u[i, j] - 4*u[i+1, j-2] + u[i+2, j-4]) / 4 / self.dy

    def ddy_bottom(self, u, i, j):
        assert j == 0

        if i == 0:
            a = -3*u[i, j] - 4*u[i+1, j] + u[i+2, j]
            b = 8 * u[i, j+1]
            c = -2 * u[i, j+2]
            return (a + b + c) / 4 / self.dy

        if i == self.n:
            a = -3*u[i, j] - 4*u[i-1, j] + u[i-2, j]
            b = 8 * u[i-1, j+1]
            c = -2 * u[i-2, j+2]
            return (a + b + c) / 4 / self.dy

        if i == 1 or i == self.n - 1:
            a = - (u[i-1, j] + 4*u[i, j] + u[i+1, j])
            b = 4 * (u[i-1, j+1] + u[i, j+1])
            c = -2 * u[i-1, j+2]
            return (a + b + c) / 4 / self.dy

        assert i >= 2
        assert not self.mask[i-2, j+4]
        assert not self.mask[i-1, j+2]
        return (-3*u[i, j] + 4*u[i-1, j+2] - u[i-2, j+4]) / 4 / self.dy

    def ddy_top(self, u, i, j):
        assert j == self.m
        assert not self.mask[i+2, j-4]
        assert not self.mask[i+1, j-2]
        return (3*u[i, j] - 4*u[i+1, j-2] + u[i+2, j-4]) / 4 / self.dy

    def ddx_left(self, u, i, j):
        assert i == 0
        if self.mask[i+2, j]:
            return self.ddx_top_left_corner(u, i, j)
        return (-6*u[i, j] + 8*u[i+1, j] - 2*u[i+2, j]) / 4 / self.dx

    def ddx_right(self, u, i, j):
        assert i == self.n - j
        if self.mask[i-2, j]:
            return self.ddx_top_right_corner(u, i, j)
        return (6*u[i, j] - 8*u[i-1, j] + 2*u[i-2, j]) / 4 / self.dx

    def ddx_center(self, u, i, j):
        assert i > 0
        assert i < self.n - j
        assert not self.mask[i+1, j]
        assert not self.mask[i-1, j]
        return (u[i+1, j] - u[i-1, j]) / self.dx / 2

    def ddx_bottom(self, u, i, j):
        assert j == 0
        if i == 0:
            return self.ddx_left(u, i, j)
        elif i == self.n:
            return self.ddx_right(u, i, j)
        return self.ddx_center(u, i, j)

    def ddx_top_corner(self, u, i, j):
        assert i == 0
        assert j == self.m
        assert self.m == self.n
        a = 8 * (u[i+1, j-1] - u[i, j-1])
        b = 2 * (u[i, j-2] - u[i+2, j-2])
        return (a + b) / 4 / self.dx

    def ddx_top_left_corner(self, u, i, j):
        assert i == 0
        assert j >= self.m - 1
        assert self.n >= self.m - 1
        a = 4 * (u[i+1, j] - u[i, j])
        b = 2 * (-u[i, j-1] + 2*u[i+1, j-1] - u[i+2, j-1])
        return (a + b) / 4 / self.dx

    def ddx_top_right_corner(self, u, i, j):
        assert i == self.m - j
        assert j >= self.m - 1
        assert self.n >= self.m - 1
        a = 4 * (u[i, j] - u[i-1, j])
        b = 2 * (u[i+1, j-1] - 2*u[i, j-1] + u[i-1, j-1])
        return (a + b) / 4 / self.dx

    def ddx_top(self, u, i, j):
        assert j == self.m
        if self.m == self.n:
            return self.ddx_top_corner(u, i, j)
        elif i == 0:
            if self.m == self.n - 1:
                return self.ddx_top_left_corner(u, i, j)
            else:
                return self.ddx_left(u, i, j)
        elif i == self.n - j:
            if self.m == self.n - 1:
                return self.ddx_top_right_corner(u, i, j)
            else:
                return self.ddx_right(u, i, j)
        return self.ddx_center(u, i, j)

    def ddx(self, u, i, j):
        if not self.is_valid_gridpoint(i, j):
            return 0.0

        # Schemes if ON a BC
        if self.is_on_bc(i, j):
            if j == 0:
                return self.ddx_bottom(u, i, j)
            if j == self.m:
                return self.ddx_top(u, i, j)
            if i == 0:
                return self.ddx_left(u, i, j)
            if i == self.n - j:
                return self.ddx_right(u, i, j)
            sys.exit(f'case not included for ddy bc {i=}, {j=}')

        return self.ddx_center(u, i, j)

    def ddy(self, u, i, j):
        if not self.is_valid_gridpoint(i, j):
            return 0.0

        # Schemes if ON a BC
        if self.is_on_bc(i, j):
            if j == 0:
                return self.ddy_bottom(u, i, j)
            if j == self.m:
                return self.ddy_top(u, i, j)
            if i == 0:
                return self.ddy_left(u, i, j)
            if i == self.n - j:
                return self.ddy_right(u, i, j)
            sys.exit(f'case not included for ddy bc {i=}, {j=}')

        # Schemes if NEAR a BC
        if j == 1 or j == self.m - 1:
            a = - (u[i, j-1] + u[i+1, j-1])
            b =   (u[i-1, j+1] + u[i, j+1])
            return (a + b) / self.dy / 4

        assert not self.mask[i-1, j+2]
        assert not self.mask[i+1, j-2]
        return (u[i-1, j+2] - u[i+1, j-2]) / self.dy / 4

    def d2dx2(self, u, i, j):
        # this never needs to be calculated on BCs
        assert i > 0
        assert i < self.n - j
        assert j > 0
        assert j < self.m
        return (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / self.dx / self.dx

    def d2dy2(self, u, i, j):
        # this never needs to be calculated on BCs
        assert i > 0
        assert i < self.n - j
        assert j > 0
        assert j < self.m

        # near BCs (i.e. j+-2 doesn't exist)
        if j == 1:
            a = 0.5 * (u[i, j-1] + u[i+1, j-1])
            b = - u[i, j]
            c = -0.5 * (u[i-1, j+1] + u[i, j+1])
            d = u[i-1, j+2]
            return (a + b + c + d) / self.dy / self.dy / 4
        if j == self.m - 1:
            a = 0.5 * (u[i-1, j+1] + u[i, j+1])
            b = - u[i, j]
            c = -0.5 * (u[i, j-1] + u[i+1, j-1])
            d = u[i+1, j-2]
            return (a + b + c + d) / self.dy / self.dy / 4

        return (u[i+1, j-2] - 2*u[i, j] + u[i-1, j+2]) / self.dy / self.dy / 4

    def d2dxdy(self, u, i, j):
        # this never needs to be calculated on BCs
        assert i > 0
        assert i < self.n - j
        assert j > 0
        assert j < self.m
        # same scheme near BCs and away from BCs
        a = 2 * (u[i, j+1] - u[i-1, j+1])
        b = 2 * (u[i, j-1] - u[i+1, j-1])
        return (a + b) / self.dx / self.dy / 4

    def fill_implicit_matrix_d2dy2(self, A, u, dt):
        for j in range(self.m+1):
            for i in range(self.n+1):
                k = self.index(i, j)
                A[k, :] = 0
                if not self.is_interior(i, j):
                    A[k, k] = 1
                elif j == 1:
                    a = dt * u[i, j] / 4 / self.dy / self.dy
                    A[k, k] = 1 + a
                    kl = self.index(i-1, j+2)
                    A[k, kl] = -a
                    kl = self.index(i, j-1)
                    A[k, kl] = -a / 2
                    kl = self.index(i+1, j-1)
                    A[k, kl] = -a / 2
                    kl = self.index(i-1, j+1)
                    A[k, kl] = a / 2
                    kl = self.index(i, j+1)
                    A[k, kl] = a / 2
                else:
                    a = dt * u[i, j] / 4 / self.dy / self.dy
                    km1 = self.index(i+1, j-2)
                    kp1 = self.index(i-1, j+2)
                    A[k, km1] = -a
                    A[k, k  ] = 1 + 2*a
                    A[k, kp1] = -a


class UnitTriangle(UnitTrapezoid):
    def __init__(self, n):
        super().__init__(n, n)

    def indices_boundary_top(self):
        return []

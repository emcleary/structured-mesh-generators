import numpy as np
import pickle
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from domain import Domain, UnitTriangle
from boundaries import Boundary


class Transform:
    def __init__(self, logical_domain, state):
        self.logical_domain = logical_domain
        self.state = state

        # Intermediate transform
        self.xi0 = None
        self.xi1 = None
        self._allocate_xi()

        # Boundaries (add later with set methods)
        self._boundary_bottom = None
        self._boundary_top = None
        self._boundary_left = None
        self._boundary_right = None

    def _allocate_xi(self):
        self.xi0 = self.logical_domain.x
        self.xi1 = self.logical_domain.y

    # fill state.s, state.v with appropriate algorithm
    def transform(self):
        raise NotImplemented()

    def is_compatible(self, transform):
        try:
            # Check type (square, triangle, trapezoid), and bools (mask, bc)
            assert self.logical_domain.is_compatible(transform.logical_domain)
        except:
            return False
        return True

    def initialize_with_transform(self, transform):
        self.state.copy(transform.state)
        self.boundary_bottom = transform.boundary_bottom
        self.boundary_top = transform.boundary_top
        self.boundary_left = transform.boundary_left
        self.boundary_right = transform.boundary_right

    def initialize_with_singularity(self, s, v):
        self._initialize_with_boundaries()
        nx, ny = self.logical_domain.shape
        for j in range(ny):
            for i in range(nx):
                if self.logical_domain.is_interior(i, j):
                    self.state.s[i, j] = s
                    self.state.v[i, j] = v

    def _initialize_with_boundaries(self):
        self._check_boundaries()

        nx, ny = self.logical_domain.shape

        t_x = np.linspace(0, 1, nx, endpoint=True)
        for k, (i, j) in enumerate(self.logical_domain.indices_boundary_bottom()):
            s, v = self.boundary_bottom.evaluate(t_x[k])
            self.state.s[i, j] = s
            self.state.v[i, j] = v
        assert t_x[k] == 1

        # this will automatically get skipped for triangular domain
        # indices should be empty list []
        for k, (i, j) in enumerate(self.logical_domain.indices_boundary_top()):
            s, v = self.boundary_top.evaluate(t_x[k])
            self.state.s[i, j] = s
            self.state.v[i, j] = v

        t_y = np.linspace(0, 1, ny, endpoint=True)
        for k, (i, j) in enumerate(self.logical_domain.indices_boundary_left()):
            s, v = self.boundary_left.evaluate(t_y[k])
            self.state.s[i, j] = s
            self.state.v[i, j] = v
        assert t_y[k] == 1

        # this will automatically get skipped for triangular domain
        # indices should be empty list []
        for k, (i, j) in enumerate(self.logical_domain.indices_boundary_right()):
            s, v = self.boundary_right.evaluate(t_y[k])
            self.state.s[i, j] = s
            self.state.v[i, j] = v
        assert t_y[k] == 1

    @property
    def boundary_bottom(self):
        return self._boundary_bottom

    @boundary_bottom.setter
    def boundary_bottom(self, boundary):
        assert isinstance(boundary, Boundary)
        self._boundary_bottom = boundary

    @property
    def boundary_top(self):
        return self._boundary_top

    @boundary_top.setter
    def boundary_top(self, boundary):
        if isinstance(self.logical_domain, UnitTriangle):
            assert boundary == None
        else:
            assert isinstance(boundary, Boundary)
        self._boundary_top = boundary

    @property
    def boundary_left(self):
        return self._boundary_left

    @boundary_left.setter
    def boundary_left(self, boundary):
        assert isinstance(boundary, Boundary)
        self._boundary_left = boundary

    @property
    def boundary_right(self):
        return self._boundary_right

    @boundary_right.setter
    def boundary_right(self, boundary):
        assert isinstance(boundary, Boundary)
        self._boundary_right = boundary

    def _check_boundaries(self):
        assert self.boundary_bottom
        assert self.boundary_left
        assert self.boundary_right
        if not isinstance(self.logical_domain, UnitTriangle):
            assert self.boundary_top
        # TODO: make sure boundary is continuous

    def dump(self, filename):
        if not filename.endswith('.pickle'):
            filename += '.pickle'

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    def load(self, filename):
        assert filename.endswith('.pickle')

        with open(filename, 'rb') as file:
            transform = pickle.load(file)

        self.initialize_with_transform(transform)

    def plot(self, filename=None):
        if filename:
            # prevent memory leak when plotting
            matplotlib.use('Agg')

        s = self.state.s
        v = self.state.v
        fig = plt.figure()

        def plot_line(idx1, idx2):
            plt.plot(
                (s[*idx1], s[*idx2]),
                (v[*idx1], v[*idx2]),
                c='k',
                linewidth=0.5
            )

        for indices in self.logical_domain.cells:
            for idx1, idx2 in zip(indices, indices[1:]):
                plot_line(idx1, idx2)
            plot_line(indices[0], indices[-1])

        plt.gca().set_aspect('equal')
        plt.axis('off')

        if filename:
            # plt.title(filename.split('.')[0])
            plt.savefig(filename)
            plt.savefig('plot_current.pdf')
        else:
            plt.show()
        plt.close()

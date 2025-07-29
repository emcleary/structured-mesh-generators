import add_src_path
import numpy as np
from diffusion import DiffusionJacobian, DiffusionCovariant
from algebraic import AlgebraicTriangle
from tensor import Tensor, TensorGradient
from boundary_lib import BoundaryLine
import sys

"""

This example solves the inverted Beltrami (diffusion) equations to
generate a triangular mesh. Contravariants are closed with a field
whose gradients affect grid cell size across the mesh.

"""


class AlgebraicExample(AlgebraicTriangle):
    def __init__(self, n):
        super(AlgebraicExample, self).__init__(n)
        h = np.sqrt(3) / 2
        self.boundary_bottom = BoundaryLine(0, 1, 0, 0, boundary_condition='dirichlet')
        self.boundary_top = None
        self.boundary_left = BoundaryLine(0, 0.5, 0, h, boundary_condition='dirichlet')
        self.boundary_right = BoundaryLine(1, 0.5, 0, h, boundary_condition='dirichlet')


class TensorExample(TensorGradient):
    def model_f(self, s1, s2):
        x = 0.5
        y = 0.28867513459481287

        phi1 = (s1-x)**2 + (s2-y)**2 - 0.03
        phi2 = (s1-x)**2 + (s2-y)**2 - 0.015
        t  = 0.05 * np.tanh(phi1 / 0.01)
        t += 0.05 * np.tanh(phi2 / 0.01)

        return t


if __name__ == '__main__':

    n = 60
    num_iter = 500
    num_plot = 100
    dt = 1e-3

    diffusion = DiffusionJacobian.factory_triangle(n, TensorExample)
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        diffusion.load(filename)
    else:
        initial_mesh = AlgebraicExample(n)
        initial_mesh.transform()
        diffusion.initialize_with_transform(initial_mesh)

    diffusion.transform(dt, num_iter, num_plot)

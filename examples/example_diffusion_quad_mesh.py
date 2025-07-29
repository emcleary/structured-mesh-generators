import add_src_path
import numpy as np
from diffusion import DiffusionJacobian
from algebraic import AlgebraicSquare
from tensor import Tensor, TensorGradient
from boundary_lib import BoundaryWaveX, BoundaryWaveY
import sys


"""

This example solves the inverted Beltrami (diffusion) equations to
generate a quadrilateral mesh. Contravariants are closed with a field
whose gradients affect grid cell size across the mesh.

"""


class AlgebraicExample(AlgebraicSquare):
    def __init__(self, n, m):
        super(AlgebraicExample, self).__init__(n, m)
        xmin = 0.0
        xmax = 1.0
        ymin = 0.0
        ymax = 1.0
        self.boundary_bottom = BoundaryWaveX(xmin, xmax, ymin, ymin,
                                             boundary_condition='dirichlet', frequency=0.5, amplitude=0.2)
        self.boundary_top = BoundaryWaveX(xmin, xmax, ymax, ymax,
                                          boundary_condition='dirichlet', frequency=0.5, amplitude=-0.2)
        self.boundary_left = BoundaryWaveY(xmin, xmin, ymin, ymax,
                                           boundary_condition='dirichlet', frequency=0.5, amplitude=0.2)
        self.boundary_right = BoundaryWaveY(xmax, xmax, ymin, ymax,
                                            boundary_condition='dirichlet', frequency=0.5, amplitude=-0.2)


class TensorExample(TensorGradient):
    def model_f(self, s1, s2):
        phi1 = (s1-0.5)**2 + (s2-0.5)**2 - 0.12
        phi2 = (s1-0.5)**2 + (s2-0.5)**2 - 0.04
        t1 = 0.2 * np.tanh(phi1 / 0.008)
        t2 = 0.05 * np.tanh(phi2 / 0.01)
        return t1 + t2


if __name__ == '__main__':

    n = 40
    m = 40
    num_iter = 400
    num_plot = 100
    dt = 1e-3

    diffusion = DiffusionJacobian.factory_square(n, m, TensorExample)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        diffusion.load(filename)
    else:
        initial_mesh = AlgebraicExample(n, m)
        initial_mesh.transform()
        diffusion.initialize_with_transform(initial_mesh)

    diffusion.transform(dt, num_iter, num_plot)

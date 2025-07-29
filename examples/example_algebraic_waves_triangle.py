import add_src_path
import numpy as np
from algebraic import AlgebraicTriangle
from boundary_lib import BoundaryLine, BoundaryWaveY


if __name__ == '__main__':
    n = 30
    m = 30

    ws = AlgebraicTriangle(n, is_deriv_on=False)
    ws.boundary_bottom = BoundaryLine(0, 1, 0, 0)
    ws.boundary_left = BoundaryWaveY(0, 0.5, 0, 1, amplitude=0.01, frequency=2)
    ws.boundary_right = BoundaryWaveY(1, 0.5, 0, 1, amplitude=0.05, frequency=2)
    ws.transform()
    ws.plot()

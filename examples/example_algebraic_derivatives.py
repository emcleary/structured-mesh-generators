import numpy as np
import add_src_path
from algebraic import AlgebraicSquare, AlgebraicTrapezoid, AlgebraicTriangle
from boundaries import Boundary, BoundaryWithDerivatives

"""
Modeling derivatives can be important in algebraic mesh generators.
These examples are great for demonstrating their importance.

These examples generate simple uniform meshes, which have analytical
expressions for derivatives needed.
"""


class BoundaryCircleFixedRadius(Boundary):
    """xi0, xi1 denote the logical domain
    radius, theta denote the intermediate domain
    x, y denote the transformed domain
    t denotes the parameter in parametric functions

    The
    Derivatives of radius and theta are constants, i.e. radius(xi) and
    theta(xi) are linear parametric functions on the boundaries.
    
    this example will not work properly on non-uniform grids

    """
    def __init__(self, radius_min, radius_max, theta_min, theta_max, fixed):
        super(BoundaryCircleFixedRadius, self).__init__()
        self.dradius_dxi0 = 0
        self.dradius_dxi1 = -(radius_max - radius_min)
        self.dtheta_dxi0 = 2 * np.pi * (theta_max - theta_min)
        self.dtheta_dxi1 = 0
        self.radius_min = None
        self.radius_max = None
        self.delta_radius = None
        self.theta_min = None
        self.theta_max = None
        self.delta_theta = None
        self._initialize(radius_min, radius_max, theta_min, theta_max, fixed)

    def _initialize(self, radius_min, radius_max, theta_min, theta_max, fixed_radius):
        self.radius_min = fixed_radius
        self.radius_max = fixed_radius
        self.delta_radius = -self.dradius_dxi0
        self.theta_min = 2 * np.pi * theta_min
        self.theta_max = 2 * np.pi * theta_max
        self.delta_theta = self.dtheta_dxi0

    def evaluate(self, t):
        radius = self._calculate_radius(t)
        theta = self._calculate_theta(t)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return x, y

    def evaluate_derivative_ddxi0(self, t):
        dx_dtheta, dy_dtheta = self._ddtheta(t)
        dx_dradius, dy_dradius = self._ddradius(t)
        dx_dxi0 = dx_dtheta*self.dtheta_dxi0 + dx_dradius*self.dradius_dxi0
        dy_dxi0 = dy_dtheta*self.dtheta_dxi0 + dy_dradius*self.dradius_dxi0
        return dx_dxi0, dy_dxi0

    def evaluate_derivative_ddxi1(self, t):
        dx_dtheta, dy_dtheta = self._ddtheta(t)
        dx_dradius, dy_dradius = self._ddradius(t)
        dx_dxi1 = dx_dradius*self.dradius_dxi1 + dx_dtheta*self.dtheta_dxi1
        dy_dxi1 = dy_dradius*self.dradius_dxi1 + dy_dtheta*self.dtheta_dxi1
        return dx_dxi1, dy_dxi1

    def _calculate_radius(self, t):
        return self.radius_min + self.delta_radius*(1-t)

    def _calculate_theta(self, t):
        return self.theta_min + self.delta_theta*t

    def _ddtheta(self, t):
        radius = self._calculate_radius(t)
        theta = self._calculate_theta(t)
        dx_dtheta = - radius * np.sin(theta)
        dy_dtheta = radius * np.cos(theta)
        return dx_dtheta, dy_dtheta

    def _ddradius(self, t):
        radius = self._calculate_radius(t)
        theta = self._calculate_theta(t)
        dx_dradius = np.cos(theta)
        dy_dradius = np.sin(theta)
        return dx_dradius, dy_dradius


class BoundaryCircleFixedTheta(BoundaryCircleFixedRadius):
    def _initialize(self, radius_min, radius_max, theta_min, theta_max, fixed_theta):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.delta_radius = -self.dradius_dxi1
        self.theta_min = 2 * np.pi * fixed_theta
        self.theta_max = 2 * np.pi * fixed_theta
        self.delta_theta = self.dtheta_dxi1


if __name__ == '__main__':

    n_xi0 = 20
    n_xi1 = 5

    radius_min = 0.5
    radius_max = 1
    theta_min = 0
    theta_max = 0.75

    cs = AlgebraicSquare(n_xi0, n_xi1, is_deriv_on=True)
    cs.boundary_bottom = BoundaryCircleFixedRadius(radius_min, radius_max, theta_min, theta_max, radius_max)
    cs.boundary_top = BoundaryCircleFixedRadius(radius_min, radius_max, theta_min, theta_max, radius_min)
    cs.boundary_left = BoundaryCircleFixedTheta(radius_min, radius_max, theta_min, theta_max, theta_min)
    cs.boundary_right = BoundaryCircleFixedTheta(radius_min, radius_max, theta_min, theta_max, theta_max)
    cs.transform()
    cs.plot()

    cs = AlgebraicTrapezoid(n_xi0, n_xi1, is_deriv_on=True)
    cs.boundary_bottom = BoundaryCircleFixedRadius(radius_min, radius_max, theta_min, theta_max, radius_max)
    cs.boundary_top = BoundaryCircleFixedRadius(radius_min, radius_max, theta_min, theta_max, radius_min)
    cs.boundary_left = BoundaryCircleFixedTheta(radius_min, radius_max, theta_min, theta_max, theta_min)
    cs.boundary_right = BoundaryCircleFixedTheta(radius_min, radius_max, theta_min, theta_max, theta_max)
    cs.transform()
    cs.plot()

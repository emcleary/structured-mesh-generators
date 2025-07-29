import numpy as np
import scipy.optimize as opt


class Boundary:
    def __init__(self, tmin=0.0, tmax=1.0):
        self.tmin = tmin
        self.tmax = tmax

    # calculate parameter from transformed grid
    # do this numerically by default in case a analytical transform does not exist
    # will need this for neuman boundary conditions
    def get_parameter(self, s0, s1):
        def objective(t):
            s0_t, s1_t = self.evaluate(t)
            ds0 = s0_t - s0
            ds1 = s1_t - s1
            return ds0*ds0 + ds1*ds1

        t0 = (self.tmin + self.tmax) / 2
        return opt.minimize(objective, t0).x[0]

    # s0, s1 = f(t)
    def evaluate(self, t):
        raise NotImplementedError()


class BoundaryWithDerivatives(Boundary):
    def evaluate_derivative_d_dxi0(self, t):
        raise NotImplementedError()

    def evaluate_derivative_d_dxi1(self, t):
        raise NotImplementedError()


# BCs don't apply to algebraic interpolation; keep them separate
class BoundaryWithCondition(Boundary):
    valid_boundary_conditions = ['dirichlet', 'neumann', 'normal']

    def __init__(self, tmin=0.0, tmax=1.0, boundary_condition='dirichlet'):
        super().__init__(tmin, tmax)
        assert boundary_condition in self.valid_boundary_conditions
        self.boundary_condition = boundary_condition

    def apply_boundary_condition(self, i, j, state, *args, **kwargs):
        if self.boundary_condition == 'dirichlet':
            return self._dirichlet_boundary_condition(i, j, state)
        elif self.boundary_condition == 'normal':
            return self._normal_boundary_condition(i, j, state, *args, **kwargs)
        elif self.boundary_condition == 'neumann':
            return self._neumann_boundary_condition(i, j, state, *args, **kwargs)
        else:
            raise NotImplementedError(f'Boundary condition {self.boundary_condition} has not been implemented.')

    def _dirichlet_boundary_condition(self, i, j, state, *args, **kwargs):
        s = state.s[i, j]
        v = state.v[i, j]
        return s, v

    def _normal_boundary_condition(self, i, j, state, *args, **kwargs):
        s0 = state.s[i, j]
        v0 = state.v[i, j]
        t0 = self.get_parameter(s0, v0)
        atol = 1e-6
        if np.isclose(t0, 0, atol) or np.isclose(t0, 1, atol):
            return s0, v0

        # to be minimized
        # ds/dxi0 * ds/dxi1 will be 0 if orthogonal
        # since xi0 and xi1 are orthogonal
        def objective(t):
            s, v = self.evaluate(t)
            state.s[i, j] = s
            state.v[i, j] = v
            dsdx = state.calculate_dsdx(i, j)
            dvdx = state.calculate_dvdx(i, j)
            dsdy = state.calculate_dsdy(i, j)
            dvdy = state.calculate_dvdy(i, j)

            a = dsdx*dsdy + dvdx*dvdy
            return a * a

        tf = opt.minimize(objective, t0).x[0]

        state.s[i, j] = s0
        state.v[i, j] = v0

        return self.evaluate(tf)

    def _neumann_boundary_condition(self, i, j, state, *args, **kwargs):
        raise NotImplementedError()


class BoundaryGradient(BoundaryWithCondition):
    def _neumann_boundary_condition(self, i, j, state, *args, **kwargs):
        raise NotImplementedError('TODO')

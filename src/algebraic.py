import numpy as np
from transform import Transform
from domain import UnitSquare, UnitTrapezoid, UnitTriangle
from state import State


class Algebraic(Transform):
    def __init__(self, logical_domain, state, is_deriv_on=False):
        super(Algebraic, self).__init__(logical_domain, state)
        self.eps = 1e-8
        self._is_deriv_on = is_deriv_on

    def transform(self):
        self._check_boundaries()
        self._set_boundaries()
        self._fill_grid()

    def _function_bottom(self, xi0, xi1):
        assert np.isclose(xi1, 0.0)
        return self.boundary_bottom.evaluate(xi0)

    def _function_top(self, xi0, xi1):
        assert np.isclose(xi1, 1.0)
        return self.boundary_top.evaluate(xi0)

    def _function_left(self, xi0, xi1):
        assert np.isclose(xi0, 0.0)
        return self.boundary_left.evaluate(xi1)

    def _function_right(self, xi0, xi1):
        assert np.isclose(xi0, 1.0)
        return self.boundary_right.evaluate(xi1)

    def _derivative_bottom(self, xi0, xi1, direction_index):
        assert np.isclose(xi1, 0.0)
        t = xi0
        if direction_index == 0:
            return self.boundary_bottom.evaluate_derivative_ddxi0(t)
        elif direction_index == 1:
            return self.boundary_bottom.evaluate_derivative_ddxi1(t)
        else:
            raise NotImplementedError(f'index {direction_index} not implemented')

    def _derivative_top(self, xi0, xi1, direction_index):
        assert np.isclose(xi1, 1.0)
        t = xi0
        if direction_index == 0:
            return self.boundary_top.evaluate_derivative_ddxi0(t)
        elif direction_index == 1:
            return self.boundary_top.evaluate_derivative_ddxi1(t)
        else:
            raise NotImplementedError(f'index {direction_index} not implemented')

    def _derivative_left(self, xi0, xi1, direction_index):
        assert np.isclose(xi0, 0.0)
        t = xi1
        if direction_index == 0:
            return self.boundary_left.evaluate_derivative_ddxi0(t)
        elif direction_index == 1:
            return self.boundary_left.evaluate_derivative_ddxi1(t)
        else:
            raise NotImplementedError(f'index {direction_index} not implemented')

    def _derivative_right(self, xi0, xi1, direction_index):
        assert np.isclose(xi0, 1.0)
        t = xi1
        if direction_index == 0:
            return self.boundary_right.evaluate_derivative_ddxi0(t)
        elif direction_index == 1:
            return self.boundary_right.evaluate_derivative_ddxi1(t)
        else:
            raise NotImplementedError(f'index {direction_index} not implemented')

    def _evaluate_function(self, xi0, xi1):
        raise NotImplementedError()

    def _evaluate_derivative(self, xi0, xi1, direction_index):
        raise NotImplementedError()

    def _function_weights(self, xi):
        xi0 = 0.0
        xi1 = 1.0
        r0 = (xi - xi0) / (xi0 - xi1)
        r1 = 1 + r0
        f0 = (1 - 2*r0) * r1 * r1
        f1 = (1 + 2*r1) * r0 * r0
        return f0, f1

    def _derivative_weights(self, xi):
        assert self._is_deriv_on
        xi0 = 0.0
        xi1 = 1.0
        r0 = (xi - xi0) / (xi0 - xi1)
        r1 = 1 + r0
        d0 = (xi - xi0) * r1 * r1
        d1 = (xi - xi1) * r0 * r0
        return d0, d1

    def _differentiate(self, xi0, xi1, f, direction_index):
        if direction_index == 0:
            f0 = f(xi0 + self.eps, xi1)
            f1 = f(xi0 - self.eps, xi1)
        elif direction_index == 1:
            f0 = f(xi0, xi1 + self.eps)
            f1 = f(xi0, xi1 - self.eps)
        else:
            sys.exit('direction_index must be less than 2')
        return (f0 - f1) / 2 / self.eps

    def _F0(self, xi0, xi1):
        f0, f1 = self._function_weights(xi0)
        f = f0 * self._evaluate_function(0, xi1)
        f += f1 * self._evaluate_function(1, xi1)

        if self._is_deriv_on:
            d0, d1 = self._derivative_weights(xi0)
            f += d0 * self._evaluate_derivative(0, xi1, 0)
            f += d1 * self._evaluate_derivative(1, xi1, 0)

        return f

    def _F1(self, xi0, xi1):
        f = self._F0(xi0, xi1)

        f0, f1 = self._function_weights(xi1)
        f += f0 * (self._evaluate_function(xi0, 0) - self._F0(xi0, 0))
        f += f1 * (self._evaluate_function(xi0, 1) - self._F0(xi0, 1))

        if self._is_deriv_on:
            d0, d1 = self._derivative_weights(xi1)
            f += d0 * (self._evaluate_derivative(xi0, 0, 1) - self._differentiate(xi0, 0, self._F0, 1))
            f += d1 * (self._evaluate_derivative(xi0, 1, 1) - self._differentiate(xi0, 1, self._F0, 1))

        return f

    def _fill_grid(self):
        for j in range(self.logical_domain.m+1):
            for i in range(self.logical_domain.n+1):
                if not self.logical_domain.is_interior(i, j):
                    continue
                xi0 = self.xi0[i, j]
                xi1 = self.xi1[i, j]
                s0, s1 = self._evaluate_function(xi0, xi1)
                self.state.s[i, j] = s0
                self.state.v[i, j] = s1

    def _set_boundaries(self):
        for j in range(self.logical_domain.m+1):
            for i in range(self.logical_domain.n+1):
                if self.logical_domain.is_on_bc(i, j):
                    xi0 = self.xi0[i, j]
                    xi1 = self.xi1[i, j]
                    s0, s1 = self._evaluate_function(xi0, xi1)
                    self.state.s[i, j] = s0
                    self.state.v[i, j] = s1


class AlgebraicSquare(Algebraic):
    def __init__(self, n, m, is_deriv_on=False):
        logical_domain = UnitSquare(n, m)
        state = State(logical_domain)
        super(AlgebraicSquare, self).__init__(logical_domain, state, is_deriv_on)

    def _evaluate_function(self, xi0, xi1):
        if xi0 == 0:
            r = self._function_left(xi0, xi1)
        elif xi0 == 1:
            r = self._function_right(xi0, xi1)
        elif xi1 == 0:
            r = self._function_bottom(xi0, xi1)
        elif xi1 == 1:
            r = self._function_top(xi0, xi1)
        else:
            r = self._F1(xi0, xi1)
        return np.asarray(r)

    def _evaluate_derivative(self, xi0, xi1, direction_index):
        assert 0 <= direction_index < 2
        if xi0 == 0:
            r = self._derivative_left(xi0, xi1, direction_index)
        elif xi0 == 1:
            r = self._derivative_right(xi0, xi1, direction_index)
        elif xi1 == 0:
            r = self._derivative_bottom(xi0, xi1, direction_index)
        elif xi1 == 1:
            r = self._derivative_top(xi0, xi1, direction_index)
        else:
            sys.exit('Only differentiate on boundaries')
        return np.asarray(r)


class AlgebraicTrapezoid(AlgebraicSquare):
    def __init__(self, n, m, is_deriv_on=False):
        logical_domain = UnitTrapezoid(n, m)
        state = State(logical_domain)
        super(AlgebraicSquare, self).__init__(logical_domain, state, is_deriv_on)

    def _allocate_xi(self):
        y_max = self.logical_domain.y[0, -1]
        self.xi0 = np.zeros(self.state.s.shape)
        self.xi1 = np.zeros(self.state.v.shape)
        for j in range(self.logical_domain.m+1):
            n = self.logical_domain.n - j
            x = np.copy(self.logical_domain.x[:n+1, j])
            x -= x[0]
            x_max = x[-1]
            if x_max == 0.0:
                x_max = 1.0
            self.xi0[:n+1, j] = x / x_max
            self.xi1[:n+1, j] = self.logical_domain.y[:n+1, j] / y_max


class AlgebraicTriangle(AlgebraicTrapezoid):
    def __init__(self, n, is_deriv_on=False):
        logical_domain = UnitTriangle(n)
        state = State(logical_domain)
        super(AlgebraicSquare, self).__init__(logical_domain, state, is_deriv_on)

    def _evaluate_function(self, xi0, xi1):
        if xi0 == 0:
            r = self._function_left(xi0, xi1)
        elif xi0 == 1:
            r = self._function_right(xi0, xi1)
        elif xi1 == 0:
            r = self._function_bottom(xi0, xi1)
        elif xi1 == 1:
            r = self._function_left(0, 1)
        else:
            r = self._F1(xi0, xi1)
        return np.asarray(r)

    def _evaluate_derivative(self, xi0, xi1, direction_index):
        assert 0 <= direction_index < 2
        if xi0 == 0:
            r = self._derivative_left(xi0, xi1, direction_index)
        elif xi0 == 1:
            r = self._derivative_right(xi0, xi1, direction_index)
        elif xi1 == 0:
            r = self._derivative_bottom(xi0, xi1, direction_index)
        elif xi1 == 1:
            r = self._derivative_left(0, 1, direction_index)
        else:
            sys.exit('Only differentiate on boundaries')
        return np.asarray(r)

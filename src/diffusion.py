import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
import time
from utility import timer
from transform import Transform
from domain import UnitSquare, UnitTrapezoid, UnitTriangle
from state import StateWithDerivatives

# Linalg solvers are significantly faster with just numpy arrays
# Slower when using lil_matrix, etc. for saving memory
import scipy.sparse as sp

# TODO: vectorize all math

class Diffusion(Transform):

    @classmethod
    def factory_square(cls, n, m, tensor_class):
        domain = UnitSquare(n, m)
        state = StateWithDerivatives(domain)
        tensor = tensor_class(state)
        return cls(domain, state, tensor)

    @classmethod
    def factory_trapezoid(cls, n, m, tensor_class):
        domain = UnitTrapezoid(n, m)
        state = StateWithDerivatives(domain)
        tensor = tensor_class(state)
        return cls(domain, state, tensor)

    @classmethod
    def factory_triangle(cls, n, tensor_class):
        domain = UnitTriangle(n)
        state = StateWithDerivatives(domain)
        tensor = tensor_class(state)
        return cls(domain, state, tensor)

    def __init__(self, domain, state, tensor):
        assert domain == state.domain
        super(Diffusion, self).__init__(domain, state)
        self.tensor = tensor
        self.iteration = 0
        self.state.s[:] = self.logical_domain.x[:]
        self.state.v[:] = self.logical_domain.y[:]
        self._allocate()

    def transform(self, dt, num_iter, plot_iter=None):
        self._check_boundaries()
        self.plot(f'plot_{self.iteration}.pdf')
        self._initialize_numerics_arrays()
        while self.iteration < num_iter:
            self.iteration += 1
            print('iteration', self.iteration)
            self._advance_timestep(dt)
            if plot_iter and self.iteration % plot_iter == 0:
                self.plot(f'plot_{self.iteration}.pdf')
                self.dump(f'data_{self.iteration}.pickle')

    def initialize_with_transform(self, transform):
        super(Diffusion, self).initialize_with_transform(transform)
        if isinstance(transform, type(self)):
            self.iteration = transform.iteration

    # Initial condition editing is less error prone if half/np1 arrays are managed at the start of iterating, rather than done manually
    def _initialize_numerics_arrays(self):
        self.s_half[:] = self.state.s[:]
        self.v_half[:] = self.state.v[:]
        self.snp1[:] = self.state.s[:]
        self.vnp1[:] = self.state.v[:]

    def _advance_timestep(self, dt):
        self.state.calculate_derivatives()
        self.tensor.calculate_tensor()
        self._calculate_numerics()
        self._solve_x_dir(dt)
        self._solve_y_dir(dt)
        self._swap()
        self._update_bcs()
        self._difference()

    def _calculate_numerics(self):
        raise NotImplementedError

    def _solve_x_dir(self, dt):
        raise NotImplementedError

    def _solve_y_dir(self, dt):
        raise NotImplementedError

    def _update_bcs(self):
        raise NotImplementedError()

    def _swap(self):
        self.state.s, self.snp1 = self.snp1, self.state.s
        self.state.v, self.vnp1 = self.vnp1, self.state.v

    def _difference(self):
        diff_x = np.sum(np.abs((self.state.s[1:-1, 1:-1] - self.snp1[1:-1, 1:-1]).ravel()))
        diff_y = np.sum(np.abs((self.state.v[1:-1, 1:-1] - self.vnp1[1:-1, 1:-1]).ravel()))
        print('  ', diff_x)
        print('  ', diff_y)

    def _allocate(self):
        self.s_half = np.zeros(self.logical_domain.shape)
        self.v_half = np.zeros(self.logical_domain.shape)
        self.snp1 = np.zeros(self.logical_domain.shape)
        self.vnp1 = np.zeros(self.logical_domain.shape)
        self.g_xi_cov = np.zeros((*self.logical_domain.shape, 2, 2))
        self.gs = np.zeros(self.logical_domain.shape)
        self.ws = np.zeros(self.logical_domain.shape)
        self.ws_g11 = np.zeros(self.logical_domain.shape)
        self.ws_g12 = np.zeros(self.logical_domain.shape)
        self.ws_g22 = np.zeros(self.logical_domain.shape)
        self.L11 = np.zeros(self.logical_domain.shape)
        self.L12 = np.zeros(self.logical_domain.shape)
        self.L22 = np.zeros(self.logical_domain.shape)
        self.F1 = np.zeros(self.logical_domain.shape)
        self.F2 = np.zeros(self.logical_domain.shape)
        size = self.state.s.size
        self.A = np.zeros((size, size))
        self.Fx = np.zeros(size)
        self.Fy = np.zeros(size)

    # Beltrami
    def model_weight(self, i, j):
        g_s = self.tensor.det_contra[i, j]
        assert g_s > 0, "weight must be positive"
        return 1 / np.sqrt(g_s)

    # TODO: find better way to skip endpoints
    # doing it inside the condition is difficult due to optimization error
    @timer
    def _update_bcs(self):
        nx, ny = self.logical_domain.shape

        sv_opt = {}
        for k, (i, j) in enumerate(self.logical_domain.indices_boundary_bottom()):
            if i == 0 and j == 0: continue
            if i == nx and j == 0: continue
            sv_opt[(i, j)] = self.boundary_bottom.apply_boundary_condition(i, j, self.state, self.tensor, derivative='xi1')

        for i, j in self.logical_domain.indices_boundary_top():
            if i == 0 and j == ny: continue
            if i == nx and j == ny: continue
            sv_opt[(i, j)] = self.boundary_top.apply_boundary_condition(i, j, self.state, self.tensor, derivative='xi1')

        for i, j in self.logical_domain.indices_boundary_left():
            if i == 0 and j == 0: continue
            if i == 0 and j == ny: continue
            sv_opt[(i, j)] = self.boundary_left.apply_boundary_condition(i, j, self.state, self.tensor, derivative='xi0')

        for i, j in self.logical_domain.indices_boundary_right():
            if i == nx and j == 0: continue
            if i == nx and j == ny: continue
            sv_opt[(i, j)] = self.boundary_right.apply_boundary_condition(i, j, self.state, self.tensor, derivative='xi0')

        for idx, (s, v) in sv_opt.items():
            self.state.s[*idx] = s
            self.state.v[*idx] = v


class DiffusionCovariant(Diffusion):
    @timer
    def _calculate_numerics(self):
        for j in range(0, self.logical_domain.m+1):
            for i in range(0, self.logical_domain.n+1):
                if self.logical_domain.is_valid_gridpoint(i, j):
                    g_ij_s = self.tensor.contra[i, j]
                    self.gs[i, j] = self.tensor.det_contra[i, j]
                    self.ws[i, j] = self.model_weight(i, j)
                    self.ws_g11[i, j] = self.ws[i, j] * g_ij_s[0, 0]
                    self.ws_g12[i, j] = self.ws[i, j] * g_ij_s[0, 1]
                    self.ws_g22[i, j] = self.ws[i, j] * g_ij_s[1, 1]
                    self.g_xi_cov[i, j] = self.tensor.cov_xi[i, j]

        for j in range(1, self.logical_domain.m):
            for i in range(1, self.logical_domain.n):
                if self.logical_domain.is_interior(i, j):
                    ds1_dxi1 = self.state.ds_dx[i, j]
                    ds1_dxi2 = self.state.ds_dy[i, j]
                    ds2_dxi1 = self.state.dv_dx[i, j]
                    ds2_dxi2 = self.state.dv_dy[i, j]
                    jacobian = ds1_dxi1*ds2_dxi2 - ds2_dxi1*ds1_dxi2

                    dwg11_dxi1 = self.logical_domain.ddx(self.ws_g11, i, j)
                    dwg12_dxi1 = self.logical_domain.ddx(self.ws_g12, i, j)
                    dwg21_dxi1 = self.logical_domain.ddx(self.ws_g12, i, j)
                    dwg22_dxi1 = self.logical_domain.ddx(self.ws_g22, i, j)

                    dwg11_dxi2 = self.logical_domain.ddy(self.ws_g11, i, j)
                    dwg12_dxi2 = self.logical_domain.ddy(self.ws_g12, i, j)
                    dwg21_dxi2 = self.logical_domain.ddy(self.ws_g12, i, j)
                    dwg22_dxi2 = self.logical_domain.ddy(self.ws_g22, i, j)

                    self.F1[i, j] = ds2_dxi2 * dwg11_dxi1 - ds2_dxi1 * dwg11_dxi2
                    self.F1[i, j] += ds1_dxi1 * dwg12_dxi2 - ds1_dxi2 * dwg12_dxi1
                    self.F1[i, j] *= jacobian

                    self.F2[i, j] = ds2_dxi2 * dwg21_dxi1 - ds2_dxi1 * dwg21_dxi2
                    self.F2[i, j] += ds1_dxi1 * dwg22_dxi2 - ds1_dxi2 * dwg22_dxi1
                    self.F2[i, j] *= jacobian

                    self.L11[i, j] = self.g_xi_cov[i, j, 0, 0] * self.ws[i, j] * self.gs[i, j]
                    self.L12[i, j] = self.g_xi_cov[i, j, 0, 1] * self.ws[i, j] * self.gs[i, j]
                    self.L22[i, j] = self.g_xi_cov[i, j, 1, 1] * self.ws[i, j] * self.gs[i, j]

    @timer
    def _solve_x_dir(self, dt):
        for j in range(self.logical_domain.m+1):
            for i in range(self.logical_domain.n+1):
                k = self.logical_domain.index(i, j)
                if not self.logical_domain.is_interior(i, j):
                    self.Fx[k] = self.state.s[i, j]
                    self.Fy[k] = self.state.v[i, j]
                else:
                    d2s1_dxi1dxi2 = self.state.d2s_dxdy[i, j]
                    d2s1_dxi2dxi2 = self.state.d2s_dy2[i, j]
                    d2s2_dxi1dxi2 = self.state.d2v_dxdy[i, j]
                    d2s2_dxi2dxi2 = self.state.d2v_dy2[i, j]
                    self.Fx[k] = self.state.s[i, j] + dt*(-2*self.L12[i,j]*d2s1_dxi1dxi2 + self.L11[i,j]*d2s1_dxi2dxi2 - self.F1[i,j])
                    self.Fy[k] = self.state.v[i, j] + dt*(-2*self.L12[i,j]*d2s2_dxi1dxi2 + self.L11[i,j]*d2s2_dxi2dxi2 - self.F2[i,j])

        self.logical_domain.fill_implicit_matrix_d2dx2(self.A, self.L22, dt)
        self.s_half[:, :] = sp.linalg.spsolve(self.A, self.Fx).reshape(self.logical_domain.shape)
        self.v_half[:, :] = sp.linalg.spsolve(self.A, self.Fy).reshape(self.logical_domain.shape)

    @timer
    def _solve_y_dir(self, dt):
        for j in range(self.logical_domain.m+1):
            for i in range(self.logical_domain.n+1):
                k = self.logical_domain.index(i, j)
                if not self.logical_domain.is_interior(i, j):
                    self.Fx[k] = self.state.s[i, j]
                    self.Fy[k] = self.state.v[i, j]
                else:
                    d2s1_dxi2dxi2 = self.state.d2s_dy2[i, j]
                    d2s2_dxi2dxi2 = self.state.d2v_dy2[i, j]
                    self.Fx[k] = self.s_half[i, j] - dt*self.L11[i,j]*d2s1_dxi2dxi2
                    self.Fy[k] = self.v_half[i, j] - dt*self.L11[i,j]*d2s2_dxi2dxi2

        self.logical_domain.fill_implicit_matrix_d2dy2(self.A, self.L11, dt)
        self.snp1[:, :] = sp.linalg.spsolve(self.A, self.Fx).reshape(self.logical_domain.shape)
        self.vnp1[:, :] = sp.linalg.spsolve(self.A, self.Fy).reshape(self.logical_domain.shape)


class DiffusionJacobian(Diffusion):
    @timer
    def _calculate_numerics(self):
        for j in range(0, self.logical_domain.m+1):
            for i in range(0, self.logical_domain.n+1):
                if self.logical_domain.is_valid_gridpoint(i, j):
                    g_ij_s = self.tensor.contra[i, j]
                    self.gs[i, j] = self.tensor.det_contra[i, j]
                    self.ws[i, j] = self.model_weight(i, j)
                    self.ws_g11[i, j] = self.ws[i, j] * g_ij_s[0, 0]
                    self.ws_g12[i, j] = self.ws[i, j] * g_ij_s[0, 1]
                    self.ws_g22[i, j] = self.ws[i, j] * g_ij_s[1, 1]
                    self.g_xi_cov[i, j] = self.tensor.cov_xi[i, j]

        for j in range(1, self.logical_domain.m):
            for i in range(1, self.logical_domain.n):
                if self.logical_domain.is_interior(i, j):
                    ds1_dxi1 = self.state.ds_dx[i, j]
                    ds1_dxi2 = self.state.ds_dy[i, j]
                    ds2_dxi1 = self.state.dv_dx[i, j]
                    ds2_dxi2 = self.state.dv_dy[i, j]
                    jacobian = ds1_dxi1*ds2_dxi2 - ds2_dxi1*ds1_dxi2

                    dwg11_dxi1 = self.logical_domain.ddx(self.ws_g11, i, j)
                    dwg12_dxi1 = self.logical_domain.ddx(self.ws_g12, i, j)
                    dwg21_dxi1 = self.logical_domain.ddx(self.ws_g12, i, j)
                    dwg22_dxi1 = self.logical_domain.ddx(self.ws_g22, i, j)

                    dwg11_dxi2 = self.logical_domain.ddy(self.ws_g11, i, j)
                    dwg12_dxi2 = self.logical_domain.ddy(self.ws_g12, i, j)
                    dwg21_dxi2 = self.logical_domain.ddy(self.ws_g12, i, j)
                    dwg22_dxi2 = self.logical_domain.ddy(self.ws_g22, i, j)

                    self.F1[i, j] = ds2_dxi2 * dwg11_dxi1 - ds2_dxi1 * dwg11_dxi2
                    self.F1[i, j] += ds1_dxi1 * dwg12_dxi2 - ds1_dxi2 * dwg12_dxi1
                    self.F1[i, j] *= jacobian

                    self.F2[i, j] = ds2_dxi2 * dwg21_dxi1 - ds2_dxi1 * dwg21_dxi2
                    self.F2[i, j] += ds1_dxi1 * dwg22_dxi2 - ds1_dxi2 * dwg22_dxi1
                    self.F2[i, j] *= jacobian

                    J_dxi1_ds1 = ds2_dxi2
                    J_dxi1_ds2 = -ds1_dxi2
                    J_dxi2_ds1 = -ds2_dxi1
                    J_dxi2_ds2 = ds1_dxi1

                    w_g11_s = self.ws_g11[i, j]
                    w_g12_s = self.ws_g12[i, j]
                    w_g21_s = self.ws_g12[i, j]
                    w_g22_s = self.ws_g22[i, j]

                    self.L11[i, j] = w_g11_s * J_dxi1_ds1 * J_dxi1_ds1
                    self.L11[i, j] += 2 * w_g12_s * J_dxi1_ds1 * J_dxi1_ds2
                    self.L11[i, j] += w_g22_s * J_dxi1_ds2 * J_dxi1_ds2

                    self.L22[i, j] = w_g11_s * J_dxi2_ds1 * J_dxi2_ds1
                    self.L22[i, j] += 2 * w_g12_s * J_dxi2_ds1 * J_dxi2_ds2
                    self.L22[i, j] += w_g22_s * J_dxi2_ds2 * J_dxi2_ds2

                    self.L12[i, j] = w_g11_s * J_dxi1_ds1 * J_dxi2_ds1
                    self.L12[i, j] += w_g12_s * J_dxi1_ds1 * J_dxi2_ds2
                    self.L12[i, j] += w_g21_s * J_dxi1_ds2 * J_dxi2_ds1
                    self.L12[i, j] += w_g22_s * J_dxi1_ds2 * J_dxi2_ds2

    @timer
    def _solve_x_dir(self, dt):
        for j in range(self.logical_domain.m+1):
            for i in range(self.logical_domain.n+1):
                k = self.logical_domain.index(i, j)
                if not self.logical_domain.is_interior(i, j):
                    self.Fx[k] = self.state.s[i, j]
                    self.Fy[k] = self.state.v[i, j]
                else:
                    d2s1_dxi1dxi2 = self.state.d2s_dxdy[i, j]
                    d2s1_dxi2dxi2 = self.state.d2s_dy2[i, j]
                    d2s2_dxi1dxi2 = self.state.d2v_dxdy[i, j]
                    d2s2_dxi2dxi2 = self.state.d2v_dy2[i, j]
                    self.Fx[k] = self.state.s[i, j] + dt*(2*self.L12[i,j]*d2s1_dxi1dxi2 + self.L22[i,j]*d2s1_dxi2dxi2 - self.F1[i,j])
                    self.Fy[k] = self.state.v[i, j] + dt*(2*self.L12[i,j]*d2s2_dxi1dxi2 + self.L22[i,j]*d2s2_dxi2dxi2 - self.F2[i,j])

        self.logical_domain.fill_implicit_matrix_d2dx2(self.A, self.L11, dt)
        self.s_half[:, :] = sp.linalg.spsolve(self.A, self.Fx).reshape(self.logical_domain.shape)
        self.v_half[:, :] = sp.linalg.spsolve(self.A, self.Fy).reshape(self.logical_domain.shape)

    @timer
    def _solve_y_dir(self, dt):
        for j in range(self.logical_domain.m+1):
            for i in range(self.logical_domain.n+1):
                k = self.logical_domain.index(i, j)
                if not self.logical_domain.is_interior(i, j):
                    self.Fx[k] = self.state.s[i, j]
                    self.Fy[k] = self.state.v[i, j]
                else:
                    d2s1_dxi2dxi2 = self.state.d2s_dy2[i, j]
                    d2s2_dxi2dxi2 = self.state.d2v_dy2[i, j]
                    self.Fx[k] = self.s_half[i, j] - dt*self.L22[i,j]*d2s1_dxi2dxi2
                    self.Fy[k] = self.v_half[i, j] - dt*self.L22[i,j]*d2s2_dxi2dxi2

        self.logical_domain.fill_implicit_matrix_d2dy2(self.A, self.L22, dt)
        self.snp1[:, :] = sp.linalg.spsolve(self.A, self.Fx).reshape(self.logical_domain.shape)
        self.vnp1[:, :] = sp.linalg.spsolve(self.A, self.Fy).reshape(self.logical_domain.shape)

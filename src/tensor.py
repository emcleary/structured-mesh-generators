import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
import time
from utility import timer


class Tensor:
    def __init__(self, state):
        self.state = state
        self.domain = state.domain
        self.contra = np.zeros((*self.domain.shape, 2, 2))
        self.det_contra = np.zeros(self.domain.shape)
        self._model_contra = np.eye(2)
        self.cov_xi = np.zeros((*self.domain.shape, 2, 2))

    def calculate(self):
        self.calculate_contravariant()
        self.calculate_contravariant_determinant()
        self.calculate_covariant_xi()

    def calculate_covariant_xi(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_valid_gridpoint(i, j):
                    g_cov = np.linalg.inv(self.contra[i, j])
                    ds1_dxi1 = self.state.ds_dx[i, j]
                    ds1_dxi2 = self.state.ds_dy[i, j]
                    ds2_dxi1 = self.state.dv_dx[i, j]
                    ds2_dxi2 = self.state.dv_dy[i, j]

                    self.cov_xi[i, j, 0, 0] = g_cov[0, 0] * ds1_dxi1 * ds1_dxi1
                    self.cov_xi[i, j, 0, 0] += g_cov[0, 1] * ds1_dxi1 * ds2_dxi1
                    self.cov_xi[i, j, 0, 0] += g_cov[1, 0] * ds2_dxi1 * ds1_dxi1
                    self.cov_xi[i, j, 0, 0] += g_cov[1, 1] * ds2_dxi1 * ds2_dxi1

                    self.cov_xi[i, j, 0, 1] = g_cov[0, 0] * ds1_dxi1 * ds1_dxi2
                    self.cov_xi[i, j, 0, 1] += g_cov[0, 1] * ds1_dxi1 * ds2_dxi2
                    self.cov_xi[i, j, 0, 1] += g_cov[1, 0] * ds2_dxi1 * ds1_dxi2
                    self.cov_xi[i, j, 0, 1] += g_cov[1, 1] * ds2_dxi1 * ds2_dxi2

                    self.cov_xi[i, j, 1, 0] = g_cov[0, 0] * ds1_dxi2 * ds1_dxi1
                    self.cov_xi[i, j, 1, 0] += g_cov[0, 1] * ds1_dxi2 * ds2_dxi1
                    self.cov_xi[i, j, 1, 0] += g_cov[1, 0] * ds2_dxi2 * ds1_dxi1
                    self.cov_xi[i, j, 1, 0] += g_cov[1, 1] * ds2_dxi2 * ds2_dxi1

                    self.cov_xi[i, j, 1, 1] = g_cov[0, 0] * ds1_dxi2 * ds1_dxi2
                    self.cov_xi[i, j, 1, 1] += g_cov[0, 1] * ds1_dxi2 * ds2_dxi2
                    self.cov_xi[i, j, 1, 1] += g_cov[1, 0] * ds2_dxi2 * ds1_dxi2
                    self.cov_xi[i, j, 1, 1] += g_cov[1, 1] * ds2_dxi2 * ds2_dxi2

                    assert np.isclose(self.cov_xi[i, j, 0, 1], self.cov_xi[i, j, 1, 0])

    def calculate_weight(self, i, j):
        return 1.0

    def model_contravariant(self, i, j):
        assert self.domain.is_valid_gridpoint(i, j)
        return self._model_contra

    def _calculate_contravariant(self, i, j):
        assert self.domain.is_valid_gridpoint(i, j)
        return self._model_contra

    def calculate_contravariant(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_valid_gridpoint(i, j):
                    weight = self.calculate_weight(i, j)
                    self.contra[i, j] = weight * self._calculate_contravariant(i, j)

    def calculate_contravariant_determinant(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_valid_gridpoint(i, j):
                    self.det_contra[i, j] = np.linalg.det(self.contra[i, j])


class TensorGradient(Tensor):
    def __init__(self, state, normal_vector_scale=0.1):
        super().__init__(state)
        self.f = np.zeros(self.domain.shape)
        self.df_ds = np.zeros(self.domain.shape)
        self.df_dv = np.zeros(self.domain.shape)

    def calculate_tensor(self):
        self.calculate_f()
        self.differentiate_f()
        self.calculate_contravariant()
        self.calculate_contravariant_determinant()
        self.calculate_covariant_xi()

    def model_f(self, s, v):
        raise NotImplementedError()

    def model_df_ds(self, s, v):
        raise NotImplementedError()

    def model_df_dv(self, s, v):
        raise NotImplementedError()

    def calculate_f(self):
        s = self.state.s
        v = self.state.v
        self.f[:] = self.model_f(s, v)

    def differentiate_f(self, eps=1e-8):
        s = self.state.s
        v = self.state.v
        self.df_ds[:] = self.calculate_df_ds(s, v, eps)
        self.df_dv[:] = self.calculate_df_dv(s, v, eps)

    def calculate_df_ds(self, s, v, eps=1e-8):
        try:
            return self.model_df_ds(s, v)
        except NotImplementedError:
            return (self.model_f(s+eps, v) - self.f) / eps

    def calculate_df_dv(self, s, v, eps=1e-8):
        try:
            return self.model_df_dv(s, v)
        except NotImplementedError:
            return (self.model_f(s, v+eps) - self.f) / eps

    def _calculate_contravariant(self, i, j):
        assert self.domain.is_valid_gridpoint(i, j)
        f = self.f[i, j]
        df_ds = self.df_ds[i, j]
        df_dv = self.df_dv[i, j]
        grad_f_norm = np.sqrt(df_ds*df_ds + df_dv*df_dv)
        c = 1 / (1 + grad_f_norm*grad_f_norm)
        g = np.eye(2)
        g[0, 0] -= c * df_ds * df_ds
        g[0, 1] -= c * df_ds * df_dv
        g[1, 0] -= c * df_dv * df_ds
        g[1, 1] -= c * df_dv * df_dv
        return g

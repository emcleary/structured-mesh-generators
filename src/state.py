import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
import time
from utility import timer


class State:
    def __init__(self, domain):
        self.domain = domain
        self.s = np.zeros(self.domain.shape)
        self.v = np.zeros(self.domain.shape)

    def copy(self, state):
        assert self.domain.is_compatible(state.domain)
        self.s[:] = state.s[:]
        self.v[:] = state.v[:]
    
class StateWithDerivatives(State):
    def __init__(self, domain):
        super(StateWithDerivatives, self).__init__(domain)
        self.ds_dx = np.zeros(self.domain.shape)
        self.dv_dx = np.zeros(self.domain.shape)
        self.ds_dy = np.zeros(self.domain.shape)
        self.dv_dy = np.zeros(self.domain.shape)
        self.d2s_dx2 = np.zeros(self.domain.shape)
        self.d2v_dx2 = np.zeros(self.domain.shape)
        self.d2s_dy2 = np.zeros(self.domain.shape)
        self.d2v_dy2 = np.zeros(self.domain.shape)
        self.d2s_dxdy = np.zeros(self.domain.shape)
        self.d2v_dxdy = np.zeros(self.domain.shape)

    @timer
    def calculate_derivatives(self):
        self.calculate_all_ds_dx()
        self.calculate_all_dv_dx()
        self.calculate_all_ds_dy()
        self.calculate_all_dv_dy()
        self.calculate_all_d2s_dx2()
        self.calculate_all_d2v_dx2()
        self.calculate_all_d2s_dy2()
        self.calculate_all_d2v_dy2()
        self.calculate_all_d2s_dxdy()
        self.calculate_all_d2v_dxdy()

    # TODO: move these general derivate methods to domain class
    def calculate_ddx(self, array, i, j):
        return self.domain.ddx(array, i, j)

    def calculate_ddy(self, array, i, j):
        return self.domain.ddy(array, i, j)

    def calculate_d2dx2(self, array, i, j):
        return self.domain.d2dx2(array, i, j)

    def calculate_d2dy2(self, array, i, j):
        return self.domain.d2dy2(array, i, j)

    def calculate_d2dxdy(self, array, i, j):
        return self.domain.d2dxdy(array, i, j)

    def calculate_dsdx(self, i, j):
        return self.domain.ddx(self.s, i, j)

    def calculate_dvdx(self, i, j):
        return self.domain.ddx(self.v, i, j)

    def calculate_dsdy(self, i, j):
        return self.domain.ddy(self.s, i, j)

    def calculate_dvdy(self, i, j):
        return self.domain.ddy(self.v, i, j)

    def calculate_d2sdx2(self, i, j):
        return self.domain.d2dx2(self.s, i, j)

    def calculate_d2vdx2(self, i, j):
        return self.domain.d2dx2(self.v, i, j)

    def calculate_d2sdy2(self, i, j):
        return self.domain.d2dy2(self.s, i, j)

    def calculate_d2vdy2(self, i, j):
        return self.domain.d2dy2(self.v, i, j)

    def calculate_d2sdxdy(self, i, j):
        return self.domain.d2dxdy(self.s, i, j)

    def calculate_d2vdxdy(self, i, j):
        return self.domain.d2dxdy(self.v, i, j)

    def calculate_all_ds_dx(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                self.ds_dx[i, j] = self.calculate_ddx(self.s, i, j)

    def calculate_all_dv_dx(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                self.dv_dx[i, j] = self.calculate_ddx(self.v, i, j)

    def calculate_all_ds_dy(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                self.ds_dy[i, j] = self.calculate_ddy(self.s, i, j)

    def calculate_all_dv_dy(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                self.dv_dy[i, j] = self.calculate_ddy(self.v, i, j)

    def calculate_all_d2s_dx2(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_interior(i, j):
                    self.d2s_dx2[i, j] = self.calculate_d2dx2(self.s, i, j)

    def calculate_all_d2v_dx2(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_interior(i, j):
                    self.d2v_dx2[i, j] = self.calculate_d2dx2(self.v, i, j)

    def calculate_all_d2s_dy2(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_interior(i, j):
                    self.d2s_dy2[i, j] = self.calculate_d2dy2(self.s, i, j)

    def calculate_all_d2v_dy2(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_interior(i, j):
                    self.d2v_dy2[i, j] = self.calculate_d2dy2(self.v, i, j)

    def calculate_all_d2s_dxdy(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_interior(i, j):
                    self.d2s_dxdy[i, j] = self.calculate_d2dxdy(self.s, i, j)

    def calculate_all_d2v_dxdy(self):
        for j in range(self.domain.m+1):
            for i in range(self.domain.n+1):
                if self.domain.is_interior(i, j):
                    self.d2v_dxdy[i, j] = self.calculate_d2dxdy(self.v, i, j)

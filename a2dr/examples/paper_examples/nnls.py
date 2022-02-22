"""
Copyright 2019 Anqi Fu, Junzi Zhang

This file is part of A2DR.

A2DR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

A2DR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with A2DR. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy as sp
import numpy.linalg as LA
import copy
import time
import scipy.sparse.linalg
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

from cvxpy import *
from scipy import sparse
from scipy import io as sio
from scipy.optimize import nnls
from sklearn.datasets import make_sparse_spd_matrix

from a2dr import a2dr
from a2dr.proximal import *
from a2dr.tests.base_test import BaseTest
from lm_aa import lmaa

class TestPaper(BaseTest):
    """Unit tests for A2DR paper experiments."""

    def setUp(self):
        np.random.seed(1)
        self.eps_rel = 1e-20   # specify these in all examples?
        self.eps_abs = 1e-20
        self.MAX_ITER = 1500

    def test_nnls(self):
        # minimize ||Fz - g||_2^2 subject to z >= 0.

        # Problem data.

        p, q = 600, 300
        density = 0.01
        F = sparse.random(p, q, density=density, data_rvs=np.random.randn)
        FTF = F.T@F
        FTF= FTF.toarray()
        sigma1=LA.norm(FTF,ord=2)
        sigma0=LA.norm(scipy.sparse.identity(FTF.shape[0])-FTF/sigma1,ord=2)
        sigma0 = (1-sigma0)*sigma1
        g = np.random.randn(p)

        # Convert problem to standard form.
        # f_1(x_1) = ||Fx_1 - g||_2^2, f_2(x_2) = I(x_2 >= 0).
        # A_1 = I_n, A_2 = -I_n, b = 0.
        prox_list = [lambda v, t: prox_sum_squares_affine(v, t, F, g), prox_nonneg_constr]
        A_list = [sparse.eye(q), -sparse.eye(q)]
        b = np.zeros(q)

        lmaa_result = lmaa(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER,eps_rel=self.eps_rel,eps_abs=self.eps_abs,sigma0=sigma0,sigma1=sigma1 )

        # Solve with DRS.
        drs_result = a2dr(prox_list, A_list, b, anderson=False, precond=True, max_iter=self.MAX_ITER,eps_rel=self.eps_rel,eps_abs=self.eps_abs)
        print('Finish DRS.')
    
        # Solve with A2DR.
        t0 = time.time()
        a2dr_result = a2dr(prox_list, A_list, b, anderson=True, precond=True, max_iter=self.MAX_ITER,eps_rel=self.eps_rel,eps_abs=self.eps_abs)
        t1 = time.time()

        name_list=['drs','a2dr','lm-aa']
        result_list=[drs_result,a2dr_result,lmaa_result]
        for i in range(len(name_list)):
            sio.savemat(name_list[i]+ 'r_dr.mat' , {'r_dr':result_list[i]['r_dr']})
            sio.savemat(name_list[i] + 'primal.mat', {'primal': result_list[i]['primal']})
            sio.savemat(name_list[i] + 'dual.mat', {'dual': result_list[i]['dual']})
            sio.savemat(name_list[i] + 'time_iter.mat', {'time_iter': result_list[i]['time_iter']})

        a2dr_beta = a2dr_result["x_vals"][-1]
        print('nonzero entries proportion = {}'.format(np.sum(a2dr_beta > 0)*1.0/len(a2dr_beta)))
        print('Finish A2DR.')
        #self.compare_total(drs_result, a2dr_result)
        self.compare_total_all(result_list,name_list)
        # Check solution correctness.
        print('run time of A2DR = {}'.format(t1-t0))
        print('constraint violation of A2DR = {}'.format(np.min(a2dr_beta)))
        print('objective value of A2DR = {}'.format(np.linalg.norm(F.dot(a2dr_beta)-g)))

if __name__ == '__main__':
    tests = TestPaper()
    tests.setUp()
    tests.test_nnls()


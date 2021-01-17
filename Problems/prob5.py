# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:30:05 2016

@author: Fin Bauer
"""

import numpy as np
import scipy.sparse as spa
import scipy.io as sio
import prox_0bfgs as p0b
import prox_0sr1 as p0s
import prox_grad as pg
import scipy.optimize as spo
import time as t

def setup_problem5():
    news20 = sio.loadmat("news20.mat")
    A = news20["A"]
    y = news20["y"].reshape(-1)
    m, n = np.shape(A)
    x0 = np.zeros((n, 1))
    l_reg= 0.001
    x0_lbfgsb = np.zeros(2 * n)
    y = spa.diags(y, 0)
    A = y * A
    A_lbfgsb = spa.hstack((A, -A))
    f = lambda x: fun(x, A, m)
    f_lbfgsb = lambda x: fun_lbfgsb(x, A_lbfgsb, m, l_reg)
    gf = lambda x: gfun(x, A, m)
    gf_lbfgsb = lambda x: gfun_lbfgsb(x, A_lbfgsb, m, l_reg)
    bounds = (-np.ones((n, 1)), np.ones((n, 1)))
    bounds_lbfgsb = 2 * n * [(0, 1.)]
    sio.savemat("prob5_setup", {"A": A, "x0": x0, "l_reg": l_reg})
    return x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, bounds, bounds_lbfgsb, l_reg

def fun(x, A, m):
    temp = np.maximum(0, 1 - A.dot(x))
    return np.dot(temp.T, temp) / (2 * m)

def gfun(x, A, m):
    temp = np.maximum(0, 1 - A.dot(x))
    return -A.T.dot(temp) / m

def fun_lbfgsb(x, A, m, l_reg):
    temp = np.maximum(0, 1 - A.dot(x))
    return np.dot(temp.T, temp) / (2 * m) + l_reg * np.sum(x)

def gfun_lbfgsb(x, A, m, l_reg):
    temp = np.maximum(0, 1 - A.dot(x))
    return -A.T.dot(temp) / m + l_reg

def callback(x, f, fvals, times, t0, t1, niter):
    t1[0] = t.time()
    fv = f(x)
    fvals[niter[0]] = fv
    times[niter[0]] = t1 - t0
    niter[0] += 1
    return

def lbfgsb(f, gf, x0, bounds):
    max_iter = 10000
    fvals = np.zeros(max_iter)
    fvals[0] = f(x0)
    times = np.zeros(max_iter)
    t0 = np.zeros(1)
    t1 = np.zeros(1)
    niter = np.ones(1, dtype=int)
    cb = lambda x: callback(x, f, fvals, times, t0, t1, niter)
    t0[0] = t.time()
    spo.fmin_l_bfgs_b(f, x0, gf, bounds=bounds, maxiter=max_iter,
                      callback=cb, factr=1e5, pgtol=1e-8)
    fvals = fvals[0:niter[0]]
    times = times[0:niter[0]]
    return fvals, times
    
if __name__=="__main__":
    # set up problem 5
    x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, bounds, bounds_lbfgsb, l_reg = setup_problem5()
    # run Prox0BFGS and Prox0SR1 comparison
    x1, f1, hist1 = p0b.prox_0bfgs(f, gf, x0, h="l1_box", l_reg=l_reg, bounds=bounds, display=0, tol=1e-5)
    x3, f3, hist3 = p0s.prox_0sr1(f, gf, x0, h="l1_box", l_reg=l_reg, bounds=bounds, display=0, tol=1e-5) 
    # compute optimal solution to high precision
    xopt, fopt, hist = p0b.prox_0bfgs(f, gf, x0, h="l1_box", l_reg=l_reg, bounds=bounds, line_search=True, tol=1e-7, tol_ssn=1e-12)
    # compute ProxGrad and LBFGSB for plotting
    x7, f7, hist7 = pg.prox_grad(f, gf, x0, h="l1_box", l_reg=l_reg, bounds=bounds, display=0, line_search=True, tol=1e-5, max_iter=20000)  
    lbfgsb_fv, lbfgsb_t = lbfgsb(f_lbfgsb, gf_lbfgsb, x0_lbfgsb, bounds_lbfgsb)
    sio.savemat("prob5_out_python", {"0b_fv": hist1.fvals, "0b_t": hist1.times,
                                     "0s_fv": hist3.fvals, "0s_t": hist3.times,
                                     "pg_fv": hist7.fvals, "pg_t": hist7.times,
                                     "lbfgsb_fv": lbfgsb_fv, "lbfgsb_t": lbfgsb_t,
                                     "fopt": fopt,
                                     "niter_ssn": hist1.niter_ssn})
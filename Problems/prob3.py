# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 14:19:36 2016

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


def setup_problem3():
    rcv1 = sio.loadmat("rcv1.mat")
    A = rcv1["A"]
    y = rcv1["y"].reshape(-1)
    m, n = np.shape(A)
    l_reg = 0.25 / m
    x0 = np.zeros((n, 1))
    x0_lbfgsb = np.zeros(2 * n)
    y = spa.diags(-y, 0)
    A = y * A
    A_lbfgsb = spa.hstack((A, -A))
    f = lambda x: fun(x, A, m)
    f_lbfgsb = lambda x: fun_lbfgsb(x, A_lbfgsb, m, l_reg)
    gf = lambda x: gfun(x, A, m)
    gf_lbfgsb = lambda x: gfun_lbfgsb(x, A_lbfgsb, m, l_reg)
    bounds = 2 * n * [(0, None)]
    #sio.savemat("prob3_setup", {"A": A, "x0": x0, "l_reg": l_reg})
    return x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, bounds, l_reg

def fun(x, A, m):
    return np.sum(np.log(1. + np.exp(A.dot(x)))) / m

def gfun(x, A, m):
    temp = np.exp(A.dot(x))
    return A.T.dot(temp / (1. + temp)) / m

def fun_lbfgsb(x, A, m, l_reg):
    return np.sum(np.log(1. + np.exp(A.dot(x)))) / m + l_reg * np.sum(x)

def gfun_lbfgsb(x, A, m, l_reg):
    temp = np.exp(A.dot(x))
    return A.T.dot(temp / (1. + temp)) / m + l_reg

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


if __name__ == "__main__":
    # set up problem 3
    x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, bounds, l_reg = setup_problem3()
    # run Prox0BFGS and Prox0SR1 comparison
    
    timing = np.zeros(6)
    for i in range(1):
        x1, f1, hist1 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, display=0, tol=1e-5)
        x2, f2, hist2 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, display=0, line_search=True, tol=1e-5)
        x3, f3, hist3 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, display=0, tol=1e-5)
        x4, f4, hist4 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, display=0, line_search=True, tol=1e-5)
        x5, f5, hist5 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, display=0, exact=True, tol=1e-5)
        x6, f6, hist6 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, display=0, exact=True, line_search=True, tol=1e-5)
        timing += np.array([hist1.times[-1], hist2.times[-1], hist3.times[-1],
                            hist4.times[-1], hist5.times[-1], hist6.times[-1]])
    # create evaluation table
    niter = np.array([hist1.niter, hist2.niter, hist3.niter, hist4.niter, hist5.niter, hist6.niter])
    timing /= 1.
    fevals = np.array([0, np.sum(hist2.fevals), 0, np.sum(hist4.fevals), 0, np.sum(hist6.fevals)])
    niter_ssn = np.array([np.sum(hist1.niter_ssn), np.sum(hist2.niter_ssn), np.sum(hist3.niter_ssn), np.sum(hist4.niter_ssn), 0, 0])
    eval_table = np.hstack((niter, timing, fevals, niter_ssn)).reshape((4, 6)).T
    sio.savemat("prob3evaltable", {"eval_table": eval_table})
    
    # compute optimal solution to high precision
    xopt, fopt, hist = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, line_search=True, tol=1e-7, tol_ssn=1e-12)
    # compute ProxGrad and LBFGSB for plotting
    x7, f7, hist7 = pg.prox_grad(f, gf, x0, h="l1", l_reg=l_reg, display=0, line_search=True, tol=1e-5, max_iter=10000) 
    lbfgsb_fv, lbfgsb_t = lbfgsb(f_lbfgsb, gf_lbfgsb, x0_lbfgsb, bounds)
    
    sio.savemat("prob3_out_python", {"0b_fv": hist1.fvals, "0b_t": hist1.times,
                                     "0s_fv": hist3.fvals, "0s_t": hist3.times,
                                     "pg_fv": hist7.fvals, "pg_t": hist7.times,
                                     "lbfgsb_fv": lbfgsb_fv, "lbfgsb_t": lbfgsb_t,
                                     "fopt": fopt, "eval_table": eval_table,
                                     "niter_ssn": hist1.niter_ssn})
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 18:48:03 2016

@author: Fin Bauer
"""

import numpy as np
import scipy.linalg as spl
import scipy.io as sio
import prox_0bfgs as p0b
import prox_0sr1 as p0s
import prox_grad as pg
import scipy.optimize as spo
import time as t

def setup_problem1(i):
    np.random.seed(int(i))
    m, n = 1500, 3000
    l_reg = 0.1
    x0 = np.zeros((n, 1))
    x0_lbfgsb = np.zeros(2 * n)
    A = np.random.normal(size = (m, n))
    A_lbfgsb = np.hstack((A, -A))
    b = np.random.normal(size = (m, 1))
    Asq = np.dot(A.T, A)
    Asq_lbfgsb = np.dot(A_lbfgsb.T, A_lbfgsb)
    Ab = np.dot(A.T, b)
    Ab_lbfgsb = np.dot(A_lbfgsb.T, b)
    f = lambda x: fun(x, A, b)
    f_lbfgsb = lambda x: fun_lbfgsb(x, A_lbfgsb, b, l_reg)
    gf = lambda x: gfun(x, Asq, Ab)
    gf_lbfgsb = lambda x: gfun_lbfgsb(x, Asq_lbfgsb, Ab_lbfgsb, l_reg)
    bounds = 2 * n * [(0, None)]
    L = spl.eigvals(Asq).real.max()
    sio.savemat("prob1_setup", {"A": A, "b": b, "x0": x0, "l_reg": l_reg, "L": L})
    return x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, l_reg, bounds, L

def fun(x, A, b):
        temp = np.dot(A, x) - b
        res = 0.5 * np.dot(temp.T, temp)
        return res[0,0]

def gfun(x, Asq, Ab):
    return np.dot(Asq, x) - Ab

def fun_lbfgsb(x, A, b, l_reg):
    temp = np.dot(A, x.reshape((-1, 1))) - b
    res = 0.5 * np.dot(temp.T, temp) + l_reg * np.sum(x)
    return res[0, 0]

def gfun_lbfgsb(x, Asq, Ab, l_reg):
    res = np.dot(Asq, x.reshape((-1, 1))) - Ab + l_reg
    return res.reshape(-1)

def callback(x, f, fvals, times, t0, t1, niter):
    t1[0] = t.time()
    fv = f(x)
    fvals[niter[0]] = fv
    times[niter[0]] = t1 - t0
    niter[0] += 1
    return

def lbfgsb(f, gf, x0, bounds):
    max_iter = 100000
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
    # run Prox0BFGS and Prox0SR1 comparison
    niter = np.zeros(6)
    timing = np.zeros(6)
    fevals = np.zeros(6)
    niter_ssn = np.zeros(6)
    for i in range(10):
        # create new instance of problem 1
        x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, l_reg, bounds, L = setup_problem1(i)
        x1, f1, hist1 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
        x2, f2, hist2 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, line_search=True)
        x3, f3, hist3 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
        x4, f4, hist4 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, line_search=True)
        x5, f5, hist5 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, exact=True)
        x6, f6, hist6 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, exact=True, line_search=True)
        niter += np.array([hist1.niter, hist2.niter, hist3.niter, hist4.niter, hist5.niter, hist6.niter])
        timing += np.array([hist1.times[-1], hist2.times[-1], hist3.times[-1], hist4.times[-1], hist5.times[-1], hist6.times[-1]])
        fevals += np.array([0, np.sum(hist2.fevals), 0, np.sum(hist4.fevals), 0, np.sum(hist6.fevals)])
        niter_ssn += np.array([np.sum(hist1.niter_ssn), np.sum(hist2.niter_ssn), np.sum(hist3.niter_ssn), np.sum(hist4.niter_ssn), 0, 0])
        print(i)
    # create evaluation table
    niter /= 10.
    timing /= 10.
    fevals /= 10.
    niter_ssn /= 10.
    eval_table = np.hstack((niter, timing, fevals, niter_ssn)).reshape((4, 6)).T
    # set up instance of problem 1 for plotting
    x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, l_reg, bounds, L = setup_problem1(2)
    # compute optimal solution to high precision
    xopt, fopt, hist = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, line_search=True, L=L, tol=1e-12, tol_ssn=1e-12)
    # compute Prox0BFGS, Prox0SR1, ProxGrad and LBFGSB for plotting
    x1, f1, hist1 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
    x3, f3, hist3 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
    x7, f7, hist7 = pg.prox_grad(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
    lbfgsb_fv, lbfgsb_t = lbfgsb(f_lbfgsb, gf_lbfgsb, x0_lbfgsb, bounds)
    sio.savemat("prob1_out_python", {"0b_fv": hist1.fvals, "0b_t": hist1.times,
                                     "0s_fv": hist3.fvals, "0s_t": hist3.times,
                                     "pg_fv": hist7.fvals, "pg_t": hist7.times,
                                     "lbfgsb_fv": lbfgsb_fv, "lbfgsb_t": lbfgsb_t,
                                     "fopt": fopt, "eval_table": eval_table,
                                     "niter_ssn": hist1.niter_ssn})
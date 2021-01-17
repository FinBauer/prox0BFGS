# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 16:56:50 2016

@author: Fin Bauer
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as spa
import scipy.sparse.linalg as spalin
import prox_0bfgs as p0b
import prox_0sr1 as p0s
import prox_grad as pg
import scipy.optimize as spo
import time as t


def setup_problem2():
    np.random.seed(0)
    n = 13**3
    l_reg = 1.
    x0 = np.ones((n, 1))
    x0_lbfgsb = np.zeros(2 * n)
    x0_lbfgsb[0:n] = 1.
    A, b = build_Ab()
    A_lbfgsb = spa.hstack((A, -A))
    Asq = A.dot(A)
    Asq_lbfgsb = A_lbfgsb.T.dot(A_lbfgsb)
    Ab = A.dot(b)
    Ab_lbfgsb = A_lbfgsb.T.dot(b)
    f = lambda x: fun(x, A, b)
    f_lbfgsb = lambda x: fun_lbfgsb(x, A_lbfgsb, b, l_reg)
    gf = lambda x: gfun(x, Asq, Ab)
    gf_lbfgsb = lambda x: gfun_lbfgsb(x, Asq_lbfgsb, Ab_lbfgsb, l_reg)
    bounds = 2 * n * [(0, None)]
    L = spalin.eigs(Asq, k=1)[0][0].real
    sio.savemat("prob2_setup", {"A": A, "b": b, "x0": x0, "l_reg": l_reg, "L": L})
    return x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, l_reg, bounds, L

def build_Ab():
    l, m, n = 13, 13, 13
    alpha, beta, gamma, sigma = 0.4, 0.7, 0.5, 50
    Il = spa.eye(l)
    Im = spa.eye(m)
    Ilm = spa.eye(l * m)
    In = spa.eye(n)
    T = spa.diags((6., -1., -1.), (0, -1, 1), (l, l))
    W = spa.kron(Im, T) + spa.kron(spa.diags((-1., -1.), (-1, 1), (m, m)), Il)
    A = spa.kron(In, W) + spa.kron(spa.diags((-1., -1.), (-1, 1), (n, n)), Ilm)
    b = u(l, m, n, alpha, beta, gamma, sigma).reshape(l * m * n, 1)
    b = A.dot(b)
    return A, b

def u(l, m, n, alpha, beta, gamma, sigma):
    h = 1. / (l + 1.)
    Y = (m + 1.) * h
    Z = (n + 1.) * h
    x1 = np.linspace(h, 1, l, endpoint=False)
    y1 = np.linspace(h, (m + 1.) * h, m, endpoint=False)
    z1 = np.linspace(h, (n + 1.) * h, n, endpoint=False)
    x, y, z = np.meshgrid(x1, y1, z1)
    uv = np.exp(-0.5 * sigma**2 * ((x - alpha)**2 + (y - beta)**2 + (z - gamma)**2))
    uv *= x * (x - 1.) * y * (y - Y) * z * (z - Z)
    return uv

def fun(x, A, b):
        temp = A.dot(x) - b
        res = 0.5 * np.dot(temp.T, temp)
        return res[0,0]

def gfun(x, Asq, Ab):
    return Asq.dot(x) - Ab

def fun_lbfgsb(x, A, b, l_reg):
    temp = A.dot(x.reshape((-1, 1))) - b
    res = 0.5 * np.dot(temp.T, temp) + l_reg * np.sum(x)
    return res[0, 0]

def gfun_lbfgsb(x, Asq, Ab, l_reg):
    res = Asq.dot(x.reshape((-1, 1))) - Ab + l_reg
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
    spo.fmin_l_bfgs_b(f_lbfgsb, x0_lbfgsb, gf_lbfgsb, bounds=bounds, maxiter=max_iter,
                      callback=cb, factr=1e5, pgtol=1e-8)
    fvals = fvals[0:niter[0]]
    times = times[0:niter[0]]
    return fvals, times


if __name__ == "__main__":
    # set up problem 2
    x0, x0_lbfgsb, f, f_lbfgsb, gf, gf_lbfgsb, l_reg, bounds, L = setup_problem2()
    # run Prox0BFGS and Prox0SR1 comparison
    timing = np.zeros(6)
    for i in range(10):
        x1, f1, hist1 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
        x2, f2, hist2 = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, line_search=True)
        x3, f3, hist3 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0)
        x4, f4, hist4 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, line_search=True)
        x5, f5, hist5 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, exact=True)
        x6, f6, hist6 = p0s.prox_0sr1(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0, exact=True, line_search=True)
        timing += np.array([hist1.times[-1], hist2.times[-1], hist3.times[-1],
                            hist4.times[-1], hist5.times[-1], hist6.times[-1]])
    # create evaluation table
    niter = np.array([hist1.niter, hist2.niter, hist3.niter, hist4.niter, hist5.niter, hist6.niter])
    timing /= 10.
    fevals = np.array([0, np.sum(hist2.fevals), 0, np.sum(hist4.fevals), 0, np.sum(hist6.fevals)])
    niter_ssn = np.array([np.sum(hist1.niter_ssn), np.sum(hist2.niter_ssn), np.sum(hist3.niter_ssn), np.sum(hist4.niter_ssn), 0, 0])
    eval_table = np.hstack((niter, timing, fevals, niter_ssn)).reshape((4, 6)).T
    # compute optimal solution to high precision
    xopt, fopt, hist = p0b.prox_0bfgs(f, gf, x0, h="l1", l_reg=l_reg, L=L, line_search=True, tol=1e-12, tol_ssn=1e-12)
    # compute ProxGrad and LBFGSB for plotting
    x7, f7, hist7 = pg.prox_grad(f, gf, x0, h="l1", l_reg=l_reg, L=L, display=0) 
    lbfgsb_fv, lbfgsb_t = lbfgsb(f_lbfgsb, gf_lbfgsb, x0_lbfgsb, bounds)
    sio.savemat("prob2_out_python", {"0b_fv": hist1.fvals, "0b_t": hist1.times,
                                     "0s_fv": hist3.fvals, "0s_t": hist3.times,
                                     "pg_fv": hist7.fvals, "pg_t": hist7.times,
                                     "lbfgsb_fv": lbfgsb_fv, "lbfgsb_t": lbfgsb_t,
                                     "fopt": fopt, "eval_table": eval_table,
                                     "niter_ssn": hist1.niter_ssn})
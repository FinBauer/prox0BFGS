# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 18:13:08 2016

@author: Fin Bauer
"""

import numpy as np
import numpy.linalg as npl
import input_parsers as ip
import subroutines as sr
import time as t

def prox_0bfgs(f, gf, x0, **options):
    """
    Proximal 0-memory BFGS method.
    Solves problems of the form
       \min_x F(x) := g(x) + h(x),
    where g is smooth and convex and h is convex.

    Input
    -----
    f : callable
        Smooth part g.
    gf : callable
        Gradient of smooth part. Needs to return ndarray of dim=(len(x0), 1).
    x0 : ndarray
        Starting point for method.

    Options
    -------
    See set_options class.

    Output
    ------
    x_opt : ndarray
        Optimal point.
    f_opt : float
        Optimal objective value.
    hist : history object
        Contains information about optimization run. See run_history class.
    """
    opts = ip.set_options("0bfgs", **options) # set default options
    nsm = ip.nonsmooth(opts.h, "0bfgs") # initialize nonsmooth function object
    hist = ip.run_history(opts.max_iter, opts.save_hist) # data collection object
    n = np.shape(x0)[0]
    x0 = x0.reshape((n, 1))
    x_new = x0.copy()
    s = np.zeros((n, 1))
    y = np.zeros((n, 1))
    gfv_new = gf(x_new)
    niter = 0

    if opts.display > 0:
        print("=" * 7 + " Proximal 0BFGS Method " + "=" * 8 + "\n")
        print(" %5s |   %12s   %12s" % ("Iter", "Obj. Val.", "Optim."))
        print("-" * 38)
        print(" %5d |   %12.4e   %12s" % (niter, f(x0) + nsm.h(x0, t=opts.l_reg, bounds=opts.bounds), " "))
    if hist.save_hist:
        hist.fvals[0] = f(x0) + nsm.h(x0, t=opts.l_reg, bounds=opts.bounds)
        t0 = t.time()

    while niter < opts.max_iter:
        x_old = x_new.copy()
        gfv_old = gfv_new.copy()
        if niter > 0:
            d, u, v, forw_step = bfgs_update(s, y, x_old, gfv_old, opts)
        else:
            d = opts.L
            u = np.zeros((n, 1))
            v = np.zeros((n, 1))
            forw_step = x_old - gfv_old / opts.L

        delta_x = proximal(forw_step, d, u, v, nsm, hist, opts) - x_old

        if opts.line_search: # line search
            x_new = sr.line_search(f, gfv_old, x_old, delta_x, nsm, hist, opts)
        else: # no line search
            x_new = x_old + delta_x
        gfv_new = gf(x_new)
        s = x_new - x_old
        y = gfv_new - gfv_old
        niter += 1

        if opts.display > 0 and np.mod(niter, opts.display) == 0:
            print(" %5d |   %12.4e   %12.4e" % (niter, f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds), npl.norm(delta_x)))
        if hist.save_hist:
            hist.fvals[niter] = f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds)
            hist.times[niter] = t.time() - t0
            hist.niter = niter

        if npl.norm(delta_x) < opts.tol: # stopping criterion
            break

    if opts.display > 1 and np.mod(niter, opts.display) != 0:
            print(" %5d |   %12.4e   %12.4e" % (niter, f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds), npl.norm(delta_x)))
    if hist.save_hist:
        hist.fvals = hist.fvals[:niter + 1]
        hist.fevals = hist.fevals[:niter + 1]
        hist.times = hist.times[:niter + 1]
        hist.niter_ssn = hist.niter_ssn[:niter + 1]
    return x_new, f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds), hist


def bfgs_update(s, y, x, gfv, opts):
    """ Computes 0-memory BFGS update and forward step: x_k - B_k * gf """
    ss = np.dot(s.T, s)
    sy = np.dot(s.T, y)
    yy = np.dot(y.T, y)
    sgf = np.dot(s.T, gfv)
    ygf = np.dot(y.T, gfv)
    tau = sy / yy # Barzilai-Borwein step length
    tau = np.median([opts.tau_min, tau, opts.tau_max]) # Projection
    d = 1. / tau
    if sy <= 1e-8 * np.sqrt(ss) * np.sqrt(yy): # skip BFGS update
        u = np.zeros((len(x), 1))
        v = np.zeros((len(x), 1))
        forw_step = x - tau * gfv
    else:
        u = y / np.sqrt(sy)
        v = d * s / np.sqrt(d * ss)
        forw_step = -tau * gfv + tau / sy * (sgf * y + ygf * s - yy * sgf * s / sy) - sgf / sy * s
        forw_step += x
    return d, u, v, forw_step


def proximal(x, d, u, v, nsm, hist, opts):
    """ Computes backward step \prox^H_h(x) """
    z = solve_dual(x, d, u, v, nsm, hist, opts)
    back_step = to_primal(z, x, d, u, v, nsm, opts)
    return back_step


def solve_dual(x, d, u, v, nsm, hist, opts):
    """ Computes dual solution """
    pv = lambda gamma: p(gamma, x, d, u, v, nsm, opts)
    invgpv = lambda gamma: invgp(gamma, x, d, u, v, nsm, opts)
    gamma = sr.semsmo_newton(pv, invgpv, np.zeros(2), hist, opts)
    z = gamma[0] * v + gamma[1] * u
    return z


def to_primal(z, x, d, u, v, nsm, opts):
    """ Computes primal solution from dual solution """
    back_step = nsm.prox(x + z / d, t=opts.l_reg / d, bounds=opts.bounds)
    return back_step


def p(gamma, x, d, u, v, nsm, opts):
    """ Function p as in thesis """
    pv = np.zeros(2)
    alpha = gamma[0]
    beta = gamma[1]
    T = x + (alpha * v + beta * u) / d
    pr = nsm.prox(T, t=opts.l_reg / d, bounds=opts.bounds)
    pv[0] = np.dot(v.T, x - pr) + alpha
    pv[1] = -np.dot(u.T, x - pr) + beta
    return pv


def invgp(gamma, x, d, u, v, nsm, opts):
    """ Inverse of generalized derivative of p """
    alpha = gamma[0]
    beta = gamma[1]
    vd = v / d
    ud = u / d
    T = x + alpha * vd + beta * ud
    G = nsm.gprox(T, t=opts.l_reg / d, bounds=opts.bounds)
    Gv = np.multiply(G, vd)
    Gu = np.multiply(G, ud)
    gpv = np.array([[-np.dot(v.T, Gv)[0,0] + 1., -np.dot(v.T, Gu)[0,0]],
                   [np.dot(u.T, Gv)[0,0], np.dot(u.T, Gu)[0,0] + 1.]])
    det = gpv[0,0] * gpv[1,1] - gpv[0,1] * gpv[1,0]
    invgpv = 1. / det * np.array([[gpv[1,1], -gpv[0,1]], [-gpv[1,0], gpv[0,0]]])
    return invgpv
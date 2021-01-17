# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:06:28 2015

@author: Fin Bauer
"""

import numpy as np
import numpy.linalg as npl
import input_parsers as ip
import subroutines as sr
import time as t

def prox_0sr1(f, gf, x0, **options):
    """
    Proximal 0-memory SR1 method.
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
    opts = ip.set_options("0sr1", **options) # set default options
    nsm = ip.nonsmooth(opts.h, "0sr1") # initialize nonsmooth function object
    hist = ip.run_history(opts.max_iter, opts.save_hist) # data collection object
    n = np.shape(x0)[0]
    x0 = x0.reshape((n, 1))
    x_new = x0.copy()
    s = np.zeros((n, 1))
    y = np.zeros((n, 1))
    gfv_new = gf(x_new)
    niter = 0
    
    if opts.display > 0:
        print("=" * 8 + " Proximal 0SR1 Method " + "=" * 8 + "\n")
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
            d, u, forw_step = sr1_update(s, y, x_old, gfv_old, opts)
        else:
            d = opts.L
            u = np.zeros((n, 1))
            forw_step = x_old - gfv_old / opts.L

        delta_x = proximal(forw_step, d, u, nsm, hist, opts) - x_old
        
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


def sr1_update(s, y, x, gfv, opts):
    """ Computes 0-memory SR1 update and forward step: x_k - B_k * gf """
    yy = np.dot(y.T, y)
    sy = np.dot(s.T, y)
    tau = sy / yy # Barzilai-Borwein step length
    tau = np.median([opts.tau_min, tau, opts.tau_max]) # Projection
    d = opts.gamma * tau
    if sy - d * yy <= 1e-8 * np.sqrt(yy) * npl.norm(s - d * y): # skip SR1 update
        uinv = np.zeros((len(x), 1))
        u = np.zeros((len(x), 1))
    else:
        uinv = (s - d * y) / np.sqrt(sy - d * yy)
        u = uinv / (d * np.sqrt(1 + np.dot(uinv.T, uinv) / d))
    forw_step = -d * gfv - np.dot(uinv.T, gfv) * uinv
    forw_step += x
    return 1. / d, u, forw_step


def proximal(x, d, u, nsm, hist, opts):
    """ Computes backward step \prox^H_h(x) """
    z = solve_dual(x, d, u, nsm, hist, opts)
    back_step = to_primal(z, x, d, u, nsm, opts)
    return back_step


def solve_dual(x, d, u, nsm, hist, opts):
    """ Computes dual solution """
    pv = lambda alpha: p(alpha, x, d, u, nsm, opts)
    if opts.exact: # compute exact dual solution (only for separable h)
        trans_points = transition_points(x, d, u, nsm, opts)
        alpha = sr.binary_search(trans_points, pv)
        z = alpha * u
        return z
    else: # compute iterative dual solution
        invgpv = lambda alpha: invgp(alpha, x, d, u, nsm, opts)
        alpha = sr.semsmo_newton(pv, invgpv, np.zeros(1), hist, opts)
        z = alpha * u
        return z

def to_primal(z, x, d, u, nsm, opts):
    """ Computes primal solution from dual solution """
    l = opts.l_reg
    back_step = nsm.prox(x + z / d, t=l / d, bounds=opts.bounds)
    return back_step


def transition_points(x, d, u, nsm, opts):
    """ Returns sorted transition points """
    l = opts.l_reg
    tp = nsm.trans_points(x, t=l / d, bounds=opts.bounds)
    # exclude indices i for which u_i = 0
    nz = u != 0
    nz = nz.reshape(-1)
    x = x[nz,]
    u = u[nz,]
    tp = tp[nz,]
    if len(u) == 0:
        return np.empty(0,)
    else:
        return np.sort(d * (tp - x) / u, axis=None)

def p(alpha, x, d, u, nsm, opts):
    """ Function p as in thesis """
    l = opts.l_reg
    pr = nsm.prox(x + alpha * u / d, t=l / d, bounds=opts.bounds)
    return np.dot(u.T, x - pr) + alpha


def invgp(alpha, x, d, u, nsm, opts):
    """ Inverse of generalized derivative of p """
    ud = u / d
    l = opts.l_reg
    G = nsm.gprox(x + alpha * ud, t=l / d, bounds=opts.bounds)
    Gu = np.multiply(G, ud)
    gpv = 1. - np.dot(u.T, Gu)
    invgpv = 1. / gpv
    return invgpv
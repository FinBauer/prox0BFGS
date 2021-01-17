# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:48:16 2015

@author: Fin Bauer
"""

import numpy as np
import numpy.linalg as npl
import input_parsers as ip
import subroutines as sr
import time as t

def prox_grad(f, gf, x0, **options):
    """
    Proximal gradient method.
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
    opts = ip.set_options("grad", **options) # set default options
    nsm = ip.nonsmooth(opts.h, "grad") # initialize nonsmooth function object
    hist = ip.run_history(opts.max_iter, opts.save_hist) # data collection object
    x_new = x0.copy()
    gfv = gf(x_new)
    niter = 0

    if opts.display > 0:
        print("=" * 6 + " Proximal Gradient Method " + "=" * 6 + "\n")
        print(" %5s |   %12s   %12s" % ("Iter", "Obj. Val.", "Optim."))
        print("-" * 38)
        print(" %5d |   %12.4e   %12s" % (niter, f(x0) + nsm.h(x0, t=opts.l_reg, bounds=opts.bounds), " "))
    if hist.save_hist:
        hist.fvals[0] = f(x0) + nsm.h(x0, t=opts.l_reg, bounds=opts.bounds)
        t0 = t.time()
    
    while niter < opts.max_iter:
        x_old = x_new.copy()
        delta_x = nsm.prox(x_old - gfv / opts.L, t=opts.l_reg / opts.L, bounds=opts.bounds) - x_old

        if opts.line_search: # line search
            x_new = sr.line_search(f, gfv, x_old, delta_x, nsm, hist, opts)
        else: # no line search
            x_new = x_old + delta_x
        gfv = gf(x_new)
        niter += 1
        
        if opts.display > 0 and np.mod(niter, opts.display) == 0:
            print(" %5d |   %12.4e   %12.4e" % (niter, f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds), npl.norm(delta_x)))
        if hist.save_hist:
            hist.fvals[niter] = f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds)
            hist.times[niter] = t.time() - t0
            hist.niter = niter
        
        if npl.norm(delta_x) < opts.tol:
            break

    if opts.display > 1 and np.mod(niter, opts.display) != 0:
            print(" %5d |   %12.4e   %12.4e" % (niter, f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds), npl.norm(delta_x)))
    if hist.save_hist:
        hist.fvals = hist.fvals[:niter + 1]
        hist.fevals = hist.fevals[:niter + 1]
        hist.times = hist.times[:niter + 1]
        hist.niter_ssn = hist.niter_ssn[:niter + 1]
    return x_new, f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds), hist
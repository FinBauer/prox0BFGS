# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 10:35:32 2016

@author: Fin Bauer
"""

import numpy as np
import numpy.linalg as npl


def semsmo_newton(f, gfinv, x0, hist, opts):
    """ Semismooth Newton Method """
    x = x0.copy()
    delta_x = np.ones((len(x), 1))
    niter = 0
    while npl.norm(delta_x) > opts.tol_ssn and niter < 20:
        t = 1.
        fv = f(x)
        nfv = npl.norm(fv)
        delta_x = -np.dot(gfinv(x), fv)
        while npl.norm(f(x + t * delta_x)) > (1. - 2. * opts.desc_param_ssn * t) * nfv:
            t *= 0.5
        x = x + t * delta_x
        niter += 1
    if hist.save_hist:
        hist.niter_ssn[hist.niter + 1] = niter
    return x


def line_search(f, gfv, x, delta_x, nsm, hist, opts):
    """ Line search as in paper of Lee et al. """
    x_new = x + delta_x
    fv = f(x) + nsm.h(x, t=opts.l_reg, bounds=opts.bounds)
    lamda = np.dot(gfv.T, delta_x)
    lamda += nsm.h(x + delta_x, t=opts.l_reg, bounds=opts.bounds)
    lamda -= nsm.h(x, t=opts.l_reg, bounds=opts.bounds)
    t = 1.
    niter = 0
    while f(x_new) + nsm.h(x_new, t=opts.l_reg, bounds=opts.bounds) > fv + opts.desc_param_ls * t * lamda and niter < 20:
        t *= 0.5
        x_new = x + t * delta_x
        niter += 1
    if hist.save_hist:
        hist.fevals[hist.niter + 1] = niter + 1
    return x_new


def binary_search(trans_points, pv):
    """ Performs binary search on points to obtain root of p. Points need to be in ascending order."""
    # no transitions points just a straight line
    if len(trans_points) == 0:
        return 0.
    else:
        leftv = pv(trans_points[0])
        rightv = pv(trans_points[-1])
        # p values of all transition points are below zero
        if np.logical_and(leftv < 0., rightv < 0.):
            endv = pv(trans_points[-1] + 1.)
            return trans_points[-1] - rightv / (endv - rightv)
        # p values of all transition points are above zero
        elif np.logical_and(leftv > 0, rightv > 0):
            endv = pv(trans_points[0] - 1.)
            return trans_points[0] - 1. - endv / (leftv - endv)
        # normal case
        else:
            left, right = 0, len(trans_points) - 1
            while right - left != 1:
                middle = int(np.floor((left + right) / 2.))
                middlev = pv(trans_points[middle])
                if middlev == 0:
                    return trans_points[middle]
                elif middlev < 0:
                    left = middle
                    leftv = middlev
                else:
                    right = middle
                    rightv = middlev
            return trans_points[left] - leftv * (trans_points[right] - trans_points[left]) / (rightv - leftv)
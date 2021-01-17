# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:11:31 2016

@author: Fin Bauer
"""

import numpy as np
import numpy.linalg as npl

############# Nonsmooth functions ############################################
def l1(x, t=1., **varargin):
    """ L1-norm: t * \|x\|_1 """
    return t * npl.norm(x, ord=1)

def hinge(x, t=1., **varargin):
    """ Hinge loss: t * \sum_i\max(0, 1-x_i) """
    return t * np.sum(np.maximum(0., 1. - x))

def linf_ball(x, **varargin):
    """ Indicator of linf-ball constraints: I_{x\in[-1, 1]} """
    if np.any(np.absolute(x) > 1.):
        return np.inf
    else:
        return 0.

def box(x, bounds=[], **varargin):
    """ Indicator of box constraints: I_{x\in[lb, ub]} """
    if np.any(np.logical_or(x < bounds[0], x > bounds[1])):
        return np.inf
    else:
        return 0.

def pos(x, **varargin):
    """ Indicator of positivity constraints: I_{x>0} """
    if np.any(x < 0):
        return np.inf
    else:
        return 0.

def l1_box(x, t=1., bounds=[], **varargin):
    """ L1-norm with box constraints """
    if np.any(np.logical_or(x < bounds[0], x > bounds[1])):
        return np.inf
    else:
        return t * npl.norm(x, ord=1)

############# Proximity operators ############################################
def prox_l1(x, t=1., **varargin):
    """ Proximity operator of l1-norm """
    return np.maximum(x - t, 0) - np.maximum(-x - t, 0)

def prox_hinge(x, t=1., **varargin):
    """ Proximity operator of hinge loss """
    return np.minimum(x + t, np.maximum(x, 1))

def proj_linf_ball(x, **varargin):
    """ Projection onto linf-ball """
    return np.median([-np.ones((len(x), 1)), x, np.ones((len(x), 1))], axis=0)

def proj_box(x, bounds=[], **varargin):
    """ Projection onto box constraints """
    return np.median([bounds[0], x, bounds[1]], axis=0)

def proj_pos(x, **varargin):
    """ Projection onto positivity constraint """
    return np.maximum(0, x)

def prox_l1_box(x, t=1., bounds=[], **varargin):
    """ Proximity operator of l1-norm with box constraints """
    return proj_box(prox_l1(x, t=t), bounds=bounds)

def prox_hinge_dual(x, t=1., **varargin):
    """ Proximity operator of conjugate of hinge loss """
    return np.minimum(0., np.maximum(-1., x - t))

def prox_box_dual(x, t=1., bounds=[], **varargin):
    """ Proximity operator of conjugate of indicator of box constraints """
    p = np.zeros((len(x), 1))
    ind = x < t * bounds[0]
    p[ind] = x[ind] - t * bounds[0][ind]
    ind = x > t * bounds[1]
    p[ind] = x[ind] - t * bounds[1][ind]
    return p

def proj_neg(x, **varargin):
    """ Projection onto negativity constraints """
    return np.minimum(x, 0)

############# Subgradients of proximity operators ############################
def gprox_l1(x, t=1., **varargin):
    """ Elementwise subgradient of proximity operator of l1-norm """
    g = np.ones((len(x), 1))
    g[np.logical_and(x >= -t, x <= t)] = 0.
    return g

def gprox_hinge(x, t=1., **varargin):
    """ Elementwise subgradient of proximity operator of hinge loss """
    g = np.ones((len(x), 1))
    g[np.logical_and(x >= 1. - t, x <= 1.)] = 0.
    return g

def gproj_linf_ball(x, **varargin):
    """ Elementwise subgradient of projection onto linf-ball """
    g = np.zeros((len(x), 1))
    g[np.logical_and(x >= -1., x <= 1.)] = 1.
    return g

def gproj_box(x, bounds=[], **varargin):
    """ Elementwise subgradient of projection onto box constraints """
    g = np.zeros((len(x), 1))
    g[np.logical_and(x >= bounds[0], x <= bounds[1])] = 1.
    return g

def gproj_pos(x, **varargin):
    """ Elementwise subgradient of projection onto positivity constraints """
    g = np.zeros((len(x), 1))
    g[x >= 0] = 1.
    return g

def gprox_l1_box(x, t=1., bounds=[], **varargin):
    """ Elementwise subgradient of proximity operator of l1-norm with box constraints """
    return gproj_box(prox_l1(x, t=t), bounds=bounds) * gprox_l1(x, t=t)

def gprox_hinge_dual(x, t=1., **varargin):
    """ Elementwise subgradient of proximity operator of conjugate of hinge loss """
    g = np.zeros((len(x), 1))
    g[np.logical_and(x >= t - 1., x <= t)] = 1.
    return g

def gprox_box_dual(x, t=1., bounds=[], **varargin):
    """ Elementwise subgradient of proximity operator of conjugate of indicator of box constraints """
    g = np.ones((len(x), 1))
    g[np.logical_and(x >= t * bounds[0], x <= t * bounds[1])] = 0.
    return g

def gproj_neg(x, **varargin):
    """ Elementwise subgradient of projection onto negativity constraints """
    g = np.zeros((len(x), 1))
    g[x <= 0] = 1.
    return g

############# Transition points of proximity operators #######################

def tp_prox_l1(x, t=1., **varargin):
    """ Transition points of proximity operator of l1-norm """
    return np.tile([-t, t], (len(x), 1))

def tp_proj_linf_ball(x, **varargin):
    """ Transition points of projection onto linf-ball """
    return np.tile([-1., 1.], (len(x), 1))

def tp_prox_hinge_dual(x, t=1., **varargin):
    """ Transition points of proximity operator of conjugate of hinge loss """
    return np.tile([t - 1., t], (len(x), 1))

def tp_prox_box_dual(x, t=1., bounds=[], **varargin):
    """ Transition points of proximity operator of conjugate of indicator of box constraints """
    return np.tile([t * bounds[0], t * bounds[1]], (len(x), 1))

def tp_proj_neg(x, **varargin):
    """ Transition points of projection onto negativity constraints """
    return np.zeros((len(x), 1))
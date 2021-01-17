# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 10:17:12 2016

@author: Fin Bauer
"""

import numpy as np
import prox_operators as po


class set_options:
    """
    Class to process optional arguments.

    Options
    -------
    bounds : list
        Bounds for projection onto box constraints. Must be list of two ndarrays with bounds[0]=lower bounds, bounds[1]=upper bounds. Default: []
    desc_param_ls : float
        Descent parameter in line search. Default: 1e-3
    desc_param_ssn : float
        Descent parameter in line search of semismooth Newton method. Default: 1e-3
    display : int
        Print progress update every 'display' iterations. Set display=0 to mute. Default: 10
    exact : bool
        Solve dual problem exactly. Only for proximal 0SR1 method and separable h. Default: True
    gamma : float
        Damping factor for 0SR1 update. Only for proximal 0SR1 method. Default: 0.8
    h : str
        Nonsmooth part of composite function. See nonsmooth object. Default: "l1"
    L : float
        Estimate of Lipschitz constant of gf. Default: 1.
    line_search : bool
        Use line search. Default: False
    l_reg : float
        Regularization parameter for nonsmooth part. Default: 1.
    max_iter : int
        Maximum number of iterations. Default: 1000000
    save_hist: bool
        Store information of optimization run. Default: True
    tau_min : float
        Minimum for Barzilai-Borwein step size. Default: 1e-8
    tau_max : float
        Maximum for Barzilai-Borwein step size. Default: 1e8
    tol : float
        Tolerance for termination criterion. Default: 1e-7
    tol_ssn : float
        Tolerance for termination criterion in semismooth Newton method. Default: 1e-9
    """
    def __init__(self, method, **options):
        options.setdefault("max_iter", 100000)
        self.max_iter = options["max_iter"]
        options.setdefault("tol", 1e-7)
        self.tol = options["tol"]
        options.setdefault("save_hist", True)
        self.save_hist = options["save_hist"]
        options.setdefault("display", 100)
        self.display = options["display"]
        options.setdefault("tol_ssn", 1e-9)
        self.tol_ssn = options["tol_ssn"]
        options.setdefault("desc_param_ssn", 1e-3)
        self.desc_param_ssn = options["desc_param_ssn"]
        options.setdefault("line_search", False)
        self.line_search = options["line_search"]
        options.setdefault("desc_param_ls", 1e-4)
        self.desc_param_ls = options["desc_param_ls"]
        options.setdefault("h", "l1")
        self.h = options["h"]
        options.setdefault("l_reg", 1.)
        self.l_reg = options["l_reg"]
        options.setdefault("bounds", None)
        self.bounds = options["bounds"]
        options.setdefault("L", 1.)
        self.L = options["L"]
        options.setdefault("exact", False)
        self.exact = options["exact"]
        options.setdefault("gamma", 0.8)
        self.gamma = options["gamma"]
        options.setdefault("tau_min", 1e-8)
        self.tau_min = options["tau_min"]
        options.setdefault("tau_max", 1e8)
        self.tau_max = options["tau_max"]

class nonsmooth:
    """
    Class for creating nonsmooth function object. \n
    Possible nonsmooth functions are: \n
    - l1-norm (h="l1")
    - hinge loss (h="hinge")
    - linf-ball constraints (h="linf_ball")
    - box constraints (h="box")
    - positivity constraints (h="pos")

    Methods
    -------
    h : callable
        Nonsmooth function
    prox : callable
        Proximity operator for nonsmooth function
    gprox : callable
        Elementwise subgradient for proximity operator of nonsmooth function
    """
    def __init__(self, h, method):
        if h == "l1":
            self.h = po.l1
            if method == "0sr1m":
                self.prox = po.proj_linf_ball
                self.gprox = po.gproj_linf_ball
                self.trans_points = po.tp_proj_linf_ball
            else:
                self.prox = po.prox_l1
                self.gprox = po.gprox_l1
                self.trans_points = po.tp_prox_l1
        elif h == "hinge":
            self.h = po.hinge
            if method == "0sr1m":
                self.prox = po.prox_hinge_dual
                self.gprox = po.gprox_hinge_dual
                self.trans_points = po.tp_prox_hinge_dual
            else:
                self.prox = po.prox_hinge
                self.gprox = po.gprox_hinge
                self.trans_points = po.tp_prox_hinge
        elif h == "linf_ball":
            self.h = po.linf_ball
            if method == "0sr1m":
                self.prox = po.prox_l1
                self.gprox = po.gprox_l1
                self.trans_points = po.tp_prox_l1
            else:
                self.prox = po.proj_linf_ball
                self.gprox = po.gproj_linf_ball
                self.trans_points = po.tp_proj_linf_ball
        elif h == "box":
            self.h = po.box
            if method == "0sr1m":
                self.prox = po.prox_box_dual
                self.gprox = po.gprox_box_dual
                self.trans_points = po.tp_prox_box_dual
            else:
                self.prox = po.proj_box
                self.gprox = po.gproj_box
                self.trans_points = po.tp_prox_box
        elif h == "pos":
            self.h = po.pos
            if method == "0sr1m":
                self.prox = po.proj_neg
                self.gprox = po.gproj_neg
                self.trans_points = po.tp_proj_neg
            else:
                self.prox = po.proj_pos
                self.gprox = po.gproj_pos
                self.trans_points = po.tp_proj_pos
        elif h == "l1_box":
            self.h = po.l1_box
            if method == "0sr1m":
                print("Not implemented!")
            else:
                self.prox = po.prox_l1_box
                self.gprox = po.gprox_l1_box
                self.trans_points = po.tp_prox_l1_box
        else:
            pass

class run_history:
    """
    Class for collecting history of optimization run.

    Fields
    ------
    fvals : ndarray
        Objective value in each iteration
    times : ndarray
        Time passed since start of method
    fevals : ndarray
        Number of objective function evaluations in each iteration
    niter_ssn : ndarray
        Number of iterations in semismooth Newton method in each iteration
    niter : float
        Total number of iterations
    save_hist : bool
        Save history
    """
    def __init__(self, max_iter, save_hist):
        if save_hist:
            self.fvals = np.zeros(max_iter + 1)
            self.times = np.zeros(max_iter + 1)
            self.fevals = np.zeros(max_iter + 1, dtype=int)
            self.niter_ssn = np.zeros(max_iter + 1, dtype=int)
            self.niter = 0
            self.save_hist = save_hist
        else:
            self.save_hist = save_hist
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Apr  6 17:40:24 2020

Solving a multiregional ETSAP/TIMES LP model with lagrangian relaxation
and dualization of the interconnections:
min_X sum_{r} f(r) / IRE(r,r',t,imp) = IRE(r',r,t,exp) (forall r-r', t)
= max_u L(u) = sum_r L(u,r) with
L(u,r) = min_X [f(r) + sum_{r-r',t} u_{r,r',t} (IRE(r,r',t,imp) - IRE(r',r,t,exp))]   
3 alternative solution methods based on the Gurobi LP/QP Solvers:
     - solve the primal aggregate LP model min_X sum_{r} f(r) / IRE(r,r',t,imp) = IRE(r',r,t,exp) forall r-r', t
     - solve the lagrangian dual [max_u L(u)] with a subgradient method
     - solve the lagrangian dual [max_u L(u)] with a proximal bundle method
Input: the single regional LP models (without interconnection coupling constraints)
    
@author: Gildas Siggini, Sophie Demassey

TODO: change POSITIVE_QUADRANT from boolean to the set of indices of dualized inequality constraints
"""

import sys
import lpparsers as lp
#import pandas as pd
import proximalbundle as bd
import matplotlib.pylab as plt
import time
import logging
from pathlib import Path

INDIR = "../input/"
OUTDIR = "../output/"

logging.basicConfig(
        handlers=[
            #logging.FileHandler(Path(OUTDIR,"timeslag.log")),
            logging.StreamHandler()
        ],
        #format="%(levelname)s: %(message)s")
        #format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO)

NEUTREU29 = {'regions': ["AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","GR","HR",
                    "HU","IE","IS","IT","LT","LU","LV","NL","NO","PL","PT","RO","SE","SI","SK","UK"],
             'dir': INDIR + "neutr_test_eu29cplex",
             'lpfiles': "neutr_test_eu29_{}.lp.gz",
#             'co2file': "neutr_test_eu29_CO2_EMI_max.csv",
             'ub': 1.5e7}
#!!! force option Positive_Quadrant when co2file is specified (dualizing total emission bound)

EU29 = {'regions': ["AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","GR","HR",
                    "HU","IE","IS","IT","LT","LU","LV","NL","NO","PL","PT","RO","SE","SI","SK","UK"],
        'dir': INDIR + "base_eu29cplex",
        'lpfiles': "base_eu29_last_{}.lp.gz",
        'ub': 1.5e7}

FRES = {'regions': ["FR", "ES"],
        'dir': INDIR + "eu2cplex",
        'lpfiles': "Reg_{}.lp.gz",
#        "co2file": "eu2_CO2_EMI_max.csv",
        'ub': 1.9e6}

#!!! more numerical errors with Cplex (unbounded QP or OptimUnfeas oracles)
USE_CPLEX = False
COMPUTE_PRIMAL_SOLUTION = True

def solve_multiregion_times(instance=NEUTREU29, bundle_otherwise_sg=True,heuristic=False, PLOT=False):
    global USE_CPLEX
    global COMPUTE_PRIMAL_SOLUTION

    starttime = time.perf_counter()
    models =  lp.create(USE_CPLEX, instance, COMPUTE_PRIMAL_SOLUTION)
    parsetime = time.perf_counter()
    logging.info(f"parse models: {parsetime - starttime:0.4f} s")
    solve_times_lagrangian_dual(models, bundle_otherwise_sg, heuristic, PLOT, lbinit=-instance['ub'])
    endtime = time.perf_counter()
    logging.info(f"solve: {endtime - parsetime:0.4f} s")
    logging.info(f"parse+solve: {endtime - starttime:0.4f} s")
    #solve_times_lagrangian_dual(models, bundle_otherwise_sg=False, lb=-instance['ub'], heuristic=models.aggregate_heuristic)

    #solve_times_primal(models)

############ MAIN SOLUTION METHODS ###########################################

def solve_times_lagrangian_dual(models, bundle_otherwise_sg, heuristic, PLOT, lbinit=None):
    """Solve the (opposite) lagrangian dual [min_u -L(u)] iteratively starting from u=0
    using either the proximal bundle method if bundle_otherwise_sg, otherwise the subgradient algorithm."""
    global USE_CPLEX

    if heuristic and not bundle_otherwise_sg:
        heuristic = models.aggregate_heuristic
        models.aggregate_models()

    dim = len(models.duals)
    uinit = [0 for i in range(dim)]
    fu, u, iters, primalsol = bd.inexact_proximal_bundle(uinit, models.oraclelr, USE_CPLEX, MAX_ITER=10000,
                                                         TOL = 1e-5, POSITIVE_QUADRANT = False) if bundle_otherwise_sg \
                       else subgradient(uinit, models.oraclelr, heuristic, MAX_ITER=1000, LB=lbinit)
    logging.info(f"solution found {-fu}") #, u)
    its, bounds = zip(*(sorted(iters.items())))
    fx, fxc, serious, prox, erragg, normsgagg, elapsedtime = zip(*bounds)
    #df = pd.DataFrame.from_dict(iters, orient='index')
    if PLOT:
        plt.plot(its, fx, label='fx')
        plt.plot(its, fxc, label='fxc')
        plt.legend()
        plt.show()
        plt.plot(its, serious, label='serious')
        plt.legend()
        plt.show()
        plt.plot(its, prox, label='prox')
        plt.legend()
        plt.show()
        plt.plot(its, erragg, label='erragg')
        plt.legend()
        plt.show()
        plt.plot(its, normsgagg, label='normsgagg')
        plt.legend()
        plt.show()
        plt.plot(its, elapsedtime, label='time')
        plt.legend()
        plt.show()
        
    if primalsol:
        models.test_primal_solution(primalsol, tol=1e-3)


def solve_times_primal(models):
    """Aggregate and solve the multiregional LP model with the interconnections."""
    models.solve_aggregate()

def solve_allregs(instance = NEUTREU29):
    global USE_CPLEX

    starttime = time.perf_counter()
    reg= "AllRegs"
    lpfilename = str(Path(instance['dir'], instance['lpfiles'].format(reg)))
    model = cplex.Cplex(lpfilename) if USE_CPLEX else gp.read(lpfilename)
    parsetime = time.perf_counter()
    print(f"parse models: {parsetime - starttime:0.4f} s")
    if USE_CPLEX:
        model.solve()
    else:
        model.optimize()
    endtime = time.perf_counter()
    print(f"solve: {endtime - parsetime:0.4f} s")
    print(f"parse+solve: {endtime - starttime:0.4f} s")

############ SUBGRADIENT ALGORITHM ###########################################

def subgradient(x, oracle, heuristic, MAX_ITER=100, LB = -1.6e9, AGILITY_MAX=5e-1, TOL=1e-5):
    """Subgradient algorithm: minimizes a convex function f(x), x free.
    
    Runs the subgradient algorithm starting from point xc and 
    returns a minimizer of the convex function oracle(x) within given tolerance and iteration number limit.
    
    Args:
        xc: the starting point.
        MAX_ITER: maximum iteration number
        AGILITY_INIT: agility parameter to update the time step
        TOL: tolerance value 
        
    Returns:
        x: the best minimizer found.
    """
    starttime = time.perf_counter()
    iters = {}
    lb = LB
    ub = 0
    xbest = x
    nbseriousstep = 0
    for k in range(MAX_ITER):
        fx, gx, sx = oracle(x)
        sgmax = max(abs(s) for s in gx)
        logging.debug(f"it {k}: f = {fx:.5f}, |sg|={sgmax:.5f},  time={time.perf_counter()-starttime:0.2f}")
        if ub > fx:
            nbseriousstep +=1
            ub = fx
            xbest = x
            logging.info(f"it {k}({nbseriousstep}): fxc = {ub:.5f}, f = {fx:.5f}, |sg|={sgmax:.5f},  time={time.perf_counter()-starttime:0.2f}")
        if heuristic:
            heurlb = heuristic()
            logging.debug(f"heurlb {heurlb}")
            if lb < heurlb:
                lb = heurlb
                logging.info(f"new lb {lb}")
        if ub - lb < TOL:
            break
        norm = 0
        for i, sg in enumerate(gx):
            norm += sg*sg
            x[i] -= sg * AGILITY_MAX * (ub - lb) / norm
            iters[k] = (fx, ub, nbseriousstep, lb, heurlb, sgmax, time.perf_counter()-starttime)
    return ub, xbest, iters, None


if __name__ == "__main__":
#standard bundle
    solve_multiregion_times(instance=FRES, PLOT=True)
#standard subgradient
#    solve_multiregion_times(instance=FRES, bundle_otherwise_sg=False, heuristic=True, PLOT=True)

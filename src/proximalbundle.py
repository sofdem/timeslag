#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:59:40 2020

Proximal bundle method by Welington de Oliveira ported to Python.

@author: Sophie Demassey
"""

import math
import gurobipy as gp
from gurobipy import GRB
#import cplex
import time
import logging
#import csv


def inexact_proximal_bundle(xc, oracle, usecplex=False, MAX_ITER=200, MAX_NOISE=10,
                            PROX_INIT=1, PROX_MIN=1e-5, PROX_UPDATE = 2, 
                            TOL = 1e-7,  LINE_SEARCH = 0.1, BDL_SZ_MAX = 500,
                            PRIMAL = True, POSITIVE_QUADRANT = False):
    """Bundle algorithm: minimizes a convex function.
    
    Runs the proximal bundle algorithm starting from point xc and 
    returns a minimizer of the convex function oracle(x) within given tolerance and iteration number limit.
    
    Args:
        xc: the starting point.
        MAX_ITER = 200: maximum iteration number
        MAX_NOISE = 10: maximum noise attenuation number by iteration
        PROX_INIT = 1:  initial value of the proximal parameter
        PROX_MIN = 1e-5: minimum value of the proximal parameter
        PROX_UPDATE = 2: proximal update parameter
        TOL = 1e-7: tolerance value 
        LINE_SEARCH = 0.1: line search parameter for the descent test
        BDL_SZ_MAX = 100: maximum bundle size
        PRIMAL = True: find direction subproblem using primal QP approximation or dual ?
        POSITIVE_QUADRANT= False: feasibility set constrained x >= 0 or not ?
        
    Returns:
        x: the last stability center found.
    """
    starttime = time.perf_counter()
    iters = {}
    prox = PROX_INIT
    fxc, gxc, sxc = oracle(xc)
    logging.info(f"init: {fxc}") #), xc, fgxc)
    
    bundle = [[gxc, 0, sxc]]

    nbconseqnullstep = 0
    nbconseqseriousstep = 0
    nbseriousstep = 0
    noiseatt = False
    
    for it in range(MAX_ITER):

        # SOLVE QP: FIND DIRECTION        
        for k in range(MAX_NOISE):
            # decr_predict = f(xc) - fmodel(x), proximity = |xc-x|^2/2t = (t/2)*|sgagg|^2
            x, mu, decr_predict, proximity, sgagg, erragg =  find_direction_primal(bundle, xc, prox, POSITIVE_QUADRANT, usecplex) if (PRIMAL) \
                                                        else find_direction_dual(bundle, xc, prox, POSITIVE_QUADRANT)
            #normsgagg = math.sqrt((2/prox) * proximity)
            normsgagg = max(abs(s) for s in sgagg)
            logging.debug(f"direction: {decr_predict:.5f} {proximity:.5f}") #, x

            tol = TOL * (1 + abs(fxc))
            # alternative stopping test: if (erragg + sgagg.xc <= tol) and (normsgagg <= 1000 * tol)
            if (erragg <= tol) and (normsgagg <= 2):
                logging.info(f"STOP ! erragg: {erragg}, |sgagg|: {normsgagg}")
                solagg = aggregate_primal_solution(bundle, mu)
                return fxc, xc, iters, solagg

            # noise attenuation [Kiwiel06]; decuple t if erragg is overly negative
            if (erragg <= -0.999 * 2 * proximity):
                prox *= 10
                # nbconseqnullstep = 0 #### WHY ??
                noiseatt = True
                logging.debug(f'noise attenuation, proximal parameter = {prox}')
            else:
                break
        
        # ORACLE
        fx, gx, sx = oracle(x)
        logging.debug(f"oracle: {fx}") #, gx) #x, gx)
        violations = [g for g in gx if g > 1e-5]
        logging.debug(f"violations: nb= {len(violations)}/{len(gx)}, max = {max(violations):.3f}")
        bdlsize = len(bundle)
        proxtmp = 2 * prox * (1 + (fxc-fx)/decr_predict)
    
        # DESCENT TEST
        if (fx <= fxc - LINE_SEARCH * decr_predict): # SERIOUS STEP
            for k in range(bdlsize):
                bundle[k][1] += fx - fxc + sum(bundle[k][0][i]*(xc[i]-xi) for i, xi in enumerate(x)) 
            errx = 0
            xc  = x
            fxc = fx
            nbconseqseriousstep += 1
            nbconseqnullstep = 0
            nbseriousstep += 1
            noiseatt = False
            prox = min(10*prox, proxtmp*PROX_UPDATE if nbconseqseriousstep > 5 else proxtmp)
            logging.info(f"It {it}({nbseriousstep}): fxc={fxc:.5f} prox={prox:.5f} err={erragg:.5f} |sg|={normsgagg:.5f} time={time.perf_counter()-starttime:0.2f}")
        else:  # NULL STEP
            errx = fxc - fx + sum(gx[i]*(xi-xc[i]) for i, xi in enumerate(x))
            nbconseqseriousstep = 0
            nbconseqnullstep += 1
            # prox does not decrease in the null steps consecutive to noise attenuation
            if nbconseqnullstep > 50:
                noiseatt = False
            if nbconseqnullstep > 0 and noiseatt == False:
                prox = min(prox, max(proxtmp, prox/PROX_UPDATE, PROX_MIN)) 
                logging.debug(f"It {it}: serious={nbseriousstep} prox={prox:.5f} err={erragg:.5f} |sg|={normsgagg:.5f} time={time.perf_counter()-starttime:0.2f}")
                
        iters[it] = (fx, fxc, nbseriousstep, prox, erragg, normsgagg, time.perf_counter()-starttime)
        #with open("Results.csv", "a+", newline="") as output:
        #    output_writer=csv.writer(output)
        #    output_writer.writerow(iters[it])

        # UPDATE BUNDLE
        if len(bundle) < BDL_SZ_MAX:
            bundle.append([gx, errx, sx])
        else: 
            ### !!!!!!!!! QUESTION !!!!!!!!!! in the constrained case: 
            ### sgagg == sum_k mu_k.sg^k - v and erragg = sum_k mu_k.sg^k + v.xc ;
            ###  should we remove the terms in v (dual of x>=0) before to add it to the bundle ?
            solagg = aggregate_primal_solution(bundle, mu)
            compress_bundle(bundle, [gx, errx, sx], [sgagg, erragg, solagg], mu) 
    logging.info('max iter reached')
    return fxc, xc, iters, None



def find_direction_primal(bundle, xc, t, positive, usecplex):
    return find_direction_primal_cplex(bundle, xc, t, positive) if usecplex \
        else find_direction_primal_gurobi(bundle, xc, t, positive)



def find_direction_primal_cplex(bundle, xc, t, positive):
    """Finds the next iterate by solving the primal QP approximation using Gurobi.
    
    Solves QP: z* = min (y + sum_i d_i^2/2t) st {d_i >= -xc_i,} y >= sum_i sg^k_i.d_i - err^k forall k 
    with {..} only in the 'positive' {x >= 0} constrained case 
    z* + f(xc) = min_{x>=0} fmodel(x) + |x-xc|^2/2t; d* = x*-xc; y* + f(xc) = fmodel(x*)
    with fmodel(x) = max_k f(x^k) + <sg^k, x-x^k> = max_k f(xc) + <sg^k, x-xc> - err^k
    
    Args:
        bundle: the  bundle [[sg^k, err^k] for k]
        xc: the stability center
        t: the proximal parameter
        positive: constrains x = xc+d >=0 or not ?
        
    Returns:
        x*: the next iterate = xc + d*
        mu*: the QP dual values [len(bundle)]
        decr*: the predicted decrease = f(xc) - fmodel(x*) = -y*
        proximity*: the proximal term = |d*|^2/2t
        sgagg*: the aggregate subgradient of fmodel at x* = -d*/t
        erragg*: the aggregate linearization error = f(xc) - (fmodel(x*) + <sgagg*,d*>) = - y* - |d*|^2/t
    """
    
    bdlsz = len(bundle)
    dim = len(xc)
    
    model = cplex.Cplex()
    model.set_log_stream(None)
    model.set_results_stream(None)


    ctrs = model.linear_constraints.add(rhs=[-b[1] for b in bundle], senses=['G'] * bdlsz)

    y = model.variables.add(obj=[1.0], lb=[-cplex.infinity], columns=[[ctrs, [1.0] * bdlsz]]) #, names=['y'])
    assert y[0] == 0

    lbs = [-v if positive else -cplex.infinity for v in xc]
    cols = [[ctrs, [-b[0][i] for b in bundle]] for i in range(dim)]
    d = model.variables.add(lb=lbs, columns=cols, names=[f'd{i}' for i in range(dim)])
    assert d[-1] == dim
    coef = 1/t #(2*t) cplex divides by 2 by default
    for i in range(1, dim+1):
        model.objective.set_quadratic_coefficients(i, i, coef)
    model.objective.set_sense(model.objective.sense.minimize)
    model.solve()

    if(model.solution.get_status() != model.solution.status.optimal):
            logging.warning(f"Wlo says: Deu merda ! non optimal status: {model.solution.get_status()}")
    vals = model.solution.get_values()
    x = [xc[i] + vals[i+1] for i in range(dim)]
    sgagg = [-vals[i+1]/t for i in range(dim)]
    decr_predict = -vals[0]
    decr_nominal = - model.solution.get_objective_value()
    proximity = decr_predict - decr_nominal
    erragg = decr_predict - 2*proximity
    mu   = model.solution.get_dual_values()
    assert len(mu) == len(ctrs)
    model.end()
    return x, mu, decr_predict, proximity, sgagg, erragg


def find_direction_primal_gurobi(bundle, xc, t, positive):
    """Finds the next iterate by solving the primal QP approximation using Gurobi.
    
    Solves QP: z* = min (y + sum_i d_i^2/2t) st {d_i >= -xc_i,} y >= sum_i sg^k_i.d_i - err^k forall k 
    with {..} only in the 'positive' {x >= 0} constrained case 
    z* + f(xc) = min_{x>=0} fmodel(x) + |x-xc|^2/2t; d* = x*-xc; y* + f(xc) = fmodel(x*)
    with fmodel(x) = max_k f(x^k) + <sg^k, x-x^k> = max_k f(xc) + <sg^k, x-xc> - err^k
    
    Args:
        bundle: the  bundle [[sg^k, err^k] for k]
        xc: the stability center
        t: the proximal parameter
        positive: constrains x = xc+d >=0 or not ?
        
    Returns:
        x*: the next iterate = xc + d*
        mu*: the QP dual values [len(bundle)]
        decr*: the predicted decrease = f(xc) - fmodel(x*) = -y*
        proximity*: the proximal term = |d*|^2/2t
        sgagg*: the aggregate subgradient of fmodel at x* = -d*/t
        erragg*: the aggregate linearization error = f(xc) - (fmodel(x*) + <sgagg*,d*>) = - y* - |d*|^2/t
    """
    
    bdlsz = len(bundle)
    dim = len(xc)

    
    model = gp.Model('bdldir')
    model.Params.OutputFlag = 0
    d = model.addVars(dim,lb=-GRB.INFINITY, name='d')
    y = model.addVar(lb=-GRB.INFINITY, name='y')
    obj = y
    for i in range(dim):
        if (positive): 
            d[i].lb = - xc[i]
        obj += d[i]*d[i]/(2*t)
    model.setObjective(obj, GRB.MINIMIZE)
    bdlctr = model.addConstrs(y >= gp.quicksum(bundle[k][0][i]*d[i] for i in range(dim)) - bundle[k][1] for k in range(bdlsz))

    model.optimize()
    if(model.Status != GRB.OPTIMAL):
            logging.warning('Deu merda ! -- Wlo')
    x = [xc[i] + d[i].x for i in range(dim)]
    sgagg = [-d[i].x/t for i in range(dim)]
    decr_predict = -y.x
    decr_nominal = - obj.getValue()
    proximity = decr_predict - decr_nominal
    erragg = decr_predict - 2*proximity
    mu   = [bdlctr[k].Pi for k in range(bdlsz)]
    return x, mu, decr_predict, proximity, sgagg, erragg





def find_direction_dual(bundle, xc, t, positive):
    """Finds the next iterate by solving the opposite dual of the QP approximation using Gurobi.
    
    Solves DP: l* = min (t/2).|{v} - sum_k u^k.sg^k|^2 + sum_k u^k.err^k {+ sum_i v_i.xc_i} st sum_k u^k = 1, u^k >=0, {v_i >= 0}
    with {..} only in the 'positive' {x >= 0} constrained case 
    l* = -z* = f(xc) - fmodel(x*) - |x*-xc|^2/2t = - y* - |d*|^2/2t = (t/2).|{v*} - sum_k u*^k.sg^k|^2 + sum_k u*^k.err^k {+ v*.xc}
    lagrangian optimality condition: d* = x*-xc = t({v*} - sum_k (u*^k.sg^k)); u* = mu* (QP duals)
    y* = fmodel(x*) - f(xc) = - |d*|^2/t - (u*.err {+ v*.xc}) = - (t/2)|{v*}-sum_k u^k.sg^k|^2 - (u*.err {+ v*.xc})
    
    Args:
        bundle: the  bundle [[sg^k, err^k] for k]
        xc: the stability center
        t: the proximal parameter
        positive: constrains x = xc+d >=0 or not ?
        
    Returns:
        x*: the next iterate = xc + d* = xc - t.sgagg*
        mu*: the QP dual values = u* [len(bundle)]
        decr*: the predicted decrease = f(xc) - fmodel(x*) = l* + |d*|^2/2t = l* + (t/2)*|sgagg*|^2
        proximity*: the proximal term = |d*|^2/2t = (t/2)*|sgagg*|^2
        sgagg*: the aggregate subgradient of fmodel at x* = sum_k (u*^k.sg^k) {- v*} = -d*/t
        erragg*: the aggregate linearization error = sum_k u*^k.err^k {+ v*.xc} = l* - (t/2).|sgagg*|^2 = f(xc) - (fmodel(x*) + <sgagg*,d*>)
    """
    bdlsz = len(bundle)
    dim = len(xc)
    
    model = gp.Model('bdldualdir')
    model.Params.OutputFlag = 0
    
    u = model.addVars(bdlsz, lb=0, ub=1, name='u')
    linobj = gp.quicksum(bundle[k][1]*u[k] for k in range(bdlsz))
    
    if positive:
        v = model.addVars(dim, lb=0, name='v')
        linobj += gp.quicksum(xc[i] * v[i] for i in range(dim))
        sghat = [-v[i] + gp.quicksum(u[k]*bundle[k][0][i] for k in range(bdlsz)) for i in range(dim)]
    else:
        sghat = [gp.quicksum(u[k]*bundle[k][0][i] for k in range(bdlsz)) for i in range(dim)]
        
    quadobj = (t/2) * gp.quicksum(sghat[i]*sghat[i] for i in range(dim))
    model.setObjective(linobj + quadobj, GRB.MINIMIZE) 
    model.addConstr(gp.quicksum(u[k] for k in range(bdlsz)) == 1)
    model.optimize()
    if(model.Status != GRB.OPTIMAL):
            logging.error('Deu merda ! -- Wlo')
            
    sgagg = [sghat[i].getValue() for i in range(dim)]
    x = [xc[i] - t * sgagg[i] for i in range(dim)]
    erragg = linobj.getValue()
    proximity = quadobj.getValue()
    #decr_nominal = erragg + proximity
    decr_predict = erragg + 2*proximity
    mu   = [u[k].x for k in range(bdlsz)]

    return x, mu, decr_predict, proximity, sgagg, erragg



def aggregate_primal_solution(bundle, mu):
    solagg = {}
    for var in bundle[0][2].keys():
        if isinstance(bundle[0][2][var], list):
            nbvars = len(bundle[0][2][var])
            solagg[var] = [sum(v * bundle[k][2][var][i] for k, v in enumerate(mu) if abs(v) > 1e-10)
                           for i in range(nbvars)]
        else:
            solagg[var] = sum(v * bundle[k][2][var] for k, v in enumerate(mu) if abs(v) > 1e-10)
    return solagg
    #return [sum(v * bundle[i][2][j] for i, v in enumerate(mu) if v > 1e-10) for j in range(dim)]


def compress_bundle(bundle, bx, bagg, mu) :
    """Update the bundle when it is full.

    When the maximumal bundle size is reached, remove the inactive planes (mu_k=0),
    add the aggregate plane and the last computed plane.

    Args:
        bundle: the  full bundle [[sg^k, err^k] for k] ie. len(bundle) is maximal
        bx: the new plane [sgx, errx]
        bagg: the aggregate plane [sgagg, erragg]
        mu: the QP dual values [len(bundle)]
    """        

    TOL = 1e-7
    bdlsz = len(bundle)
    
    # remove all the inactive planes from the bundle    
    bundle[:] = [bundle[i] for i, v in enumerate(mu) if v > TOL]
    
    # if all planes are active: replace the two less active
    if len(bundle) == bdlsz:
        idxmin, idxmin2 = twominidx(mu)
        bundle[idxmin]  = bagg
        bundle[idxmin2] = bx
    else:
        # add the last computed plane 
        bundle.append(bx)
        # there is room to also add the aggregated plane
        if len(bundle) < bdlsz:
            bundle.append(bagg)


def twominidx(a):
    """Returns the indexes of the first and second minimum entries of array a. """ 
    imin = 0 if (a[0]<=a[1]) else 1
    imin2 = 1 - imin
    for i, v in enumerate(a):
        if v < a[imin2]:
            if v < a[imin]:
                imin2 = imin
                imin = i
            else:
                imin2 = i
    return imin, imin2


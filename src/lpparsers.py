#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:40:24 2020

Parsers for regional ETSAP/TIMES LP models and the associated interconnection constraints.

@author: Sophie Demassey
"""

import re
import csv
import sys
import gurobipy as gp
from gurobipy import GRB
import logging
from pathlib import Path
#import cplex

#models = {} # dictionary of gurobi models by region
#duals = []  # list of dualized constraints: co2 emission and interconnection constraints
#aggregate = {} # aggregate model including the dualized constraints


def create(USE_CPLEX=False, *args, **kwargs):
    return _LPCplex(*args, **kwargs) if USE_CPLEX else _LPGurobi(*args, **kwargs)

class _LinProgs:
    """Variables are stored (in duals/capvars) as objects for Gurobi, and as indices for Cplex."""

    @classmethod
    def create(cls, instance:dict, USE_CPLEX: bool, COMPUTE_PRIMAL_SOLUTION):
        return _LPCplex(instance, COMPUTE_PRIMAL_SOLUTION) if USE_CPLEX else _LPGurobi(instance, COMPUTE_PRIMAL_SOLUTION)

    def __init__(self, instance: dict, COMPUTE_PRIMAL_SOLUTION=True):
        self.instance = instance
        self.aggregate = {}
        self.stats = []
        self.PRIMAL_SOLUTION = COMPUTE_PRIMAL_SOLUTION
        self.co2limits = self._parse_CO2_limits()

    """Create the aggregate model and store the capacity variables (heuristic purpose)."""
    def aggregate_models(self):
        self.aggregate['model'] = self._build_aggregate_primal_model()
        self.aggregate['capvars'] = self._parse_capacity_vars()
        return self.aggregate['model']

    def _parse_models(self):
        """Parse the regional LP models and extract the variables and the objective"""
        models = {}
        for reg in self.instance['regions']:
            lppath = Path(self.instance['dir'], self.instance['lpfiles'].format(reg))
            logging.debug(f"read {lppath}")
            models[reg] = self._import_model(str(lppath))
            self._turnlogoff(models[reg])
        return models

    def _parse_CO2_limits(self):
        filename = self.instance.get('co2file')
        co2limits = {}
        if filename:
            csvpath = Path(self.instance['dir'], filename)
            with open(csvpath) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=';')
                for row in csvreader:
                    logging.debug(f'{", ".join(row)}')
                    co2limits[int(row[1])] = float(row[2])
        return co2limits

    def _parse_capacity_vars(self):
        """Collect the capacity variables from the regional models"""
        logging.debug("parse capacity vars")
        capvars = {}
        #vformat = '(VAR_NCAP\[{}.*\])'
        vformat = '(VAR_NCAP\({}.*\))'
        nbvars = 0
        for reg, mr in self.models.items():
            capvareg = []
            for var in self._getvars(mr):
                match = re.match(vformat.format(reg), self._varname(mr, var))
                if (match):
                    capvareg.append(var)
            capvars[reg] = capvareg
            nbvars += len(capvareg)
        logging.debug(f"{nbvars} capacity vars")
        return capvars

    def _parse_coupling_constraints(self):
        duals = []
        self._parse_interconnections(duals)
        if self.co2limits:
            self._parse_CO2_constraints(duals)
        return duals

    def _parse_CO2_constraints(self, duals):
        for year, limit in self.co2limits.items():
            regvars = {}
            for reg, mr in self.models.items():
                vr = self._var_co2(reg, year)
                assert vr, self._var_co2_name(reg, year)
                assert self._getlb(reg, vr) >= 0
                self._fixub(reg, vr, limit)
                regvars[reg] = vr
            duals.append({"year": year, "co2vars": regvars, "co2limit": limit})


    def _parse_interconnections(self, duals):
        """Collect the dualized interconnection constraints:
        VAR_IRE(reg_imp, year, ts, elchig, IMP) - VAR_IRE(reg_exp, year, ts, elchig, EXP) = 0
        for each interconnection/direction (reg_exp -> reg_imp), for each time (year, ts).""" 
        logging.debug("parse interconnections")
        ub = 1000
        logging.warning(f"set the UB of interconnection variables to {ub} because of solver instability")
        for reg1, mr1 in self.models.items():
            for v in self._getvars(mr1):
                reg2, year, ts = self._parse_var_ire(self._varname(mr1, v), reg1)
                if (reg2 in self.models):
                    # ire(reg1, IMP) - ire(reg2, EXP) = 0                 
                    varimp = v
                    varexp = self._var_ire(reg2, year, ts, reg1, reg2, "EXP")
                    self._fixub(reg1, varimp, ub)
                    self._fixub(reg2, varexp, ub)
                    duals.append({'regimp': reg1, 'regexp': reg2, 'year': year, 'ts': ts, 'varimp': varimp, 'varexp': varexp})
                    # ire(reg2, IMP) - ire(reg1, EXP) = 0                 
                    varimp = self._var_ire(reg2, year, ts, reg1, reg2, "IMP")
                    varexp = self._var_ire(reg1, year, ts, reg1, reg2, "EXP")
                    self._fixub(reg2, varimp, ub)
                    self._fixub(reg1, varexp, ub)
                    duals.append({'regimp': reg2, 'regexp': reg1, 'year': year, 'ts': ts, 'varimp': varimp, 'varexp': varexp})
                elif (reg2):
                    logging.warning(f"WARNING ! region {reg2} not considered")
        logging.info(f"{len(duals)} dualized constraints")

    def _parse_var_ire(self, varname: str, reg: str):
        """If the input variable is an import from reg then retrieves the export region, the year and the timeslice.""" 
        #okformat = 'VAR_IRE\[{},(.*?),(.*?),TB_ELCHIG_{}_(.*?)_01,(.*?),(.*?),IMP\]'.format(reg, reg)
        okformat = 'VAR_IRE\({},(.*?),(.*?),TB_ELCHIG_{}_(.*?)_01,(.*?),(.*?),IMP\)'.format(reg, reg)
        match = re.match(okformat, varname)
        if (match):
            year = match.group(1)
            reg2 = match.group(3)
            timeslice = match.group(5)
            return reg2, year, timeslice
        else:
            return False, False, False
    
    def _var_ire_name(self, reg, year, ts, reg1, reg2, direction):
        """Returns the (reg1-reg2, direction) interconnection variable at time (year, ts) in models[reg]."""
        #return "VAR_IRE[{},{},{},TB_ELCHIG_{}_{}_01,ELCHIG,{},{}]".format(reg, year, year, reg1, reg2, ts, direction)
        return f"VAR_IRE({reg},{year},{year},TB_ELCHIG_{reg1}_{reg2}_01,ELCHIG,{ts},{direction})"

    def _var_co2_name(self, reg, year):
        return f"VAR_COMNET({reg},{year},TOTCO2,ANNUAL)"

    def oraclelr(self, u):
        """Returns the value and a subgradient of the convex function -L to minimize at point u.
        -L(u) = sum_y u[y].co2lim[y]
                - sum_{r} [ min f(r) + sum_{e[r,imp]} u[e].IREIMP[e] - sum_{e[r,exp]} u[e].IREEXP[e] 
                           + sum_{y} u[y].EMI[r,y] ] 
        sg(-L(u)) = ( (IREEXP[e] - IREIMP[e])_{e} , (co2lim[y]-sum_r EMI[r,y])_{y} )
        """
        runtimes={}
        constantterm = self._update_multipliers(u)
        objsum = 0
        primalsol = {}
        for reg, model in self.models.items():
            obj, runtime, niter, status = self._solve(model)
            self._turntoprimalsimplex(model)
            if not obj:
                logging.error(f'Optimization of model {reg} was stopped with status {status}')
                sys.exit(1)
            #logging.info(f'{reg}: {runtime:.2f} s, {round(niter)} it')
            runtimes[reg] = (runtime, round(niter))
            self.stats.append((reg, obj, runtime))
            objsum += obj
            if self.PRIMAL_SOLUTION:
                primalsol[reg] = self._save_primal_solution(model)
        reg = max(runtimes, key=runtimes.get)
        logging.debug(f'slowest subproblem = {reg}: {runtimes[reg][0]:.2f} s, {runtimes[reg][1]} it')

        subgradient = self._opposite_constraint()
        return constantterm-objsum, subgradient, primalsol

    def solve_aggregate(self):
        if not 'model' in self.aggregate:
            logging.info('aggregate models')
            self.aggregate["model"] = self._build_aggregate_primal_model()
        logging.info('solve the aggregate model')
        obj, time, niter, status = self._solve(self.aggregate['model'])
        logging.info(f"opt={obj:.8f}, {time:.2f} s, {niter} it, {status}")

    def test_primal_solution(self, primalsol, tol=1e-4):
        self.aggregate["model"] = self._build_aggregate_primal_model()
        self._set_var_bounds(primalsol, tol)
        logging.info('test the primal solution wrt the aggregate model')
        obj, time, niter, status = self._solve(self.aggregate['model'])
        logging.info(f"opt={obj:.8f}, {time:.2f} s, {niter} it, {status}")


class _LPGurobi(_LinProgs):

    def __init__(self, instance: dict, COMPUTE_PRIMAL_SOLUTION):
        _LinProgs.__init__(self, instance, COMPUTE_PRIMAL_SOLUTION)
        self.models = self._parse_models()
        self.duals = self._parse_coupling_constraints()

    @classmethod
    def _import_model(self, lppath):
        return gp.read(lppath)

    def _turnlogoff(self, model):
        model.setParam(GRB.Param.OutputFlag, 0)

    def _turntoprimalsimplex(self, model):
        model.setParam(GRB.Param.Method, 0)
#        model.setParam(GRB.Param.FeasibilityTol, 1e-2)
#        model.setParam(GRB.Param.OptimalityTol, 1e-2)

    def _getvars(self, model):
        return model.getVars()

    def _varname(self, model, var):
        return var.varName

    def _fixub(self, reg, var, ub):
        if var.ub >= GRB.INFINITY:
            var.ub = ub

    def _getlb(self, reg, var):
        return var.lb

    def _var_ire(self, reg, year, ts, reg1, reg2, direction):
        return self.models[reg].getVarByName(self._var_ire_name(reg, year, ts, reg1, reg2, direction))

    def _var_co2(self, reg, year):
        return self.models[reg].getVarByName(self._var_co2_name(reg, year))

    def _save_primal_solution(self, model):
        return [v.x for v in model.getVars()]
            #for i, v in enumerate(self.models[reg].getVars()):
            #    primalsol[reg, i] = v.x

    def _build_aggregate_primal_model(self):
        """Parse the regional LP models and return the aggregate multiregional LP model with the interconnections added."""
        m = False
        self.aggregate["offset"] = {}
        for reg, mr in self.models.items():
            mr.update()
            if m:
                self.aggregate["offset"][reg] = m.numVars
                vareg = {}
                for v in mr.getVars():
                    vareg[v.VarName] = m.addVar(lb=v.lb, ub=v.ub, obj=v.obj, vtype=v.vtype, name=v.VarName)
                for c in mr.getConstrs():
                    expr = mr.getRow(c)
                    newexpr = gp.LinExpr()
                    for i in range(expr.size()):
                        newexpr.add(vareg[expr.getVar(i).VarName], expr.getCoeff(i))
                    m.addConstr(newexpr, c.Sense, c.RHS, name=c.ConstrName)
            else:
                self.aggregate["offset"][reg] = 0
                m = mr.copy()
        m.update()

        for d in self.duals:
            co2vars = d.get("co2vars")
            if co2vars:
                co2limit = d["co2limit"]
                m.addConstr(gp.quicksum(co2vars.values()) <= co2limit)
            else:
                varimp = m.getVarByName(d['varimp'].VarName)
                varexp = m.getVarByName(d['varexp'].VarName)
                m.addConstr(varimp <= varexp)

        m.setParam(GRB.Param.OutputFlag, 1)
        return m


    def _set_var_bounds(self, primalsol, tol):
       m = self.aggregate["model"]
       for reg, mr in self.models.items():
            offset = self.aggregate['offset'][reg]
            for i, v in enumerate(mr.getVars()):
                var = m.getVars()[i + offset]
                assert var.varName == v.varName
                val = primalsol[reg][i]
                assert v.lb <= val + tol, f"lb = {v.lb}, val= {val}, tol = {tol}"
                var.lb = max(v.lb, val-tol)
                assert v.ub >= val - tol, f"ub = {v.ub}, val= {val}, tol = {tol}"
                var.ub = min(v.ub, val+tol)

    def _update_multipliers(self, u):
        constantterm = 0
        for i, v in enumerate(u):
            d = self.duals[i]
            co2vars = d.get("co2vars")
            if co2vars:
                constantterm += d["co2limit"] * v
                for co2var in co2vars.values():
                    co2var.obj = v
            else:
                d["varimp"].obj = v
                d["varexp"].obj = -v
        return constantterm

    def _solve(self, model):
        model.optimize()
        status = model.status
        if status == GRB.UNBOUNDED:
            logging.warning('model unbounded')
            for v in model.getVars():
                if abs(v.UnbdRay) > 1e-5:
                    logging.warning(v.VarName, ", ray =", v.UnbdRay)
        elif status == GRB.INFEASIBLE:
            #model.computeIIS()
            #logging.warning('IIS is minimal' if model.IISMinimal else 'IIS is not minimal')
            #logging.warning('The following constraint(s) cannot be satisfied:')
            #for c in model.getConstrs():
            #    if c.IISConstr:
            #        logging.warning(c.constrName)
            model.Params.FeasibilityTol = 1.e-4
            logging.warning("update feasibility tol and resolve")
            model.optimize()
            status = model.status
        obj = model.ObjVal if status == GRB.OPTIMAL else None
        return obj, model.Runtime, model.IterCount, status

    def _opposite_constraint(self):
        ctr = []
        for d in self.duals:
            co2vars = d.get("co2vars")
            if co2vars:
                ctr.append(d["co2limit"] - sum(v.x for v in co2vars.values()))
            else:
                ctr.append(d["varexp"].x - d["varimp"].x)
        return ctr

    def aggregate_heuristic(self):
        """ A heuristic for the subgradient algorithm solving the aggregate primal model with the interconnections
        after fixing all the capacity 'NCAP' variables to their values in the lagrangian subproblem."""
        m = self.aggregate['model']
        self._turnlogoff(m)
        for reg, capvars in self.aggregate['capvars'].items():
            for v in capvars:
                var = m.getVarByName(v.varName)
                var.ub = v.x
                var.lb = v.x
        m.optimize()
        ub = m.ObjVal
        return -ub




class _LPCplex(_LinProgs):

    def __init__(self, instance: dict, COMPUTE_PRIMAL_SOLUTION):
        _LinProgs.__init__(self, instance, COMPUTE_PRIMAL_SOLUTION)
        self.models = self._parse_models()
        self.duals = self._parse_coupling_constraints()

    def _import_model(self, lppath):
        return cplex.Cplex(lppath)

    def _turnlogoff(self, model):
        model.set_log_stream(None)
        model.set_results_stream(None)

    def _turntoprimalsimplex(self, model):
        model.parameters.lpmethod.set(model.parameters.lpmethod.values.primal)

    def _getvars(self, model):
        return range(model.variables.get_num())

    def _varname(self, model, var):
        return model.variables.get_names(var)

    def _fixub(self, reg, var, ub):
        if self.models[reg].variables.get_upper_bounds(var) >= cplex.infinity:
            self.models[reg].variables.set_upper_bounds(var, ub)

    def _getlb(self, reg, var):
        return self.models[reg].variables.get_lower_bounds(var)

    def _var_ire(self, reg, year, ts, reg1, reg2, direction):
        return self.models[reg].variables.get_indices(self._var_ire_name(reg, year, ts, reg1, reg2, direction))

    def _var_co2(self, reg, year):
        return self.models[reg].variables.get_indices(self._var_co2_name(reg, year))

    def _save_primal_solution(self, model):
        return model.solution.get_values()
#            for v in range(self.models[reg].variables.get_num()):
#               primalsol[reg, v] = self.models[reg].solution.get_values(v)

    def _build_aggregate_primal_model(self):
        """Parse the regional LP models and return the aggregate multiregional LP model with the interconnections added."""
        m = cplex.Cplex()
        self.aggregate["offset"] = {}

        for reg, mr in self.models.items():
            self.aggregate["offset"][reg] = m.variables.get_num()
            vnames = mr.variables.get_names()
            vlbs = mr.variables.get_lower_bounds()
            vubs = mr.variables.get_upper_bounds()
            #vtypes = mr.variables.get_types()
            vobjs = mr.objective.get_linear()
            vind = m.variables.add(obj=vobjs, lb=vlbs, ub=vubs, names=vnames)

            nct = mr.linear_constraints.get_num()
            for ct in range(nct):
                ctsense =  mr.linear_constraints.get_senses(ct)
                ctrhs =  mr.linear_constraints.get_rhs(ct)
                ctrange = mr.linear_constraints.get_range_values(ct)
                ind,val = mr.linear_constraints.get_rows(ct).unpack()
                ctrow = cplex.SparsePair(ind=[i+vind[0] for i in ind], val=val)
                m.linear_constraints.add(senses=[ctsense], rhs=[ctrhs], range_values=[ctrange], lin_expr=[ctrow])

        unitvector = [1.0] * len(self.models)
        for d in self.duals:
            co2vars = d.get("co2vars")
            if co2vars:
                co2limit = d["co2limit"]
                m.linear_constraints.add(lin_expr=[[co2vars.values(), unitvector]], senses=['L'], rhs=[co2limit])
            else:
                varimp = self._varname(self.models[d['regimp']], d['varimp'])
                varexp = self._varname(self.models[d['regexp']], d['varexp'])
                m.linear_constraints.add(lin_expr=[[[varimp, varexp], [1.0, -1.0]]], senses=['E'], rhs=[0.0])
        #m.parameters.lpmethod.set(m.parameters.lpmethod.values.primal)
        return m

    def _set_var_bounds(self, primalsol, tol):
       m = self.aggregate["model"]
       for reg, mr in self.models.items():
            offset = self.aggregate['offset'][reg]
            for vr in range(mr.variables.get_num()):
                namer = mr.variables.get_names(vr)
                lbr = mr.variables.get_lower_bounds(vr)
                ubr = mr.variables.get_upper_bounds(vr)
                v = vr + offset
                assert namer == m.variables.get_names(v)
                val = primalsol[reg][vr]
                assert lbr <= val + tol, f"lb = {lbr}, val= {val}, tol = {tol}"
                m.variables.set_upper_bounds(v, max(lbr, val-tol))
                assert ubr >= val - tol, f"ub = {ubr}, val= {val}, tol = {tol}"
                m.variables.set_upper_bounds(v, min(ubr, val+tol))

    def _update_multipliers(self, u):
        constantterm = 0
        for i, v in enumerate(u):
            d = self.duals[i]
            co2vars = d.get("co2vars")
            if co2vars:
                constantterm += d["co2limit"] * v
                for reg, co2var in co2vars.items():
                    self.models[reg].objective.set_linear(co2var, v)
            else:
                self.models[d["regimp"]].objective.set_linear(d["varimp"], v)
                self.models[d["regexp"]].objective.set_linear(d["varexp"], -v)
        return constantterm

    def _solve(self, model):
        start = model.get_time()
        model.solve()
        status = model.solution.get_status()
        obj = model.solution.get_objective_value() if status == model.solution.status.optimal else None
        if status==model.solution.status.optimal_infeasible:
            logging.warning(f"cplex status= optimal-infeasible: stability issues")
            obj = model.solution.get_objective_value()
        runtime =model.get_time() - start
        niter = model.solution.progress.get_num_iterations()
        return obj, runtime, niter, status

    def _getval(self, dual, imporexp):
        return self.models[dual["reg"+imporexp]].solution.get_values(dual["var"+imporexp])

    def _opposite_constraint(self):
        ctr = []
        for d in self.duals:
            co2vars = d.get("co2vars")
            if co2vars:
                ctr.append(d["co2limit"] - sum(self.models[reg].solution.get_values(v) for reg, v in co2vars.items()))
            else:
                ctr.append(self._getval(d, "exp") - self._getval(d, "imp"))
        return ctr

    def aggregate_heuristic(self):
        """ A heuristic for the subgradient algorithm solving the aggregate primal model with the interconnections
        after fixing all the capacity 'NCAP' variables to their values in the lagrangian subproblem."""
        m = self.aggregate['model']
        self._turnlogoff(m)
        for reg, capvars in self.aggregate['capvars'].items():
            for v in capvars:
                val = self.models[reg].solution.get_values(v)
                var = self.models[reg].variables.get_names(v)
                m.variables.set_upper_bounds(var, val)
                m.variables.set_lower_bounds(var, val)
        m.solve()
        ub = m.solution.get_objective_value()
        return -ub



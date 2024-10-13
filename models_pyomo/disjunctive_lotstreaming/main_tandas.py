import pyomo.environ as pyo
import pandas as pd
import numpy as np
from params import JobShopRandomParams
from disjunctive import DisjModel
import sys

# Instantiate random params of a 4x3 problem
params = JobShopRandomParams(2, 2, 2, seed=10) # m, j, t, seed
params.printParams()
print(params.p_times_tandas)

disj_model = DisjModel(params)

solver = pyo.SolverFactory("glpk", options=dict(cuts="on", sec=40))
res = solver.solve(disj_model, tee=False)
print(res)
 
disj_model.plot_variables()
disj_model.plot()


sys.exit()
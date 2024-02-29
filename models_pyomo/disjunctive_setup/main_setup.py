import pyomo.environ as pyo
import pandas as pd
import numpy as np
from params import JobShopRandomParams
from disjunctive import DisjModel
import sys

# Instantiate random params of a 4x3 problem
params = JobShopRandomParams(3, 3, seed=9) # m, j, seed
params.printParams()

disj_model = DisjModel(params)

solver = pyo.SolverFactory("cbc", options=dict(cuts="on", sec=40))
res = solver.solve(disj_model, tee=False)
print(res)
 
disj_model.plot_variables()
disj_model.plot()


sys.exit()
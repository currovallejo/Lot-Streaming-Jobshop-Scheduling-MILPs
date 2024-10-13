'''
MODEL DEVELOPMENT AND RESULTS REPRESENTATION
----------------------------------------------------------------------
- This is the main script
- The data is obtained, the model is defined, solved and the results are printed in a Dataframe and in a Gantt.
'''

#--------- LIBRARIES ---------
import gurobipy as gp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from xlsxwriter import Workbook

#--------- OTHER PYTHON FILES USED ---------
from params import JobShopRandomParams, job_params_from_json
import gantt_jobshop_px as gantt
import model_v9
import plotting

#--------- GET JOBSHOP DATA ---------
# problem data (machines, jobs, batches, seed= ) is obtained calling Jobshop
# can be obtained by 
    # 1. by generating random data indicating the seed 
    # 2. by obtaining the data from a json file (previously generated in the same way, with the advantage that it can be modified) --> method job_params_from_json

def getJobShopParameters(machines:int, jobs:int, maxlots:int, seed:int):
    params = JobShopRandomParams(machines, jobs, maxlots, seed=seed) # generate jobshop instance
    # params = job_params_from_json('ist_2m_4j_2u_seed7_v67.json')

    return params

#--------- MAIN ---------
def main():
    # get and print data
    d = 100
    demand={0:d,1:d,2:d,3:d,4:d,5:d}
    params = getJobShopParameters(machines=3, jobs=3, maxlots=3, seed=5)
    plotting.printData(params, demand)
 
    model, variables = model_v9.build_model_v9(params, demand)
    model, variables = model_v9.solve_model(model, variables, objective=variables['C'], timelimit=1200)
    df_results = model_v9.build_solution_df(variables, params)
    print(df_results)
    plotting.plot_pxt_gantt(df_results,params,d, show=True, version=0)

    # resultsDf = gurobiModel(params,demand)

    # gantt.print_pxt_gantt(resultsDf)

if __name__=="__main__":
    main()
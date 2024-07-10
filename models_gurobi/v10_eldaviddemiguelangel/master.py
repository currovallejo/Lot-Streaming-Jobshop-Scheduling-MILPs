'''
MODEL DEVELOPMENT AND RESULTS REPRESENTATION
----------------------------------------------------------------------
- This is the main script
- The data is obtained, the model is defined, solved and the results are printed in a Dataframe and in a Gantt.
'''

#--------- LIBRARIES ---------
import numpy as np

#--------- OTHER PYTHON FILES USED ---------
from params import JobShopRandomParams, job_params_from_json
import model_v10
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
    d = 50
    demand={0:d,1:d,2:d,3:d,4:d,5:d}
    params = getJobShopParameters(machines=3, jobs=3, maxlots=3, seed=3)
    plotting.printData(params, demand)
    shift_time = 400

    model, variables = model_v10.build_model_v10(params, demand, shift_time)
    # model, variables = model_v10.solve_model(model, variables, objective=variables['C'], timelimit=120)

    df_results = model_v10.build_solution_df(variables, params)
    print(df_results)
    plotting.plot_pxt_gantt(df_results,params,d, shift_time, show=True, version=0)

if __name__=="__main__":
    main()
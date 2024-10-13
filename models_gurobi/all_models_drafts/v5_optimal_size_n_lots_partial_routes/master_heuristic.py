'''
MODEL DEVELOPMENT AND RESULTS REPRESENTATION
----------------------------------------------------------------------
- This is the main script
- The data is obtained, the model is defined, solved and the results are printed in a Dataframe and in a Gantt.
'''
#--------- PYTHON SCRIPTS ---------
from params import JobShopRandomParams, job_params_from_json
import plotting as pltt
import model_v5 as m5

#--------- GET JOBSHOP DATA ---------
# problem data (machines, jobs, batches, seed= ) is obtained calling Jobshop
# can be obtained by 
    # 1. by generating random data indicating the seed 
    # 2. by obtaining the data from a json file (previously generated in the same way, with the advantage that it can be modified) --> method job_params_from_json

def getJobShopParameters(machines:int, jobs:int, batches:int, seed:int):
    params = JobShopRandomParams(machines, jobs, batches, seed=seed) # generate jobshop instance
    # params = job_params_from_json('ist_2m_4j_2u_seed7_v67.json')

    return params

#--------- MAIN ---------
def main():
    d = 1000  # demand for each job
    demand = {0:d,1:d,2:d,3:d,4:d,5:d,6:d,7:d}
    n_machines, n_jobs, maxlots, seed = 3, 3, 5, 5
    params = getJobShopParameters(n_machines, n_jobs, maxlots, seed) #2,2,4,3

    # plot job shop problem data
    pltt.printData(params, demand)

    # get model with constraints and objective function
    model, variables = m5.model_v5(params,demand)

    # solve model
    model, variables, elapsed_times, objectives, gaps = m5.solve_model(model, variables, timelimit=3600, plot_evolution=True)

    # plot solution evolution
    pltt.plot_solution_evolution(elapsed_times, objectives, gaps, show=False)

    # get results
    df_results = m5.get_model_df_results(model, variables, params)
 
    # plot gantt chart
    pltt.plot_pxt_gantt(df_results,params,d, show=True, version=0)

    # improve results (eliminate idle time)
    model, variables = m5.model_v5_improvement(model, variables, params, demand)

    # # solve improved model
    model, variables, *_ = m5.solve_model(model, variables, timelimit=3600, plot_evolution=False)

    # get results
    df_results = m5.get_model_df_results(model, variables, params)

    # plot gantt chart
    pltt.plot_pxt_gantt(df_results,params,d, show=True, version=2)

if __name__=="__main__":
    main()
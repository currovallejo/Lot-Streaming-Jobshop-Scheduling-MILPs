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
import time
import math
from xlsxwriter import Workbook
from matplotlib.ticker import FuncFormatter, MultipleLocator


#--------- OTHER PYTHON FILES USED ---------
from params import JobShopRandomParams
import gantt_jobshop_px as gantt

#--------- GET JOBSHOP DATA ---------
# problem data (machines, jobs, batches, seed= ) is obtained calling Jobshop
# can be obtained by 
    # 1. by generating random data indicating the seed 
    # 2. by obtaining the data from a json file (previously generated in the same way, with the advantage that it can be modified) --> method job_params_from_json

def getJobShopParameters(machines:int, jobs:int, batches:int, seed:int):
    params = JobShopRandomParams(machines, jobs, batches, seed=seed) # generate jobshop instance
    # params = job_params_from_json('ist_2m_4j_2u_seed7_v67.json')

    return params

#--------- MILP MODEL ---------
def gurobiModel(params, demand:dict): #builds and solve the model
    
    # renaming parameters
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    setup_time = params.setup
    seq = params.seq
    lotes = params.lotes
    s = setup_time

    # sets definition
    set={}
    set["batches_machines"] = doble_um = gp.tuplelist([(u,m) for u in params.lotes for m in params.machines]) # usado para plotting, no para el modelo
    set["jobs_batches"] = doble_ju = gp.tuplelist([(j,u) for j in params.jobs for u in params.lotes])
    set["machines_jobs_batches"] = triple_mju = gp.tuplelist([(m,j,u) for m in params.machines for j in params.jobs if m in params.seq[j] for u in params.lotes])
    set["machines_jobs_jobs_batches_batches"] = penta_mklju = gp.tuplelist([(m,k,l,j,u) for m in params.machines for k in params.jobs for j in params.jobs if (m in params.seq[j] and m in params.seq[k]) for l in params.lotes for u in params.lotes if j!=k or u!=l])

    # print("[SETS] are: \n", set)
    
    # model initialization
    model = gp.Model('Jobshop')

    # continuous decision variables
    p = model.addVars(set["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'p')
    x = model.addVars(set["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'x')
    y = model.addVars(set["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'y')
    z = model.addVars(penta_mklju,vtype=gp.GRB.BINARY, name = 'z')
    C = model.addVar(vtype=gp.GRB.INTEGER, name = 'C')
    q = model.addVars(doble_ju,vtype=gp.GRB.INTEGER, name = 'q')
    print('la variable q', q)
    Q = model.addVars(doble_ju,vtype=gp.GRB.BINARY, name = 'Q')

    V = sum(process_time[(m,j)]*q[j,u] for m,j,u in set["machines_jobs_batches"])

    # constraints
    for u in lotes:
        for j in jobs:
            for m in machines:
                if m in seq[j]:
                    if seq[j].index(m) > 0:
                        # Find the previous machine 'o' in the sequence
                        o = seq[j][seq[j].index(m) - 1]
                        # Add the constraint using 'o'
                        model.addConstr(y[m, j, u] >= x[o, j, u] + p[o, j, u])#1

    model.addConstrs(y[m,j,u]>=x[m,j,u]-s[m,j] for m,j,u in triple_mju) #5

    for j in jobs:
        for m in machines:
            if m in seq[j]:
                model.addConstrs((x[m,j,u]>=x[m,j,u-1]+p[m,j,u-1]) for u in lotes if u!=0) #2
    
    # for j in jobs:
    #     for m in machines:
    #         if m in seq[j]:
    #             model.addConstrs((y[m,j,u]>=x[m,j,u-1]) for u in lotes if u!=0) #2'
                
    model.addConstrs((y[m,j,u] + V*(1-z[m,k,l,j,u])-x[m,k,l]-p[m,k,l]>=0) for m,k,l,j,u in penta_mklju) #3
    model.addConstrs(z[m,k,l,j,u]+z[m,j,u,k,l]==1 for m,k,l,j,u in penta_mklju) #4
    model.addConstrs(x[m,j,u]>=y[m,j,u]+s[m,j]*Q[j,u] for m,j,u in triple_mju) #5

    model.addConstrs(C>=x[m,j,u]+p[m,j,u] for m,j,u in triple_mju) #6

    model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju) #7
    model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju) #8

    model.addConstrs(p[m,j,u]==process_time[(m,j)]*q[j,u] for m,j,u in triple_mju) #10

    model.addConstrs(gp.quicksum(q[j,u] for u in lotes)==demand[j] for j in jobs) #11

    model.addConstrs(q[j, u] <= V*Q[j,u] for j, u in doble_ju) #12
    model.addConstrs(Q[j,u] <= q[j, u] for j, u in doble_ju) #13
    model.addConstrs(Q[j,u]<=Q[j,u-1] for j,u in doble_ju if u>0) #14
    
    # Set the maximum solving time in seconds
    model.setParam('TimeLimit', 3600)

    # Set model objective (minimize makespan)
    model.setObjective(C,gp.GRB.MINIMIZE)

    # Solve model
    # model.optimize()
    optimize_and_plot_mip(model, callback_interval=15)

    print('la variable q despues de optimizar', q)

    # Check the optimization status
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found!")
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Optimization terminated due to time limit.")
    else:
        print("No solution found within the time limit.")

    print("\n El makespan máximo es: ", C.X, "\n")

    # get the run_time
    run_time = model.Runtime

    # get the MIP gap
    optimality_gap = model.MIPGap

    #--------- DATA FRAME -----------
    results = []

    # Iterate through the variables and store the variable values in a list
    for u in lotes:
        for j in jobs:
            for m in machines:
                if (m in seq[j] and q[j,u].X>0):
                    results.append({
                        'Job': j,
                        'Lote': u,
                        'Machine': m,                                
                        'Setup Time': s[m, j],
                        'Processing Time': process_time[(m, j)],
                        'Processing Time for u':p[m,j,u].X,
                        'Start Time (x)': x[m, j, u].X,
                        'Setup Start Time (y)': y[m, j, u].X,
                        'Lotsize' : q[j,u].X,
                        'makespan':x[m,j,u].X + p[m,j,u].X
                    })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    # Set the Pandas option to display all rows (max_rows=None)
    pd.set_option('display.max_rows', None)

    # Print the DataFrame
    # print(results_df)

    # Export the DataFrame to an Excel file
    # with pd.ExcelWriter(r'C:\Users\Usuario\Documents\MII+MOIGE\TFM - LOCAL/RESULTADOS.xlsx', engine='xlsxwriter') as writer:
    # results_df.to_excel(writer, sheet_name='Sheet1', index=False)

    return results_df, run_time, optimality_gap

#--------- PRINT DATA AND RESULTS ---------
def printData(params, demand):
    print("--------- JOBSHOP DATA --------- \n")
    # job shop parameters
    params.printParams()
    
    # demand
    jobs = [f'Job {i}' for i in range(len(demand))]
    demandDf = pd.DataFrame(list(demand.values()), columns=['Demand'], index=jobs)
    print("[DEMAND] The current demand is: \n", demandDf, "\n")

def optimize_and_plot_mip(model, callback_interval=15):
    # Record the start time
    start_time = time.time()

    # Data storage for the callback
    times = []
    objectives = []
    gaps = []

    # Callback function to capture the objective value at multiples of callback_interval seconds
    def callback(model, where):
        nonlocal times, objectives
        if where == gp.GRB.Callback.MIPSOL:
            current_time = time.time()
            elapsed_time = current_time - start_time


            # Check if the elapsed time is a multiple of callback_interval seconds
            obj_bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
            obj_best = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)
            gap = abs(obj_bound - obj_best) / (abs(obj_best) + 1e-6)  # Calculate MIP gap
            if obj_best < 1e+10:
                times.append(current_time)
                objectives.append(obj_best)
                gaps.append(gap)


    # Set the callback function using a lambda to pass start_time, times, and objectives
    model.optimize(lambda model, where: callback(model, where))

    # Adjust times to be relative to the start time
    elapsed_times = [t - start_time for t in times]

    # Plot the objective value over time
    fig, ax1 = plt.subplots()

    # Plot the objective value with the primary y-axis
    ax1.plot(elapsed_times, objectives, marker='o', color='tab:blue', label='Objective Value')
    ax1.set_xlabel('Tiempo (minutos)', fontsize=14)
    ax1.set_ylabel('Makespan', color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Set major and minor ticks on the x-axis
    ax1.xaxis.set_major_locator(MultipleLocator(300))
    ax1.xaxis.set_minor_locator(MultipleLocator(60))

    # Customize tick parameters
    ax1.tick_params(axis='x', which='minor', length=6, width=1)  # Increase minor tick size
    ax1.tick_params(axis='x', which='major', length=10, width=2)  # Customize major tick size
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x/60)))

    # Create a secondary y-axis for the MIP gap
    ax2 = ax1.twinx()
    ax2.plot(elapsed_times, [gap * 100 for gap in gaps], marker='x', color='tab:red', label='MIP Gap')
    ax2.set_ylabel('MIP Gap (%)', color='tab:red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add a custom formatter to display percentages on the y-axis
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y / 100)))

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.title('Evolución del Makespan y MIP Gap en el tiempo', fontsize=18)
    plt.grid(True)
    plt.show()

#--------- MAIN ---------
def main():
    d =50
    demand={0:d,1:d,2:d,3:d,4:d,5:d,6:d,7:d}
    # demand={0:20,1:40,2:100,3:200,4:300,5:200}
    machines, jobs, maxlots, seed = 3, 3, 3, 5
    params = getJobShopParameters(machines, jobs, maxlots, seed) #2,2,4,3

    printData(params, demand)
    
    resultsDf, runtime, gap = gurobiModel(params,demand)

    gantt.print_pxt_gantt(resultsDf,machines,jobs,maxlots,seed,d)

if __name__=="__main__":
    main()
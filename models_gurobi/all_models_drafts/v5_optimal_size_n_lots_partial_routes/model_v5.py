from params import JobShopRandomParams
import gurobipy as gp
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from plotting import plot_pxt_gantt

def model_v5(params:JobShopRandomParams, demand:dict): #builds and solve the model
    
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
    
    # model initialization
    model = gp.Model('Jobshop')

    # continuous decision variables
    p = model.addVars(set["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'p')
    x = model.addVars(set["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'x')
    y = model.addVars(set["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'y')
    z = model.addVars(penta_mklju,vtype=gp.GRB.BINARY, name = 'z')
    C = model.addVar(vtype=gp.GRB.INTEGER, name = 'C')
    q = model.addVars(doble_ju,vtype=gp.GRB.INTEGER, name = 'q')
    Q = model.addVars(doble_ju,vtype=gp.GRB.BINARY, name = 'Q')

    V = sum(process_time[(m,j)]*q[j,u] for m,j,u in set["machines_jobs_batches"])

    variables = {'p':p, 'x':x, 'y':y, 'z':z, 'C':C, 'q':q, 'Q':Q, 'V':V}

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
                model.addConstrs((y[m,j,u]>=x[m,j,u-1]+p[m,j,u-1]) for u in lotes if u!=0) #2
    
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

    # Set model objective (minimize makespan)
    model.setObjective(C,gp.GRB.MINIMIZE)

    return model, variables

def model_v5_improvement(model, variables, params, demand):

    d = demand[0]

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

    # set upper bound for makespan
    variables['C'].ub = model.getVarByName('C').X

    # minimize number of lots
    Q = variables['Q']
    n_lots = model.addVar(vtype=gp.GRB.CONTINUOUS, name = 'E')
    model.addConstr(n_lots==gp.quicksum(Q[j,u] for j,u in doble_ju)) #15
    model.setObjective(n_lots,gp.GRB.MINIMIZE)
    model.optimize()

    # get results
    df_results = get_model_df_results(model, variables, params)

    # plot gantt chart
    plot_pxt_gantt(df_results,params,d, show=True, version=1)

    # set constants by fixing bounds of indicated variables
    constants = ['p','z','q','Q']
    for var_group_name,var_group in variables.items():
        if var_group_name in constants:
            for var in var_group.values():
                var.lb = var.X
                var.ub = var.X

    # minimize earliness
    x = variables['x']
    E = model.addVar(vtype=gp.GRB.CONTINUOUS, name = 'E')    
    model.addConstr(E==gp.quicksum(x[m,j,u] for m,j,u in triple_mju)) #15
    model.setObjective(E,gp.GRB.MINIMIZE)
    model.optimize()
    


    return model, variables

def solve_model(model:gp.Model, variables, timelimit:int,plot_evolution:True):

    
    # Set the maximum solving time in seconds
    model.setParam('TimeLimit', timelimit)

    # Solve model
    if plot_evolution:
        model, elapsed_times, objectives, gaps = optimize_recording_solutions(model, callback_interval=15)
    else:
        model.optimize()
        elapsed_times, objectives, gaps = None, None, None

    # Check the optimization status
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found!")
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Optimization terminated due to time limit.")
    else:
        print("No solution found within the time limit.")

    print("\n El makespan máximo es: ", model.getVarByName('C').X, "\n")

    # get the run_time
    run_time = model.Runtime

    # get the MIP gap
    optimality_gap = model.MIPGap

    print("Model run time: ", run_time, "Optimality gap: ", optimality_gap, "\n")

    return model, variables, elapsed_times, objectives, gaps

def get_model_df_results(model:gp.Model, variables, params:JobShopRandomParams):

    # Get the variables from the 'variables' dictionary
    p = variables['p']
    x = variables['x']
    y = variables['y']
    q = variables['q']

    #--------- DATA FRAME -----------
    results = []

    # Iterate through the variables and store the variable values in a list
    for u in params.lotes:
        for j in params.jobs:
            for m in params.machines:
                if (m in params.seq[j] and q[j,u].X>0):
                    results.append({
                        'Job': j,
                        'Lote': u,
                        'Machine': m,                                
                        'Setup Time': params.setup[m, j],
                        'Processing Time': params.p_times[(m, j)],
                        'Processing Time for u':p[m,j,u].X,
                        'Start Time (x)': x[m, j, u].X,
                        'Setup Start Time (y)': y[m, j, u].X,
                        'Lotsize' : q[j,u].X,
                        'makespan':x[m,j,u].X + p[m,j,u].X
                    })

    # Create a DataFrame from the results list
    results_df = pd.DataFrame(results)

    return results_df

def optimize_recording_solutions(model, callback_interval=15):
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

    return model, elapsed_times, objectives, gaps

if __name__=='__main__':
    pass
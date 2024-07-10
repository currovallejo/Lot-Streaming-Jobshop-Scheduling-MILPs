import gurobipy as gp
import pandas as pd
from params import JobShopRandomParams

def build_model_v9(params:JobShopRandomParams, demand:dict):
    
    # renaming parameters
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    setup_time = params.setup
    seq = params.seq
    lotes = params.lotes
    s = setup_time

    # build sets
    sets={}

    sets["jobs_batches"] = doble_ju = set(
        gp.tuplelist([(params.jobs[0], 0)] + 
                     [(j, u) 
                        for j in params.jobs[1:-1] 
                        for u in params.lotes] + 
                     [(params.jobs[-1], 0)]))

    sets["machines_jobs_batches"] = triple_mju = set(
        gp.tuplelist([(m, j, u) 
                      for m in params.machines 
                      for j in params.jobs 
                      if m in params.seq[j] 
                      for u in (params.lotes if j not in [0, params.jobs[-1]] else [0])]))

    sets["machines_jobs_jobs_batches_batches"] = penta_mklju = set(
        gp.tuplelist([(m,j,u,k,l) 
                      for m in params.machines 
                      for k in params.jobs 
                      for j in params.jobs if (m in params.seq[j] and m in params.seq[k]) 
                      for l in (params.lotes if k not in [0, params.jobs[-1]] else [0]) 
                      for u in (params.lotes if j not in [0, params.jobs[-1]] else [0]) if j!=k or u!=l]))
    
    #initialize model
    model = gp.Model('Jobshop')

    # continuous decision variables
    p = model.addVars(sets["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'p')
    x = model.addVars(sets["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'x')
    y = model.addVars(sets["machines_jobs_batches"],vtype=gp.GRB.CONTINUOUS, name = 'y')
    z = model.addVars(penta_mklju,vtype=gp.GRB.BINARY, name = 'z')
    C = model.addVar(vtype=gp.GRB.INTEGER, name = 'C')
    c = model.addVars(triple_mju,vtype=gp.GRB.CONTINUOUS, name = 'c')
    q = model.addVars(doble_ju,vtype=gp.GRB.INTEGER, name = 'q')
    Q = model.addVars(doble_ju,vtype=gp.GRB.BINARY, name = 'Q')

    V = 2*sum(process_time[(m,j)]*q[j,u] for m,j,u in sets["machines_jobs_batches"])

    variables = {'p':p, 'x':x, 'y':y, 'z':z, 'C':C, 'q':q, 'Q':Q, 'V':V}

    # CONSTRAINTS
    model.addConstrs(C>=c[m,j,u] for m,j,u in triple_mju)#1
    model.addConstrs(c[m,j,u]>=x[m,j,u]+p[m,j,u] for m,j,u in triple_mju) #2
    model.addConstrs(x[m,j,u]>=y[m,j,u] for m,k,l,j,u in penta_mklju if (m in seq[j] and m in seq[k])) #3
    model.addConstrs(x[m,jobs[-1],0]>=c[m,j,u] for m,j,u in triple_mju if j!=jobs[-1])#4
    model.addConstrs(y[m,j,u]>=c[m,0,0] for m,j,u in triple_mju if j!=0)#5
    model.addConstrs(y[m,k,l]>=x[m,k,l]-s[m,k,j]*Q[k,l] - V*(1-z[m,j,u,k,l]) for m,k,l,j,u in penta_mklju) #5'

    for j in jobs:
        for m in machines:
            if m in seq[j]:
                model.addConstrs((y[m,j,u]>=x[m,j,u-1]+p[m,j,u-1]) for u in lotes if (u!=0 and j!=0 and j!=params.jobs[-1])) #6

    model.addConstrs((x[m,k,l] >= y[m,k,l] + s[m,k,j]*Q[k,l] - V*(1-z[m,j,u,k,l])) for m,k,l,j,u in penta_mklju) #7
    model.addConstrs((y[m,k,l]>=x[m,j,u] + p[m,j,u] - V*(1-z[m,j,u,k,l])) for m,k,l,j,u in penta_mklju)#8

    # model.addConstrs(z[m,k,l,j,u]+z[m,j,u,k,l]==1 for m,k,l,j,u in penta_mklju)

    for m in machines:
        for j in jobs:
            if j!=jobs[-1] and m in seq[j]:
                for u in (lotes if j!=0 else [0]):
                    model.addConstr(gp.quicksum(z[m,j,u,k,l]*Q[j,u] for k,l in doble_ju if k!=0 if (m in seq[j] and m in seq[k]) if (k!=j or l!=u))==1)#9
    
    for m in machines:
        for k in jobs:
            if k!=0 and m in seq[k]:
                for l in (lotes if k!=jobs[-1] else [0]):
                        model.addConstr(gp.quicksum(z[m,j,u,k,l]*Q[j,u] for j,u in doble_ju if j!=jobs[-1] if (m in seq[j] and m in seq[k]) if (k!=j or l!=u))==1)#10
       
    for m,j,u in triple_mju:
        if m in seq[j]:
            if seq[j].index(m) > 0: # if m is not first machine in seq
                # Find the previous machine 'o' in the sequence
                o = seq[j][seq[j].index(m) - 1]
                # Add the constraint using 'o'
                model.addConstr(y[m, j, u] >= x[o, j, u] + p[o, j, u])#11

    model.addConstrs(gp.quicksum(q[j,u] for u in lotes)==demand[j] for j in jobs[1:-1]) #12
    model.addConstrs(p[m,j,u]==process_time[(m,j)]*q[j,u] for m,j,u in triple_mju) #12
    model.addConstrs(q[j, u] <= V*Q[j,u] for j, u in doble_ju) #13
    model.addConstrs(Q[j,u] <= q[j, u] for j, u in doble_ju) #14
    model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju) #15
    model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju) #16
    model.addConstrs(c[m, j, u] >= 0 for m, j, u in triple_mju) #16
    model.addConstr(C >= 0) #16
    
    return model, variables

def solve_model(model, variables, objective, timelimit=60,):
    
    # Set the maximum solving time to 40 seconds
    model.setParam('TimeLimit', timelimit)

    # Set model objective (minimize makespan)
    model.setObjective(objective,gp.GRB.MINIMIZE)

    # Solve model
    model.optimize()

    # Check the optimization status
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found!")
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Optimization terminated due to time limit.")
    else:
        print("No solution found within the time limit.")

    print("The best value found for the objective function is: ", objective.X, "\n")

    return model, variables

def build_solution_df(variables, params:JobShopRandomParams):

    # Get the variables from the 'variables' dictionary
    p = variables['p']
    x = variables['x']
    y = variables['y']
    q = variables['q']

    #--------- DATA FRAME -----------
    results = []

    # Iterate through the variables and store the variable values in a list
    for m in params.machines: 
        for j in params.jobs:
            if j!=0 and j!=params.jobs[-1]:
                if m in params.seq[j]:
                    for u in params.lotes:
                        
                            if (q[j,u].X>0):
                                results.append({
                                    'Lote': u,
                                    'Job': j,
                                    'Machine': m,                                
                                    'Setup Time': x[m, j, u].X - y[m, j, u].X,
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

if __name__ == '__main__':
    pass
import gurobipy as gp
import pandas as pd
from params import JobShopRandomParams
import numpy as np

def maxSlots(shift_time:int, params:JobShopRandomParams,demand:dict):
    maxSlots=0
    for m in params.machines:
        slots_m=0
        for j in params.jobs:
            if m in params.seq[j] and j!=0 and j!=params.jobs[-1]:
                x=1
                while 50+params.p_times[m,j]*x<=shift_time: # 50 xq es el setup time minimo
                    x+=1

                x=x-1
                slots_j=demand[j]//x+1
                slots_m+=slots_j
        if slots_m>maxSlots:
            maxSlots=slots_m
    
    return maxSlots

def build_model_v10(params:JobShopRandomParams, demand:dict, shift_time:int, power:dict):
    
    # renaming parameters
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    setup_time = params.setup
    seq = params.seq
    lotes = params.lotes
    s = setup_time

    # approx max number of shifts (slots)
    n_slots=maxSlots(shift_time, params, demand)

    print('numero de slots:' , n_slots)

    b=0
    n_lotes=len(lotes)-1
    while b==0 and n_lotes<n_slots:
        n_lotes+=1
        print('numero de lotes:', n_lotes)

        slots=np.arange(n_slots,dtype=int)
        lotes=np.arange(n_lotes,dtype=int)
        params.lotes=lotes

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
        
        sets["machines_slots_jobs_lots"]= quad_moju = set(
            gp.tuplelist([(m,o,j,u) 
                        for m in machines 
                        for o in slots for j in jobs for u in lotes]))

        sets["machines_slots"]=doble_mo=gp.tuplelist([(m,o) for m in machines for o in slots])
        
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
        W = model.addVars(quad_moju,vtype=gp.GRB.BINARY, name='W')
        E = model.addVar(vtype=gp.GRB.INTEGER, name='E')

        V = 2*sum(process_time[(m,j)]*q[j,u] for m,j,u in sets["machines_jobs_batches"])

        variables = {'p':p, 'x':x, 'y':y, 'z':z, 'C':C, 'q':q, 'Q':Q, 'W':W, 'V':V}

        # CONSTRAINTS
        #------------------ constraints from model v9 (sequence dependent setup times) ------------------
        for m,j,u in triple_mju:
            if m in seq[j]:
                if seq[j].index(m) > 0: # if m is not first machine in seq
                    # Find the previous machine 'o' in the sequence
                    o = seq[j][seq[j].index(m) - 1]
                    # Add the constraint using 'o'
                    model.addConstr(y[m, j, u] >= x[o, j, u] + p[o, j, u])#1 respect job route
        
        for j in jobs:
            for m in machines:
                if m in seq[j]:
                    model.addConstrs((y[m,j,u]>=x[m,j,u-1]+p[m,j,u-1]) for u in lotes if (u!=0 and j!=0 and j!=params.jobs[-1])) #2

        model.addConstrs(y[m,k,l]>=x[m,j,u] + p[m,j,u] - V*(1-z[m,j,u,k,l]) for m,k,l,j,u in penta_mklju)#3
        
        model.addConstrs(y[m,k,l]>=x[m,k,l]-s[m,k,j]*Q[k,l] - V*(1-z[m,j,u,k,l]) for m,k,l,j,u in penta_mklju) #4
        
        for m in machines:
            for j in jobs:
                if j!=jobs[-1] and m in seq[j]:
                    for u in (lotes if j!=0 else [0]):
                        model.addConstr(gp.quicksum(z[m,j,u,k,l]*Q[j,u] for k,l in doble_ju if k!=0 if (m in seq[j] and m in seq[k]) if (k!=j or l!=u))==1)#5
        
        for m in machines:
            for k in jobs:
                if k!=0 and m in seq[k]:
                    for l in (lotes if k!=jobs[-1] else [0]):
                            model.addConstr(gp.quicksum(z[m,j,u,k,l]*Q[j,u] for j,u in doble_ju if j!=jobs[-1] if (m in seq[j] and m in seq[k]) if (k!=j or l!=u))==1)#6
        
        model.addConstrs((x[m,k,l] >= y[m,k,l] + s[m,k,j]*Q[k,l] - V*(1-z[m,j,u,k,l])) for m,k,l,j,u in penta_mklju) #7

        model.addConstrs(C>=x[m,j,u]+p[m,j,u] for m,j,u in triple_mju)#8

        model.addConstrs(gp.quicksum(q[j,u] for u in lotes)==demand[j] for j in jobs[1:-1]) #9

        model.addConstrs(q[j, u] <= V*Q[j,u] for j, u in doble_ju) #10

        model.addConstrs(Q[j,u] <= q[j, u] for j, u in doble_ju) #11

        model.addConstrs(Q[j,u] <= Q[j,u-1] for j,u in doble_ju if u!=0) #12

        #------------------ constraints from model v7 (shift constraints) ------------------
        
        for m, o in doble_mo:     
            model.addConstr(gp.quicksum((x[m,j,u]+p[m,j,u]-y[m,j,u])*W[m,o,j,u] for m,j,u in triple_mju)<=shift_time) #13
        
        for j in jobs:
            for m in machines:
                if m in seq[j]:
                    model.addConstrs(gp.quicksum(W[m,o,j,u] for o in slots)==Q[j,u] for u in lotes if j!=0 and j!=params.jobs[-1]) #14

        model.addConstrs(y[m,j,u]>=o*shift_time*W[m,o,j,u] for m,o,j,u in quad_moju if m in seq[j] if  j!=0 and j!=params.jobs[-1]) #15

        model.addConstrs((x[m,j,u]+p[m,j,u])*W[m,o,j,u]<=(o+1)*shift_time for m,o,j,u in quad_moju if m in seq[j] if j!=0 and j!=params.jobs[-1]) #16

        #------------------ variable type constraints ------------------

        model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju) #17
        model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju) #18

        # 19 and #20 are included in variable declaration in the previous section

        model.addConstrs(p[m,j,u]==process_time[(m,j)]*q[j,u] for m,j,u in triple_mju) # data requirement

        #------------------ Energy consumption objective function ------------------
        model.addConstr(E>=gp.quicksum((x[m,j,u]-y[m,j,u])*power[m] for m,j,u in triple_mju if j!=0 and j!=params.jobs[-1])) #21

        # Set the maximum solving time to 40 seconds
        model.setParam('TimeLimit', 1200)

        # Set model objective (minimize makespan + energy consumption)
        # model.setObjective(0.5*E+0.5*C/10,gp.GRB.MINIMIZE)

        # Set model objective (minimize makespan)
        model.setObjective(C,gp.GRB.MINIMIZE)

        # Solve model
        model.optimize()

        # Check the optimization status
        if model.status == gp.GRB.OPTIMAL:
            energy_consumption=E.X/60 # [kWh]
            print("Optimal solution found!")
            print("\n El makespan máximo es: ", C.X, "\n")
            print("\n La energía consumida es: ", energy_consumption, "\n")
            b=1
        elif model.status == gp.GRB.TIME_LIMIT:
                print("Optimization terminated due to time limit.")
                if model.status == gp.GRB.OPTIMAL:
                    print("\n El makespan máximo es: ", C.X, "\n")
                    b=1
                elif model.status == gp.GRB.INFEASIBLE:
                    print("Model is infeasible. Number of lots increased to ", n_lotes+1)
                else:
                    b=1
        elif model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible. Number of lots increased to ", n_lotes+1)
        else:
            print("No solution found within the time limit.")

    return model, variables

def solve_model(model, variables, objective, timelimit=600,):
    
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
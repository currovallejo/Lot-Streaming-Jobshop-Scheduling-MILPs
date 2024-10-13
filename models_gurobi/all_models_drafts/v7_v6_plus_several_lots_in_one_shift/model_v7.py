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
import numpy as np
from xlsxwriter import Workbook

#--------- OTHER PYTHON FILES USED ---------
from params import JobShopRandomParams, job_params_from_json
import gantt_jobshop_px as gantt

#--------- GET JOBSHOP DATA ---------
# problem data (machines, jobs, batches, seed= ) is obtained calling Jobshop
# can be obtained by 
    # 1. by generating random data indicating the seed 
    # 2. by obtaining the data from a json file (previously generated in the same way, with the advantage that it can be modified) --> method job_params_from_json

def maxSlots(shift_time:int, params:JobShopRandomParams,demand:dict):
    maxSlots=0
    for m in params.machines:
        slots_m=0
        for j in params.jobs:
            if m in params.seq[j]:
                x=1
                while params.setup[m,j]+params.p_times[m,j]*x<=shift_time:
                    x+=1
                x=x-1
                slots_j=demand[j]//x+1
                slots_m+=slots_j
        if slots_m>maxSlots:
            maxSlots=slots_m
    
    return maxSlots
                    
def getJobShopParameters(machines:int, jobs:int, batches:int, seed:int):
    params = JobShopRandomParams(machines, jobs, batches, seed=seed) # generate jobshop instance
    # params = job_params_from_json('ist_2m_4j_2u_seed7_v67.json')

    return params

#--------- MILP MODEL ---------
def gurobiModel(params:JobShopRandomParams, demand:dict, shift_time:int,maxSolverTime:int): #builds and solve the model
    
    # renaming parameters
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    setup_time = params.setup
    seq = params.seq
    s = setup_time 
    
    #max slots number
    n_slots=maxSlots(shift_time, params, demand)
    print("max number of shifts is ", n_slots)

    # estimation of number of slots
    # n_slots=0
    # max_p=0
    # total_p=0
    # for j in jobs:
    #     if total_p>max_p:
    #         max_p=total_p
    #     total_p=0
    #     for m in machines:
    #         if m in seq[j]:
    #             total_p+=process_time[(m, j)]*demand[j]
    # n_slots=max_p//shift_time+4
    # print("max p ", max_p)
    # print("estimated number of shifts is ", n_slots)

    # while loop to get a feasible problem modifying number of lots and slots
    b=0
    n_lotes=1
    while b==0 and n_lotes<n_slots:
        n_lotes+=1
        print("while loop iteration number (lots quantity) ", n_lotes)
        
        slots=np.arange(n_slots,dtype=int)
        lotes=np.arange(n_lotes,dtype=int)
        params.lotes=lotes

        # sets definition
        set={}
        set["batches_machines"] = doble_um = gp.tuplelist([(u,m) for u in params.lotes for m in params.machines]) # usado para plotting, no para el modelo
        set["jobs_batches"] = doble_ju = gp.tuplelist([(j,u) for j in params.jobs for u in params.lotes])
        set["machines_jobs_batches"] = triple_mju = gp.tuplelist([(m,j,u) for m in params.machines for j in params.jobs if m in params.seq[j] for u in params.lotes])
        set["machines_jobs_jobs_batches_batches"] = penta_mklju = gp.tuplelist([(m,k,l,j,u) for m in params.machines for k in params.jobs for j in params.jobs if (m in params.seq[j] and m in params.seq[k]) for l in params.lotes for u in params.lotes if j!=k or u!=l])
        set["machines_slots_jobs_lots"]=quad_moju=gp.tuplelist([(m,o,j,u) for m in machines for o in slots for j in jobs for u in lotes])
        set["machines_slots"]=doble_mo=gp.tuplelist([(m,o) for m in machines for o in slots])

        #print("[SETS] are: \n", set)
        
        # model initialization
        model = gp.Model('Jobshop')

        # continuous decision variables
        p = model.addVars(triple_mju,vtype=gp.GRB.INTEGER, name = 'p')
        x = model.addVars(triple_mju,vtype=gp.GRB.INTEGER, name = 'x')
        y = model.addVars(triple_mju,vtype=gp.GRB.INTEGER, name = 'y')
        c = model.addVars(machines,vtype=gp.GRB.INTEGER, name = 'c')
        z = model.addVars(penta_mklju,vtype=gp.GRB.BINARY, name = 'z')
        C = model.addVar(vtype=gp.GRB.INTEGER, name = 'C')
        q = model.addVars(doble_ju,vtype=gp.GRB.INTEGER, name = 'q')
        Q = model.addVars(doble_ju,vtype=gp.GRB.BINARY, name = 'Q')
        W = model.addVars(quad_moju,vtype=gp.GRB.BINARY, name='W')
        g = model.addVars(triple_mju,vtype=gp.GRB.INTEGER, name= 'g')

        V = sum(process_time[(m,j)]*demand[j] for m in machines for j in jobs)

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

        for j in jobs:
            for m in machines:
                if m in seq[j]:
                    model.addConstrs((y[m,j,u]>=x[m,j,u-1]+p[m,j,u-1]) for u in lotes if u!=0) #2
                    
        model.addConstrs((y[m,j,u] + V*(1-z[m,k,l,j,u])>=x[m,k,l]+p[m,k,l]) for m,k,l,j,u in penta_mklju) #3

        model.addConstrs(z[m,k,l,j,u]+z[m,j,u,k,l]==1 for m,k,l,j,u in penta_mklju) #4
        model.addConstrs(x[m,j,u]>=y[m,j,u]+s[m,j]*Q[j,u] for m,j,u in triple_mju) #5

        model.addConstrs(C>=x[m,j,u]+p[m,j,u] for m,j,u in triple_mju) #6'
        # model.addConstrs(c[m]>=x[m,j,u]+p[m,j,u] for m,j,u in triple_mju) #6'

        model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju) #7
        model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju) #8

        model.addConstrs(p[m,j,u]==process_time[(m,j)]*q[j,u] for m,j,u in triple_mju) # parametro

        model.addConstrs(gp.quicksum(q[j,u] for u in lotes)==demand[j] for j in jobs) #9

        model.addConstrs(q[j, u] <= V*Q[j,u] for j, u in doble_ju) #10
        model.addConstrs(Q[j,u] <= q[j, u] for j, u in doble_ju) #11
        model.addConstrs((p[m,j,u]+s[m,j])*Q[j,u]<=shift_time for m, j, u in triple_mju) #12
        model.addConstrs(Q[j,u]<=Q[j,u-1] for j, u in doble_ju if u>0) #13
        for m, o in doble_mo:     
            model.addConstr(gp.quicksum((s[m,j]+p[m,j,u])*W[m,o,j,u] for j in jobs for u in lotes if m in seq[j])<=shift_time) #14

        for j in jobs:
            for m in machines:
                if m in seq[j]:
                    model.addConstrs(gp.quicksum(W[m,o,j,u] for o in slots)==Q[j,u] for u in lotes) #15

        model.addConstrs(y[m,j,u]>=o*shift_time*W[m,o,j,u] for m,o,j,u in quad_moju if m in seq[j]) #16

        model.addConstrs((x[m,j,u]+p[m,j,u])*W[m,o,j,u]<=(o+1)*shift_time for m,o,j,u in quad_moju if m in seq[j]) #17

        # model.addConstr(C>=gp.quicksum(c[m] for m in machines)) #makespan de lotes

        # Set the maximum solving time to 40 seconds
        model.setParam('TimeLimit', maxSolverTime)

        # Set model objective (minimize makespan)
        model.setObjective(C,gp.GRB.MINIMIZE)

        # Solve model
        model.optimize()

        # Check the optimization status
        if model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found!")
            print("\n El makespan máximo es: ", C.X, "\n")
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
    
    #--------- DATA FRAME -----------
    results = []

    # Iterate through the variables and store the variable values in a list
    for u in lotes:
        for j in jobs:
            for m in machines:
                if (m in seq[j] and q[j,u].X>0):
                    results.append({
                        'Lote': u,
                        'Job': j,
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

    return results_df

#--------- PRINT DATA AND RESULTS ---------
def printData(params, demand):
    
    print("--------- JOBSHOP DATA --------- \n")
    # job shop parameters
    params.printParams()
    
    # demand
    jobs = [f'Job {i}' for i in range(len(demand))]
    demandDf = pd.DataFrame(list(demand.values()), columns=['Demand'], index=jobs)
    print("[DEMAND] The current demand is: \n", demandDf, "\n")

#--------- MAIN ---------
def main():
    demand = {key: 100 for key in range(6)}
    # demand={0:300,1:100,2:100,3:200,4:300,5:200}
    params = getJobShopParameters(machines=3, jobs=3, batches=5, seed=5)
    shift_time=480

    print(maxSlots(shift_time, params, demand))

    printData(params, demand)
    
    resultsDf = gurobiModel(params,demand,shift_time,maxSolverTime=1200)

    gantt.print_pxt_gantt(resultsDf,shift_time)

if __name__=="__main__":
    main()
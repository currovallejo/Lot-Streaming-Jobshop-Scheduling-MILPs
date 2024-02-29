#--------- LIBRARIES ---------
import gurobipy as gp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#--------- OTHER PYTHON FILES ---------
from params import JobShopRandomParams


# datos del problema

params = JobShopRandomParams(2, 2, 3, seed=10)

machines = params.machines
jobs = params.jobs
demanda={0:100,1:200,2:100} #,3:200}
process_time = params.p_times
setup_time = params.setup
seq = params.seq
lotes = params.lotes


s = setup_time

# definicion de conjuntos

doble_tm = gp.tuplelist([(t,m) for t in lotes for m in machines]) # usado para plotting, no para el modelo

doble_jt = gp.tuplelist([(j,t) for j in jobs for t in lotes])
triple_mjt = gp.tuplelist([(m,j,t) for m in machines for j in jobs for t in lotes])
penta_mkljt = gp.tuplelist([(m,k,l,j,t) for m in machines for k in jobs for l in lotes for j in jobs for t in lotes if j!=k or t!=l])

# inicialización del modelo

model = gp.Model('Jobshop')

# variables de decisión continuas

p = model.addVars(triple_mjt,vtype=gp.GRB.INTEGER, name = 'p')
x = model.addVars(triple_mjt,vtype=gp.GRB.INTEGER, name = 'x')
y = model.addVars(triple_mjt,vtype=gp.GRB.INTEGER, name = 'y')
z = model.addVars(penta_mkljt,vtype=gp.GRB.BINARY, name = 'z')
C = model.addVar(vtype=gp.GRB.INTEGER, name = 'C')
q = model.addVars(doble_jt,vtype=gp.GRB.INTEGER, name = 'q')
Q = model.addVars(doble_jt,vtype=gp.GRB.BINARY, name = 'Q')

V = sum(process_time[(m,j)]*q[j,t] for m,j,t in triple_mjt)

# constraints

for t in lotes:
    for j in jobs:
        for m in machines:
            if seq[j].index(m) > 0:
                # Find the previous machine 'o' in the sequence
                o = seq[j][seq[j].index(m) - 1]
                # Add the constraint using 'o'
                model.addConstr(y[m, j, t] >= x[o, j, t] + p[o, j, t])#1

model.addConstrs((y[m,j,t]>=x[m,j,t-1]+p[m,j,t-1]) for t in lotes if t!=0) #2
model.addConstrs((y[m,j,t] + V*(1-z[m,k,l,j,t])-x[m,k,l]-p[m,k,l]>=0) for m,k,l,j,t in penta_mkljt) #3
model.addConstrs(z[m,k,l,j,t]+z[m,j,t,k,l]==1 for m,k,l,j,t in penta_mkljt) #4
model.addConstrs(x[m,j,t]>=y[m,j,t]+s[m,j]*Q[j,t] for m,j,t in triple_mjt) #5

model.addConstrs(C>=x[m,j,t]+p[m,j,t] for m,j,t in triple_mjt) #6

model.addConstrs(x[m, j, t] >= 0 for m, j, t in triple_mjt) #7
model.addConstrs(y[m, j, t] >= 0 for m, j, t in triple_mjt) #8

model.addConstrs(p[m,j,t]==process_time[(m,j)]*q[j,t] for m,j,t in triple_mjt) #10

model.addConstrs(gp.quicksum(q[j,t] for t in lotes)==demanda[j] for j in jobs) #11

model.addConstrs(q[j, t] <= V*Q[j,t] for j, t in doble_jt) #12
model.addConstrs(Q[j,t] <= q[j, t] for j, t in doble_jt) #13


# Set the maximum solving time to 40 seconds
# model.setParam('TimeLimit', 120)
model.setObjective(C,gp.GRB.MINIMIZE)
model.optimize()

# Check the optimization status
if model.status == gp.GRB.OPTIMAL:
    print("Optimal solution found!")
elif model.status == gp.GRB.TIME_LIMIT:
    print("Optimization terminated due to time limit.")
else:
    print("No solution found within the time limit.")

#--------- DATA FRAME -----------
results = []

# Iterate through the variables and store the variable values in a list
for t in lotes:
    for j in jobs:
        for m in machines:
            results.append({
                'Lote': t,
                'Job': j,
                'Machine': m,                                
                'Setup Time': s[m, j],
                'Processing Time': process_time[(m, j)],
                'Processing Time for t':p[m,j,t].X,
                'Start Time (x)': x[m, j, t].X,
                'Setup Start Time (y)': y[m, j, t].X,
                'Lotsize' : q[j,t].X,
                'makespan':x[m,j,t].X + p[m,j,t].X
            })

print("\n El makespan máximo es: ", C.X, "\n")

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Set the Pandas option to display all rows (max_rows=None)
pd.set_option('display.max_rows', None)

# Print the DataFrame
print(results_df)

#---------- PLOTTING ---------------

cmap = mpl.colormaps["Dark2"] # mapa de colores qualitative
colors = cmap.colors #extraigo la lista de colores asociados al mapa

figsize=[7, 3] 
dpi=100

fig, ax = plt.subplots(figsize=figsize, dpi=dpi)



for i,j in enumerate(jobs):
    starts_setup = []
    machines = []
    starts = []
    spans = []
    spans_setup = []
    for t,m in doble_tm:
        if q[j,t].X>0:        
            # Access 'y' variable
            y_var = y[m, j, t]
            if y_var is not None:
                starts_setup.append(y_var.X)

            # Access 'x' variable
            x_var = x[m, j, t]
            if x_var is not None:
                machine_index = m
                machines.append(machine_index)
                starts.append(x_var.X)
            
            # Access 'p' variable
            p_var = p[m, j, t]
            if p_var is not None:
                spans.append(p_var.X)
            
            # Access 's' variable
            s_var = s[m, j]
            if s_var is not None:
                spans_setup.append(s_var)
            
            if i >= len(colors):
                i = i % len(colors)

            color=colors[i]
            ax.barh(machines, width=spans_setup, left=starts_setup,color='white', edgecolor='black', hatch="/")
            ax.barh(machines, width=spans, left=starts, color=color, edgecolor='black', label=f"Job {j} lote {t}")

ax.set_yticks(params.machines)
ax.set_xlabel("Time")
ax.set_ylabel("Machine")
# x_axe = [i for i in range (0, 61, 2)]
# ax.set_xticks(x_axe)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
fig.tight_layout() # ensure
plt.show()


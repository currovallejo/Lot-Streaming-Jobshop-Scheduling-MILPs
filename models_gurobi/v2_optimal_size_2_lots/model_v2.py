#--------- LIBRARIES ---------
import gurobipy as gp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

#--------- OTHER PYTHON FILES ---------
from params import JobShopRandomParams


# datos del problema

params = JobShopRandomParams(2, 2, 2, seed=10)

machines = params.machines
jobs = params.jobs
demanda = {0:100, 1:200}
process_time = params.p_times
setup_time = params.setup
seq = params.seq
lotes = params.lotes

V = 50000
s = setup_time

# definicion de conjuntos

doble_tm = gp.tuplelist([(t,m) for t in lotes for m in machines]) # usado para plotting, no para el modelo

doble_jt = gp.tuplelist([(j,t) for j in jobs for t in lotes])
triple_mjt = gp.tuplelist([(m,j,t) for m in machines for j in jobs for t in lotes])
quad_mkjt = gp.tuplelist([(m,k,j,t) for m in machines for k in jobs for j in jobs for t in lotes if j!=k])

# inicialización del modelo

model = gp.Model('Jobshop')

# variables de decisión continuas

p = model.addVars(triple_mjt,vtype=gp.GRB.CONTINUOUS, name = 'p')
x = model.addVars(triple_mjt,vtype=gp.GRB.CONTINUOUS, name = 'x')
y = model.addVars(triple_mjt,vtype=gp.GRB.CONTINUOUS, name = 'y')
z = model.addVars(quad_mkjt,vtype=gp.GRB.BINARY, name = 'z')
d = model.addVars(lotes,vtype=gp.GRB.CONTINUOUS, name = 'd')
C = model.addVar(vtype=gp.GRB.CONTINUOUS, name = 'C')
q = model.addVars(doble_jt,vtype=gp.GRB.INTEGER, name = 'q')

# constraints

for t in lotes:
    for j in jobs:
        for m in machines:
            if seq[j].index(m) > 0:
                # Find the previous machine 'o' in the sequence
                o = seq[j][seq[j].index(m) - 1]
                # Add the constraint using 'o'
                model.addConstr(y[m, j, t] >= x[o, j, t] + p[o, j, t])#1

model.addConstrs((y[m,j,t] + V*(1-z[m,k,j,t])-x[m,k,t]-p[m,k,t]>=0) for m,k,j,t in quad_mkjt) #2
model.addConstrs(x[m,j,t]>=y[m,j,t]+s[m,j] for m,j,t in triple_mjt) #3
model.addConstrs(z[m,k,j,t]+z[m,j,k,t]==1 for m,k,j,t in quad_mkjt) #4
model.addConstrs(y[m,j,t]>=d[t-1] for m,j,t in triple_mjt if t!=0) #5
model.addConstrs(d[t]>=x[m,j,t]+p[m,j,t] for m,j,t in triple_mjt) #6
model.addConstr(C>=d[lotes[-1]]) #7

model.addConstrs(p[m, j, t] >= 0 for m, j, t in triple_mjt)
model.addConstrs(x[m, j, t] >= 0 for m, j, t in triple_mjt)
model.addConstrs(y[m, j, t] >= 0 for m, j, t in triple_mjt)

model.addConstrs(p[m,j,t]==process_time[(m,j)]*q[j,t] for m,j,t in triple_mjt)

model.addConstrs(gp.quicksum(q[j,t] for t in lotes)==demanda[j] for j in jobs)

# model.addConstrs(gp.quicksum(p[m,j,t] for t in lotes)<=1200 for m in machines for j in jobs)

model.setObjective(C,gp.GRB.MINIMIZE)
model.optimize()

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
        ax.barh(machines, width=spans, left=starts, color=color, edgecolor='black', label=f"Job {j}")

ax.set_yticks(params.machines)
ax.set_xlabel("Time")
ax.set_ylabel("Machine")
# x_axe = [i for i in range (0, 61, 2)]
# ax.set_xticks(x_axe)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
fig.tight_layout() # ensure
plt.show()


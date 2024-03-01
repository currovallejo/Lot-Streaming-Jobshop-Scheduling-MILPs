'''
DESARROLLO DEL MODELO Y REPRESENTACIÓN DE RESULTADOS
----------------------------------------------------------------------Es el Script principal. Se obtienen los datos, se define el modelo, se resuelve y se imprimen los resultados en un Dataframe y en un Gantt
'''

#--------- LIBRARIES ---------
import gurobipy as gp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from xlsxwriter import Workbook

#--------- OTHER PYTHON FILES USED---------
from params import JobShopRandomParams, job_params_from_json
import gantt_jobshop_px as gantt

# datos del problema (máquinas, trabajos, lotes, seed= )
# se pueden obtener generando datos random indicando la semilla o bien obteniendo los datos de un archivo json (previamente generado de igual manera, con la ventaja de que se puede modificar)

params = JobShopRandomParams(3, 3, 2, seed=6) # m, j, u

# params = job_params_from_json('ist_2m_4j_2u_seed7_v67.json')

machines = params.machines
jobs = params.jobs
demanda={0:300,1:400,2:100,3:200,4:300,5:200}
process_time = params.p_times
setup_time = params.setup
seq = params.seq
lotes = params.lotes
s = setup_time


# definicion de conjuntos

doble_um = gp.tuplelist([(u,m) for u in lotes for m in machines]) # usado para plotting, no para el modelo

doble_ju = gp.tuplelist([(j,u) for j in jobs for u in lotes])
triple_mju = gp.tuplelist([(m,j,u) for m in machines for j in jobs if m in seq[j] for u in lotes])
penta_mklju = gp.tuplelist([(m,k,l,j,u) for m in machines for k in jobs for j in jobs if (m in seq[j] and m in seq[k]) for l in lotes for u in lotes if j!=k or u!=l])

# inicialización del modelo

model = gp.Model('Jobshop')

# variables de decisión continuas

p = model.addVars(triple_mju,vtype=gp.GRB.CONTINUOUS, name = 'p')
x = model.addVars(triple_mju,vtype=gp.GRB.CONTINUOUS, name = 'x')
y = model.addVars(triple_mju,vtype=gp.GRB.CONTINUOUS, name = 'y')
z = model.addVars(penta_mklju,vtype=gp.GRB.BINARY, name = 'z')
C = model.addVar(vtype=gp.GRB.INTEGER, name = 'C')
q = model.addVars(doble_ju,vtype=gp.GRB.INTEGER, name = 'q')
Q = model.addVars(doble_ju,vtype=gp.GRB.BINARY, name = 'Q')

V = sum(process_time[(m,j)]*q[j,u] for m,j,u in triple_mju)

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
            
model.addConstrs((y[m,j,u] + V*(1-z[m,k,l,j,u])-x[m,k,l]-p[m,k,l]>=0) for m,k,l,j,u in penta_mklju) #3
model.addConstrs(z[m,k,l,j,u]+z[m,j,u,k,l]==1 for m,k,l,j,u in penta_mklju) #4
model.addConstrs(x[m,j,u]>=y[m,j,u]+s[m,j]*Q[j,u] for m,j,u in triple_mju) #5

model.addConstrs(C>=x[m,j,u]+p[m,j,u] for m,j,u in triple_mju) #6

model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju) #7
model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju) #8

model.addConstrs(p[m,j,u]==process_time[(m,j)]*q[j,u] for m,j,u in triple_mju) #10

model.addConstrs(gp.quicksum(q[j,u] for u in lotes)==demanda[j] for j in jobs) #11

model.addConstrs(q[j, u] <= V*Q[j,u] for j, u in doble_ju) #12
model.addConstrs(Q[j,u] <= q[j, u] for j, u in doble_ju) #13


# Set the maximum solving time to 40 seconds
model.setParam('TimeLimit', 600)
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

print("\n El makespan máximo es: ", C.X, "\n")

# Create a DataFrame from the results list
results_df = pd.DataFrame(results)

# Set the Pandas option to display all rows (max_rows=None)
pd.set_option('display.max_rows', None)

# Print the DataFrame
# print(results_df)

# Export the DataFrame to an Excel file
# with pd.ExcelWriter(r'C:\Users\Usuario\Documents\MII+MOIGE\TFM - LOCAL/RESULTADOS.xlsx', engine='xlsxwriter') as writer:
#     results_df.to_excel(writer, sheet_name='Sheet1', index=False)

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
    for u,m in doble_um:
        if (m in seq[j]):
            if q[j,u].X>0:        
                # Access 'y' variable
                y_var = y[m, j, u]
                if y_var is not None:
                    starts_setup.append(y_var.X)

                # Access 'x' variable
                x_var = x[m, j, u]
                if x_var is not None:
                    machine_index = m
                    machines.append(machine_index)
                    starts.append(x_var.X)
                
                # Access 'p' variable
                p_var = p[m, j, u]
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
                ax.barh(machines, width=spans, left=starts, color=color, edgecolor='black', label=f"Job {j} lote {u}")

ax.set_yticks(params.machines)
ax.set_xlabel("Time")
ax.set_ylabel("Machine")
# x_axe = [i for i in range (0, 61, 2)]
# ax.set_xticks(x_axe)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
fig.tight_layout() # ensure

# plt.show()

gantt.print_pxt_gantt(results_df)


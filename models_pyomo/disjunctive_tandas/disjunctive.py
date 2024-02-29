import pyomo.environ as pyo
from base import JobShopModel


def cstr_1(model, m, j, t):
    o = model.seq[j].prev(m) # maquina en la que se hizo la anterior operación del trabajo j
    if o is not None:
        return model.y[m, j, t] >= model.x[o,j, t] + model.p[o, j, t]
    else:
        return pyo.Constraint.Skip

def cstr_2(model, m, j, k, t):
    if j != k:
        return model.y[m,j,t] + model.V*(1-model.z[m,k,j,t]) >= model.x[m,k,t] + model.p[m,k,t]
    else:
        return pyo.Constraint.Skip
        
def cstr_3(model, m, j, t):
    return model.x[m, j, t] >= model.y[m, j, t] + model.s[m, j]

def cstr_4(model, m, j, k, t):
    if j != k:
        return model.z[m, j, k, t] + model.z[m, k, j, t] == 1
    else:
        return model.z[m, j, k, t] == 0
        
def cstr_5(model, m, j, t):
    if t!=0:        
        return model.y[m,j,t] >= model.d[t-1]
    else:
        return pyo.Constraint.Skip

def cstr_6(model, j, t):
    m = model.seq[j][-1]
    return model.d[t]>=model.x[m, j, t] + model.p[m, j, t]

def cstr_7(model, t):  
    return model.C>=model.d[t]

# def cstr_tandas(model, m, j, t):
#     o = model.seq[j].prev(m) # maquina en la que se hizo la anterior operación del trabajo j
#     if o is not None and t !=0:
#         s = model.tandas[t-1]
#         return model.y[m, j, t]>=model.x[o, j, s] + model.p[o, j, s]
#     else:
#         return pyo.Constraint.Skip

# def cstr_10(model, m, j, t):
#     if t != 0:
#         o = model.seq[j][-1]
#         return model.y[m, j, t]>=model.x[o, j, t-1] + model.p[m, j, t-1]
#     else:
#         return pyo.Constraint.Skip
    
# def cstr_11(model, m, j, t):
#     if t!=0:
#         return model.x[m, j, t]>=model.x[m, j, t-1]
#     else:
#         return pyo.Constraint.Skip

# def cstr_12(model, m, j, t):
#     return model.x[m, j, t]>=model.y[m, j, t] + model.s[m, j]

# def cstr_13(model, m, j, t):
#     for i in model.jobs:
#         if t!=0:
#             m=1
#             return model.y[m, j, t]>=model.x[m , i, t-1] + model.p[m, i, t-1]
#         else:
#             return pyo.Constraint.Skip

# def cstr_14(model):
#     return model.y[0, 0, 1]>=model.x[0 , 1, 0] + model.p[0, 1, 0]
 

class DisjModel(JobShopModel):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._create_vars()
        self._create_cstr()
        self.obj = pyo.Objective(rule=self.C, sense=pyo.minimize)

    def _create_vars(self):
        self.x = pyo.Var(self.M, self.J, self.T, within=pyo.NonNegativeReals)
        self.z = pyo.Var(self.M, self.J, self.J, self.T, within=pyo.Binary)
        self.d = pyo.Var(self.T, within=pyo.NonNegativeReals) 
        self.C = pyo.Var(within=pyo.NonNegativeReals)
        self.y = pyo.Var(self.M, self.J, self.T, within=pyo.NonNegativeReals, initialize=0)

    def _create_cstr(self):
        self.cstr_1 = pyo.Constraint(self.M, self.J, self.T, rule=cstr_1)
        self.cstr_2 = pyo.Constraint(self.M, self.J, self.J, self.T, rule=cstr_2)
        self.cstr_3 = pyo.Constraint(self.M, self.J, self.T, rule=cstr_3)
        self.cstr_4=pyo.Constraint(self.M, self.J, self.J, self.T,  rule=cstr_4)
        self.cstr_5 = pyo.Constraint(self.M, self.J, self.T, rule=cstr_5)
        self.cstr_6 = pyo.Constraint(self.J, self.T, rule=cstr_6)
        self.cstr_7 = pyo.Constraint(self.T, rule=cstr_7)

    def _get_elements(self, j, t):
        
        starts_setup = [y.value for y in self.y[:,j,t]]
        machines = [x.index()[0] for x in self.x[:, j, t]]
        starts = [x.value for x in self.x[:, j, t]]
        spans = [self.p[m, j, t] for m in machines]
        spans_setup = [self.s[m, j] for m in machines]
        return machines, starts, spans, starts_setup, spans_setup

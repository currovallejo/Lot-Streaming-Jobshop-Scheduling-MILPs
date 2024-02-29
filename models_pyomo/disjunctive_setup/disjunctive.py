import pyomo.environ as pyo
from base import JobShopModel


def cstr_seq(model, m, j):
    o = model.seq[j].prev(m) # maquina en la que se hizo la anterior operación del trabajo j
    if o is not None:
        return model.y[m, j] >= model.x[o,j] + model.p[o, j]
    else:
        return pyo.Constraint.Skip

def cstr_precede(model, m, j, k):
    if j != k:
        return model.y[m,j] + model.V*(1-model.z[m,k,j]) >= model.x[m,k] + model.p[m,k]
    else:
        return pyo.Constraint.Skip

def cstr_setup(model, m, j):
    return model.x[m, j] >= model.y[m, j] + model.s[m, j]

def cstr_comp_precede(model, m, j, k):
    if j != k:
        return model.z[m, j, k] + model.z[m, k, j] == 1
    else:
        return model.z[m, j, k] == 0

def cstr_total_time(model, j):
    m = model.seq[j][-1]
    return model.x[m, j] + model.p[m, j] <= model.C

class DisjModel(JobShopModel):

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self._create_vars()
        self._create_cstr()
        self.obj = pyo.Objective(rule=self.C, sense=pyo.minimize)

    def _create_vars(self):
        self.x = pyo.Var(self.M, self.J, within=pyo.NonNegativeReals)
        self.z = pyo.Var(self.M, self.J, self.J, within=pyo.Binary)
        self.C = pyo.Var(within=pyo.NonNegativeReals)
        self.y = pyo.Var(self.M, self.J, within=pyo.NonNegativeReals)

    def _create_cstr(self):
        self.cstr_seq = pyo.Constraint(self.M, self.J, rule=cstr_seq)
        self.cstr_precede = pyo.Constraint(self.M, self.J, self.J, rule=cstr_precede)
        self.cstr_comp_precede = pyo.Constraint(self.M, self.J, self.J, rule=cstr_comp_precede)
        self.cstr_total_time = pyo.Constraint(self.J, rule=cstr_total_time)
        self.cstr_setup=pyo.Constraint(self.M, self.J, rule=cstr_setup)

    def _get_elements(self, j):
        machines = [x.index()[0] for x in self.x[:, j]]
        starts_setup = [y.value for y in self.y[:,j]]
        spans_setup = [self.s[m, j] for m in machines]
        starts = [x.value for x in self.x[:, j]]
        spans = [self.p[m, j] for m in machines]
        return machines, starts, spans, starts_setup, spans_setup

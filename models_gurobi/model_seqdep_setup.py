"""
Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH MILP (Gurobi)

Module: model with sequence dependent setup times

Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro
"""

from collections import namedtuple
import gurobipy as gp
import pandas as pd
from itertools import product

from params import JobShopRandomParamsSeqDep
import plot
import model_basic


def build(params: JobShopRandomParamsSeqDep):
    """Build the model with sequence dependent setup times

    Args:
        params (JobShopRandomParams): jobshop parameters

    Returns:
        model: Gurobi model
        variables: Gurobi variables
    """

    # renaming parameters
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    seq = params.seq
    lots = params.lots
    demand = params.demand
    s = params.setup

    # sets definition
    sets = {}

    sets["jobs_batches"] = doble_ju = set(
        gp.tuplelist(
            [(params.jobs[0], 0)]
            + [(j, u) for j in params.jobs[1:-1] for u in params.lots]
            + [(params.jobs[-1], 0)]
        )
    )

    sets["machines_jobs_batches"] = triple_mju = set(
        gp.tuplelist(
            [
                (m, j, u)
                for m in params.machines
                for j in params.jobs
                if m in params.seq[j]
                for u in (params.lots if j not in [0, params.jobs[-1]] else [0])
            ]
        )
    )

    sets["machines_jobs_jobs_batches_batches"] = penta_mklju = set(
        gp.tuplelist(
            [
                (m, j, u, k, l)
                for m in params.machines
                for k in params.jobs
                for j in params.jobs
                if (m in params.seq[j] and m in params.seq[k])
                for l in (params.lots if k not in [0, params.jobs[-1]] else [0])
                for u in (params.lots if j not in [0, params.jobs[-1]] else [0])
                if j != k or u != l
            ]
        )
    )

    # model initialization
    model = gp.Model("Jobshop")

    # continuous decision variables
    p = model.addVars(sets["machines_jobs_batches"], vtype=gp.GRB.CONTINUOUS, name="p")
    x = model.addVars(sets["machines_jobs_batches"], vtype=gp.GRB.CONTINUOUS, name="x")
    y = model.addVars(sets["machines_jobs_batches"], vtype=gp.GRB.CONTINUOUS, name="y")
    z = model.addVars(penta_mklju, vtype=gp.GRB.BINARY, name="z")
    C = model.addVar(vtype=gp.GRB.INTEGER, name="C")
    c = model.addVars(triple_mju, vtype=gp.GRB.CONTINUOUS, name="c")
    q = model.addVars(doble_ju, vtype=gp.GRB.INTEGER, name="q")
    Q = model.addVars(doble_ju, vtype=gp.GRB.BINARY, name="Q")

    V = 2 * sum(
        process_time[(m, j)] * q[j, u] for m, j, u in sets["machines_jobs_batches"]
    )

    GurobiVars = namedtuple("GurobiVars", ["p", "x", "y", "z", "C", "q", "Q", "V"])
    variables = GurobiVars(p, x, y, z, C, q, Q, V)

    # constraints
    # 1
    model.addConstrs(C >= c[m, j, u] for m, j, u in triple_mju)

    # 2
    model.addConstrs(c[m, j, u] >= x[m, j, u] + p[m, j, u] for m, j, u in triple_mju)

    # 3
    model.addConstrs(
        x[m, j, u] >= y[m, j, u]
        for m, k, l, j, u in penta_mklju
        if (m in seq[j] and m in seq[k])
    )

    # 4
    model.addConstrs(
        x[m, jobs[-1], 0] >= c[m, j, u] for m, j, u in triple_mju if j != jobs[-1]
    )

    # 5
    model.addConstrs(y[m, j, u] >= c[m, 0, 0] for m, j, u in triple_mju if j != 0)

    # 6
    model.addConstrs(
        y[m, k, l] >= x[m, k, l] - s[m, k, j] * Q[k, l] - V * (1 - z[m, j, u, k, l])
        for m, k, l, j, u in penta_mklju
    )

    # 7
    for j in jobs:
        for m in machines:
            if m in seq[j]:
                model.addConstrs(
                    (y[m, j, u] >= x[m, j, u - 1] + p[m, j, u - 1])
                    for u in lots
                    if (u != 0 and j != 0 and j != params.jobs[-1])
                )

    # 8
    model.addConstrs(
        (x[m, k, l] >= y[m, k, l] + s[m, k, j] * Q[k, l] - V * (1 - z[m, j, u, k, l]))
        for m, k, l, j, u in penta_mklju
    )

    # 9
    model.addConstrs(
        (y[m, k, l] >= x[m, j, u] + p[m, j, u] - V * (1 - z[m, j, u, k, l]))
        for m, k, l, j, u in penta_mklju
    )

    # 10
    for m in machines:
        for j in jobs:
            if j != jobs[-1] and m in seq[j]:
                for u in lots if j != 0 else [0]:
                    model.addConstr(
                        gp.quicksum(
                            z[m, j, u, k, l] * Q[j, u]
                            for k, l in doble_ju
                            if k != 0
                            if (m in seq[j] and m in seq[k])
                            if (k != j or l != u)
                        )
                        == 1
                    )

    # 11
    for m in machines:
        for k in jobs:
            if k != 0 and m in seq[k]:
                for l in lots if k != jobs[-1] else [0]:
                    model.addConstr(
                        gp.quicksum(
                            z[m, j, u, k, l] * Q[j, u]
                            for j, u in doble_ju
                            if j != jobs[-1]
                            if (m in seq[j] and m in seq[k])
                            if (k != j or l != u)
                        )
                        == 1
                    )

    # 12
    for m, j, u in triple_mju:
        if m in seq[j]:
            if seq[j].index(m) > 0:  # if m is not first machine in seq
                # Find the previous machine 'o' in the sequence
                o = seq[j][seq[j].index(m) - 1]
                # Add the constraint using 'o'
                model.addConstr(y[m, j, u] >= x[o, j, u] + p[o, j, u])

    # 13
    model.addConstrs(
        gp.quicksum(q[j, u] for u in lots) == demand[j] for j in jobs[1:-1]
    )

    # 14
    model.addConstrs(
        p[m, j, u] == process_time[(m, j)] * q[j, u] for m, j, u in triple_mju
    )

    # 15
    model.addConstrs(q[j, u] <= V * Q[j, u] for j, u in doble_ju)

    # 16
    model.addConstrs(Q[j, u] <= q[j, u] for j, u in doble_ju)

    # 17
    model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju)

    # 18
    model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju)

    # 19
    model.addConstrs(c[m, j, u] >= 0 for m, j, u in triple_mju)

    # 20
    model.addConstr(C >= 0)

    return model, variables


def get_df_results(model: gp.Model, variables, params: JobShopRandomParamsSeqDep):
    """Extracts the results from the model and stores them in a pandas DataFrame

    Args:
        model (gp.Model): Gurobi model with the results
        variables (namedtuple): named tuple with the variables of the model
        params (JobShopRandomParams): object with the parameters of the problem

    Returns:
        results_df (pd.DataFrame): DataFrame with the results of the model
    """

    # Get the variables from the 'variables' dictionary
    p = variables.p
    x = variables.x
    y = variables.y
    q = variables.q

    # --------- DATA FRAME -----------
    # Iterate through the variables and store the variable values in a list
    results_df = pd.DataFrame(
        [
            {
                "Job": j,
                "Lot": u,
                "Machine": m,
                "Setup Time": x[m, j, u].X - y[m, j, u].X,
                "Processing Time": params.p_times[(m, j)],
                "Processing Time for u": p[m, j, u].X,
                "Start Time (x)": x[m, j, u].X,
                "Setup Start Time (y)": y[m, j, u].X,
                "Lotsize": q[j, u].X,
                "Makespan": x[m, j, u].X + p[m, j, u].X,
            }
            for u, j, m in product(params.lots, params.jobs, params.machines)
            if j not in [0, params.jobs[-1]]
            if m in params.seq[j] and q[j, u].X > 0
        ]
    )

    return results_df


# --------- MAIN ---------
def main():
    params = JobShopRandomParamsSeqDep(n_machines=3, n_jobs=3, n_lots=3, seed=4)
    model, variables = build(params)
    model, variables, *_ = model_basic.solve(
        model, variables, timeLimit=300, plotSolutionEvolution=False
    )
    df_results = get_df_results(model, variables, params)
    plot.gantt(df_results, params, show=True, version='seqdep_setup')


if __name__ == "__main__":
    main()

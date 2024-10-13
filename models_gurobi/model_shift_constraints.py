"""
Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH MILP (Gurobi)

Module: model with shift constraints

Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro
"""

import gurobipy as gp
import numpy as np
from itertools import product
from collections import namedtuple
from params import JobShopRandomParams

import model_basic
import plot


def build_and_solve(params: JobShopRandomParams, shift_time: int, timeLimit=300):
    """Function to build the model with shift constraints

    Args:
        params (JobShopRandomParams): jobshop parameters
        shift_time (int): maximum time for a shift

    Returns:
    """
    def build():
        '''Build the model with shift constraints

        Returns:
            model: Gurobi model
            variables: Gurobi variables
        '''
        # sets definition
        set = {}
        set["jobs_batches"] = doble_ju = gp.tuplelist(
            [(j, u) for j in params.jobs for u in params.lots]
        )
        set["machines_jobs_batches"] = triple_mju = gp.tuplelist(
            [
                (m, j, u)
                for m in params.machines
                for j in params.jobs
                if m in params.seq[j]
                for u in params.lots
            ]
        )
        set["machines_jobs_jobs_batches_batches"] = penta_mklju = gp.tuplelist(
            [
                (m, k, l, j, u)
                for m in params.machines
                for k in params.jobs
                for j in params.jobs
                if (m in params.seq[j] and m in params.seq[k])
                for l in params.lots
                for u in params.lots
                if j != k or u != l
            ]
        )
        set["machines_slots_jobs_lots"] = quad_moju = gp.tuplelist(
            [(m, o, j, u) for m in machines for o in slots for j in jobs for u in lots]
        )
        set["machines_slots"] = doble_mo = gp.tuplelist(
            [(m, o) for m in machines for o in slots]
        )

        # model initialization
        model = gp.Model("Jobshop")

        # continuous decision variables
        p = model.addVars(triple_mju, vtype=gp.GRB.INTEGER, name="p")
        x = model.addVars(triple_mju, vtype=gp.GRB.INTEGER, name="x")
        y = model.addVars(triple_mju, vtype=gp.GRB.INTEGER, name="y")
        z = model.addVars(penta_mklju, vtype=gp.GRB.BINARY, name="z")
        C = model.addVar(vtype=gp.GRB.INTEGER, name="C")
        q = model.addVars(doble_ju, vtype=gp.GRB.INTEGER, name="q")
        Q = model.addVars(doble_ju, vtype=gp.GRB.BINARY, name="Q")
        W = model.addVars(quad_moju, vtype=gp.GRB.BINARY, name="W")

        V = sum(process_time[(m, j)] * demand[j] for m in machines for j in jobs)

        GurobiVars = namedtuple(
            "GurobiVars", ["p", "x", "y", "z", "C", "q", "Q", "W", "V"]
        )
        variables = GurobiVars(p, x, y, z, C, q, Q, W, V)

        # constraints
        # 1
        for u, j, m in product(lots, jobs, machines):
            if m in seq[j]:
                if seq[j].index(m) > 0:
                    # Find the previous machine 'o' in the sequence
                    o = seq[j][seq[j].index(m) - 1]
                    # Add the constraint using 'o'
                    model.addConstr(y[m, j, u] >= x[o, j, u] + p[o, j, u])

        # 2
        for j, m in product(jobs, machines):
            if m in seq[j]:
                model.addConstrs(
                    (y[m, j, u] >= x[m, j, u - 1] + p[m, j, u - 1])
                    for u in lots
                    if u != 0
                )

        # 3
        model.addConstrs(
            (y[m, j, u] + V * (1 - z[m, k, l, j, u]) >= x[m, k, l] + p[m, k, l])
            for m, k, l, j, u in penta_mklju
        )

        # 4
        model.addConstrs(
            z[m, k, l, j, u] + z[m, j, u, k, l] == 1 for m, k, l, j, u in penta_mklju
        )

        # 5
        model.addConstrs(
            x[m, j, u] >= y[m, j, u] + s[m, j] * Q[j, u] for m, j, u in triple_mju
        )

        # 6
        model.addConstrs(C >= x[m, j, u] + p[m, j, u] for m, j, u in triple_mju)  # 6'

        # 7
        model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju)

        # 8
        model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju)

        # 9
        model.addConstrs(
            p[m, j, u] == process_time[(m, j)] * q[j, u] for m, j, u in triple_mju
        )

        # 10
        model.addConstrs(gp.quicksum(q[j, u] for u in lots) == demand[j] for j in jobs)

        # 11
        model.addConstrs(q[j, u] <= V * Q[j, u] for j, u in doble_ju)

        # 12
        model.addConstrs(Q[j, u] <= q[j, u] for j, u in doble_ju)

        # 13
        model.addConstrs(
            (p[m, j, u] + s[m, j]) * Q[j, u] <= shift_time for m, j, u in triple_mju
        )

        # 14
        model.addConstrs(Q[j, u] <= Q[j, u - 1] for j, u in doble_ju if u > 0)
        for m, o in doble_mo:
            model.addConstr(
                gp.quicksum(
                    (s[m, j] + p[m, j, u]) * W[m, o, j, u]
                    for j in jobs
                    for u in lots
                    if m in seq[j]
                )
                <= shift_time
            )

        # 15
        for j, m in product(jobs, machines):
            if m in seq[j]:
                model.addConstrs(
                    gp.quicksum(W[m, o, j, u] for o in slots) == Q[j, u] for u in lots
                )

        # 16
        model.addConstrs(
            y[m, j, u] >= o * shift_time * W[m, o, j, u]
            for m, o, j, u in quad_moju
            if m in seq[j]
        )

        # 17
        model.addConstrs(
            (x[m, j, u] + p[m, j, u]) * W[m, o, j, u] <= (o + 1) * shift_time
            for m, o, j, u in quad_moju
            if m in seq[j]
        )

        # Set model objective (minimize makespan)
        model.setObjective(C, gp.GRB.MINIMIZE)

        return model, variables

    def solve(model, variables, b, timeLimit=timeLimit):
        '''Function to solve the model

        Args:
            model: Gurobi model
            variables: Gurobi variables
            b: flag to check if the model is feasible
            timeLimit: maximum solving time

        Returns:
            model: Gurobi model
            variables: Gurobi variables
            b: flag to check if the model is feasible
        '''
        # Set the maximum solving time to 40 seconds
        model.setParam("TimeLimit", timeLimit)

        # Solve model
        model.optimize()

        # Check the optimization status
        if model.status == gp.GRB.OPTIMAL:
            print("Optimal solution found!")
            print("\n El makespan máximo es: ", variables.C.X, "\n")
            b = 1
        elif model.status == gp.GRB.TIME_LIMIT:
            print("Optimization terminated due to time limit.")
            if model.status == gp.GRB.OPTIMAL:
                print("\n El makespan máximo es: ", variables.C.X, "\n")
                b = 1
            elif model.status == gp.GRB.INFEASIBLE:
                print("Model is infeasible. Number of lots increased to ", n_lots + 1)
            else:
                b = 1
        elif model.status == gp.GRB.INFEASIBLE:
            print("Model is infeasible. Number of lots increased to ", n_lots + 1)
        else:
            print("No solution found within the time limit.")

        return model, variables, b

    def maxSlots(shift_time: int, params: JobShopRandomParams):
        """Function to calculate (approx) the maximum number of shifts needed to solve
        the problem

        Args:
            shift_time (int): maximum time for a shift
            params (JobShopRandomParams): jobshop parameters

        Returns:
            maxSlots: maximum number of shifts
        """
        demand = params.demand
        maxSlots = 0
        for m in params.machines:
            slots_m = 0
            for j in params.jobs:
                if m in params.seq[j]:
                    x = 1
                    while params.setup[m, j] + params.p_times[m, j] * x <= shift_time:
                        x += 1
                    x = x - 1
                    slots_j = demand[j] // x + 1
                    slots_m += slots_j
            if slots_m > maxSlots:
                maxSlots = slots_m

        return maxSlots
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    s = params.setup
    seq = params.seq
    demand = params.demand

    # max slots number
    n_slots = maxSlots(shift_time, params)
    print("max number of shifts is ", n_slots)

    # while loop to get a feasible problem modifying number of lots and slots
    b = 0
    n_lots = 1
    while b == 0 and n_lots < n_slots:
        n_lots += 1
        print("max lots set to: ", n_lots)

        slots = np.arange(n_slots, dtype=int)
        lots = np.arange(n_lots, dtype=int)
        params.lots = lots

        model, variables = build()
        model, variables, b = solve(model, variables, b)

    return model, variables


# --------- MAIN ---------
def main():
    params = JobShopRandomParams(n_machines=3, n_jobs=3, n_lots=3, seed=5)
    model, variables = build_and_solve(params, shift_time=480)
    df_results = model_basic.get_df_results(model, variables, params)
    plot.gantt(df_results, params, show=True, version='shift_constraints')


if __name__ == "__main__":
    main()

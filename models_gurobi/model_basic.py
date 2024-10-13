"""
Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH MILP (Gurobi)

Module: basic MILP model for the lot streaming job shop scheduling
    problem
    - Sequence independent setup times
    - No shifts constraints

Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro
"""

import gurobipy as gp
import pandas as pd
import time
from itertools import product
from collections import namedtuple

from params import JobShopRandomParams
import plot


def build(params: JobShopRandomParams):
    """Builds the basic model for the lot streaming job shop scheduling problem
    using Gurobi

    Args:
        params (JobShopRandomParams): object with the parameters of the problem

    Returns:
        model (gp.Model): Gurobi model
        variables (namedtuple): named tuple with the variables of the model
    """

    # renaming parameters
    machines = params.machines
    jobs = params.jobs
    process_time = params.p_times
    s = params.setup
    seq = params.seq
    lots = params.lots
    demand = params.demand

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

    # model initialization
    model = gp.Model("Jobshop")

    # continuous decision variables
    p = model.addVars(set["machines_jobs_batches"], vtype=gp.GRB.CONTINUOUS, name="p")
    x = model.addVars(set["machines_jobs_batches"], vtype=gp.GRB.CONTINUOUS, name="x")
    y = model.addVars(set["machines_jobs_batches"], vtype=gp.GRB.CONTINUOUS, name="y")
    z = model.addVars(penta_mklju, vtype=gp.GRB.BINARY, name="z")
    C = model.addVar(vtype=gp.GRB.INTEGER, name="C")
    q = model.addVars(doble_ju, vtype=gp.GRB.INTEGER, name="q")
    Q = model.addVars(doble_ju, vtype=gp.GRB.BINARY, name="Q")

    V = sum(process_time[(m, j)] * q[j, u] for m, j, u in set["machines_jobs_batches"])

    GurobiVars = namedtuple("GurobiVars", ["p", "x", "y", "z", "C", "q", "Q", "V"])
    variables = GurobiVars(p, x, y, z, C, q, Q, V)

    # constraints
    # 1
    for m, j, u in product(machines, jobs, lots):
        if m in seq[j] and seq[j].index(m) > 0:
            # Find the previous machine 'o' in the sequence
            o = seq[j][seq[j].index(m) - 1]
            # Add the constraint using 'o'
            model.addConstr(y[m, j, u] >= x[o, j, u] + p[o, j, u])

    # 2
    model.addConstrs(y[m, j, u] >= x[m, j, u] - s[m, j] for m, j, u in triple_mju)

    # 3
    for j, m in product(jobs, machines):
        if m in seq[j]:
            model.addConstrs(
                (y[m, j, u] >= x[m, j, u - 1] + p[m, j, u - 1]) for u in lots if u != 0
            )

    # 4
    model.addConstrs(
        (y[m, j, u] + V * (1 - z[m, k, l, j, u]) - x[m, k, l] - p[m, k, l] >= 0)
        for m, k, l, j, u in penta_mklju
    )

    # 5
    model.addConstrs(
        z[m, k, l, j, u] + z[m, j, u, k, l] == 1 for m, k, l, j, u in penta_mklju
    )

    # 6
    model.addConstrs(
        x[m, j, u] >= y[m, j, u] + s[m, j] * Q[j, u] for m, j, u in triple_mju
    )

    # 7
    model.addConstrs(C >= x[m, j, u] + p[m, j, u] for m, j, u in triple_mju)

    # 8
    model.addConstrs(x[m, j, u] >= 0 for m, j, u in triple_mju)

    # 9
    model.addConstrs(y[m, j, u] >= 0 for m, j, u in triple_mju)

    # 10
    model.addConstrs(
        p[m, j, u] == process_time[(m, j)] * q[j, u] for m, j, u in triple_mju
    )

    # 11
    model.addConstrs(gp.quicksum(q[j, u] for u in lots) == demand[j] for j in jobs)

    # 12
    model.addConstrs(q[j, u] <= V * Q[j, u] for j, u in doble_ju)

    # 13
    model.addConstrs(Q[j, u] <= q[j, u] for j, u in doble_ju)

    # 14
    model.addConstrs(Q[j, u] <= Q[j, u - 1] for j, u in doble_ju if u > 0)

    # Set model objective (minimize makespan)
    model.setObjective(C, gp.GRB.MINIMIZE)

    return model, variables


def solve(model: gp.Model, variables, timeLimit: int, plotSolutionEvolution: True):
    """Solves the model using Gurobi's optimization engine

    Args:
        model (gp.Model): Gurobi model to be solved
        variables (namedtuple): named tuple with the variables of the model
        timeLimit (int): maximum time in seconds to solve the model
        plotSolutionEvolution (bool): whether to return values to plot the solution
            evolution or not (elapsed times, objectives, gaps)

    Returns:
        model (gp.Model):
            Gurobi model with the results.
        variables (namedtuple):
            Named tuple with the variables of the model.
        elapsed_times (list, optional):
            Elapsed times during the optimization.
        objectives (list, optional):
            Objective values during the optimization.
        gaps (list, optional):
            Gaps during the optimization.
    """

    def track_solution_evolution(model):
        """Tracks the solution evolution during the optimization process

        Args:
            model (gp.Model): Gurobi model to be solved

        Returns:
            model (gp.Model): Gurobi model with the results
            elapsed_times (list): elapsed times during the optimization
            objectives (list): objective values during the optimization
            gaps (list): gaps during the optimization
        """

        # Callback function to capture the objective value at every improvement
        def callback(model, where):
            nonlocal times, objectives
            if where == gp.GRB.Callback.MIPSOL:
                current_time = time.time()

                # Check if the elapsed time is a multiple of callback_interval seconds
                obj_bound = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND)
                obj_best = model.cbGet(gp.GRB.Callback.MIPSOL_OBJBST)
                gap = abs(obj_bound - obj_best) / (
                    abs(obj_best) + 1e-6
                )  # Calculate MIP gap
                if obj_best < 1e10:
                    times.append(current_time)
                    objectives.append(obj_best)
                    gaps.append(gap)

        # Record the start time
        start_time = time.time()

        # Data storage for the callback
        times = []
        objectives = []
        gaps = []

        # Set the callback function using a lambda function
        model.optimize(lambda model, where: callback(model, where))

        # Adjust times to be relative to the start time
        elapsed_times = [t - start_time for t in times]

        return model, elapsed_times, objectives, gaps

    # Set the maximum solving time in seconds
    model.setParam("TimeLimit", timeLimit)

    # Solve model
    if plotSolutionEvolution:
        model, elapsed_times, objectives, gaps = track_solution_evolution(model)
    else:
        model.optimize()
        elapsed_times, objectives, gaps = None, None, None

    # Check the optimization status
    if model.status == gp.GRB.OPTIMAL:
        print("Optimal solution found!")
    elif model.status == gp.GRB.TIME_LIMIT:
        print("Optimization terminated due to time limit.")
    else:
        print("No solution found within the time limit.")

    print("\n El makespan máximo es: ", model.getVarByName("C").X, "\n")

    # get the run_time
    run_time = model.Runtime

    # get the MIP gap
    optimality_gap = model.MIPGap

    print("Model run time: ", run_time, "Optimality gap: ", optimality_gap, "\n")

    return model, variables, elapsed_times, objectives, gaps


def improve_solution(model, variables, params):
    """Improves the solution by minimizing the number of lots and then minimizing
    earliness of lots

    Args:
        model (gp.Model): Gurobi model to be solved
        variables (namedtuple): named tuple with the variables of the model
        params (JobShopRandomParams): object with the parameters of the problem

    Returns:
        model (gp.Model): Gurobi model with the results
        variables (namedtuple): named tuple with the variables of the model
    """

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

    # set upper bound for makespan
    variables.C.ub = model.getVarByName("C").X

    # minimize number of lots
    Q = variables["Q"]
    n_lots = model.addVar(vtype=gp.GRB.CONTINUOUS, name="E")
    model.addConstr(n_lots == gp.quicksum(Q[j, u] for j, u in doble_ju))  # 15
    model.setObjective(n_lots, gp.GRB.MINIMIZE)
    model.optimize()

    # get results
    df_results = get_df_results(model, variables, params)

    # plot gantt chart
    plot.gantt(df_results, params, show=True, version=1)

    # set constants by fixing bounds of indicated variables
    constants = ["p", "z", "q", "Q"]
    # Iterate only over the fields that are in the constants list
    for field in constants:
        var_group = getattr(variables, field)
        for var in var_group:
            var.lb = var.X  # Set lower bound to current value
            var.ub = var.X  # Set upper bound to current value

    # minimize earliness
    x = variables.x
    E = model.addVar(vtype=gp.GRB.CONTINUOUS, name="E")
    model.addConstr(E == gp.quicksum(x[m, j, u] for m, j, u in triple_mju))  # 15
    model.setObjective(E, gp.GRB.MINIMIZE)
    model.optimize()

    return model, variables


def get_df_results(model: gp.Model, variables, params: JobShopRandomParams):
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
                "Setup Time": params.setup[m, j],
                "Processing Time": params.p_times[(m, j)],
                "Processing Time for u": p[m, j, u].X,
                "Start Time (x)": x[m, j, u].X,
                "Setup Start Time (y)": y[m, j, u].X,
                "Lotsize": q[j, u].X,
                "Makespan": x[m, j, u].X + p[m, j, u].X,
            }
            for u, j, m in product(params.lots, params.jobs, params.machines)
            if m in params.seq[j] and q[j, u].X > 0
        ]
    )

    return results_df


def main():
    params = JobShopRandomParams(3, 3, 2, 1)
    params.print_params()
    model, variables = build(params)
    model, variables, elapsed_times, objectives, gaps = solve(
        model, variables, timeLimit=10, plotSolutionEvolution=True
    )
    results_df = get_df_results(model, variables, params)
    plot.gantt(results_df, params, show=True, version=1)


if __name__ == "__main__":
    main()

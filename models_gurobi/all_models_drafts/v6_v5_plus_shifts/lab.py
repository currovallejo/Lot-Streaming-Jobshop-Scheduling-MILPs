from params import JobShopRandomParams

params = JobShopRandomParams(n_machines=3, n_jobs=3, n_lotes=2, seed=0)
print(params.p_times)
params.printParams()
"""
Project: LOT STREAMING JOB SHOP SCHEDULING PROBLEM SOLVED WITH MILP (Gurobi)

Module: generation of the job shop problem parameters

Author: Francisco Vallejo
LinkedIn: https://www.linkedin.com/in/franciscovallejogt/
Github: https://github.com/currovallejog
Website: https://franciscovallejo.pro
"""

import numpy as np
from typing import Iterable
import json
import pandas as pd
import os


def custom_serializer(obj):
    if isinstance(obj, tuple):
        return str(obj)
    return obj


class JobSequence(list):
    """Class for job sequence in job-shop problem

    Includes methods to check job position, swap jobs, and get previous and next jobs
    """

    def prev(self, x):
        if self.is_first(x):
            return None
        else:
            i = self.index(x)
            return self[i - 1]

    def next(self, x):
        if self.is_last(x):
            return None
        else:
            i = self.index(x)
            return self[i + 1]

    def is_first(self, x):
        return x == self[0]

    def is_last(self, x):
        return x == self[-1]

    def swap(self, x, y):
        i = self.index(x)
        j = self.index(y)
        self[i] = y
        self[j] = x

    def append(self, __object) -> None:
        if __object not in self:
            super().append(__object)
        else:
            pass


class JobShopParams:
    """
    Represents parameters for a job-shop scheduling problem.

    This class encapsulates the essential data for defining a job-shop
    scheduling scenario, including machines, jobs, processing times,
    operation sequences, setup times, lots, and job demands.

    Attributes:
        machines (Iterable): The set of available machines.
        jobs (Iterable): The set of jobs to be scheduled.
        p_times (dict): Processing times for each machine-job pair.
            Key: tuple(machine, job), Value: processing time.
        seq (dict): Sequence of operations (machines) for each job.
            Key: job, Value: list of machines in order of operations.
        setup (dict): Setup times for each machine-job pair.
            Key: tuple(machine, job), Value: setup time.
        lots (Iterable): The set of lots.
        demand (dict): Demand quantity for each job.
            Key: job, Value: demand quantity.

    Note:
        All time-related values (processing times, setup times) should be
        in consistent units (e.g., minutes).
    """

    def __init__(
        self,
        machines: Iterable,
        jobs: Iterable,
        p_times: dict,
        seq: dict,
        setup: dict,
        lots: Iterable,
        demand: dict,
    ):

        self.machines = machines
        self.jobs = jobs
        self.p_times = p_times
        self.seq = seq
        self.setup = setup
        self.lots = lots
        self.demand = demand


class JobShopRandomParams(JobShopParams):
    """
    Generate parameters for job-shop scheduling problems.

    This class creates randomized parameters for job-shop scenarios,
    including processing times for jobs on different machines.

    Attributes:
        n_machines (int): The number of machines in the job shop.
        n_jobs (int): The number of jobs to be scheduled.
        n_lots (int): The number of max lots for splitting each job.
        seed (int | None): Seed for random number generation.
        t_span (tuple): Range of possible processing times (min, max).
        t_span_setup (tuple): Range of possible setup times (min, max).

    Example:
        >>> myParams = JobShopRandomParams(3, 3, 3, seed=42)
    """

    def __init__(
        self,
        n_machines: int,
        n_jobs: int,
        n_lots: int,
        seed=None,
        t_span=(1, 20),
        t_span_setup=(50, 100),
    ):

        self.t_span = t_span
        self.seed = seed
        self.t_span_setup = t_span_setup

        demand = {product: 50 for product in range(0, n_jobs + 1)}
        machines = np.arange(n_machines, dtype=int)
        jobs = np.arange(n_jobs)
        p_times = self._random_times(machines, jobs, t_span)
        lots = np.arange(n_lots, dtype=int)
        seq = self._random_sequences(machines, jobs)
        setup = self._random_setup(machines, jobs, t_span_setup)
        super().__init__(machines, jobs, p_times, seq, setup, lots, demand)

    def _random_times(self, machines, jobs, t_span):
        """Generate random processing times for jobs on machines"""
        np.random.seed(self.seed)
        t = np.arange(t_span[0], t_span[1])
        return {(m, j): np.random.choice(t) for m in machines for j in jobs}

    def _random_sequences(self, machines, jobs):
        """Generate random sequences of operations for each job"""
        np.random.seed(self.seed)
        return {j: self._generate_random_sequence(machines) for j in jobs}

    def _generate_random_sequence(self, machines):
        """Generate a random PARTIAL sequence of operations for a job (may not contain
        all machines)
        """
        # Decide on the length of the sequence (integer between 1 and len(machines))
        sequence_length = np.random.randint(1, len(machines) + 1)

        # Randomly select machines for the sequence
        sequence = np.random.choice(machines, size=sequence_length, replace=False)
        sequence = sequence.astype(int)

        return JobSequence(sequence)

    def _random_setup(self, machines, jobs, t_span_setup):
        """Generate random setup times for jobs on machines"""
        np.random.seed(self.seed)
        t = np.arange(
            t_span_setup[0], t_span_setup[1]
        )  # meto en un vector todos los tiempos posibles
        return {(m, j): np.random.choice(t) for m in machines for j in jobs}

    def _random_sequence_dependent_setup(self, machines, jobs, t_span_setup):
        np.random.seed(self.seed)
        t = np.arange(t_span_setup[0], t_span_setup[1])
        setup_times = {}
        for m in machines:
            for j in jobs:
                if j == 0 or j == jobs[-1]:  # for dummy jobs
                    for k in jobs:
                        setup_times[m, j, k] = 0
                else:
                    setup_times[m, j, 0] = 50
                    for k in jobs:
                        if j == k:
                            setup_times[m, j, k] = 0
                        elif k != 0:
                            setup_times[m, j, k] = np.random.choice(t)

        return setup_times

    def _print_sequence_dependent_setup(self):
        print(
            "[SEQUENCE DEPENDENT SETUP TIMES] setup time for each job on each machine:"
        )
        for m in self.machines:
            print("Machine ", m)
            n_columns = len(self.jobs) - 2
            n_rows = len(self.jobs) - 1
            matrix = np.zeros((n_rows, n_columns), dtype=int)
            for key, value in self.sd_setup.items():
                if key[0] == m:
                    if key[1] != 0 and key[1] != self.jobs[-1]:
                        if key[2] != self.jobs[-1]:
                            matrix[key[2]][key[1] - 1] = value

            # Create a DataFrame with row and column labels
            setup_jobs = [f"Job {i+1}" for i in range(n_columns)]
            precedence_jobs = [f"Job {j}" for j in range(n_rows)]

            df = pd.DataFrame(matrix, columns=setup_jobs, index=precedence_jobs)

            # Print the DataFrame
            print(df, "\n")

    def print_params(self):
        """Print the parameters of the job-shop problem"""
        print("[MACHINES]: \n", self.machines, "\n")
        print("[JOBS]: \n", self.jobs, "\n")
        print("[BATCHES]: \n", self.lots, "\n")
        print(
            "[PROCESS TIMES] the working time associated with each job on each machine:"
        )
        # Determine the dimensions of the matrix
        max_job = max(key[1] for key in self.p_times.keys())
        max_machine = max(key[0] for key in self.p_times.keys())

        # Create an empty matrix filled with zeros
        matrix = np.zeros((max_job + 1, max_machine + 1), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.p_times.items():
            matrix[key[1]][key[0]] = value

        # Transpose the matrix to have jobs as rows and machines as columns
        transposed_matrix = matrix.T

        # Create a DataFrame with row and column labels
        jobs = [f"Job {i}" for i in range(max_job + 1)]
        machines = [f"Machine {j}" for j in range(max_machine + 1)]

        processTimes_df = pd.DataFrame(transposed_matrix, columns=jobs, index=machines)

        # Print the DataFrame
        print(processTimes_df, "\n")

        print(
            "[SETUP TIMES] the setup time associated with each job on each machine is:"
        )
        # Determine the dimensions of the matrix
        max_job = max(key[1] for key in self.setup.keys())
        max_machine = max(key[0] for key in self.setup.keys())

        # Create an empty matrix filled with zeros
        matrix = np.zeros((max_machine + 1, max_job + 1), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.setup.items():
            matrix[key[0]][key[1]] = value

        # Create a DataFrame with row and column labels
        jobs = [f"Job {i}" for i in range(max_job + 1)]
        machines = [f"Machine {j}" for j in range(max_machine + 1)]

        setupTimes_df = pd.DataFrame(matrix, columns=jobs, index=machines)

        # Print the DataFrame
        print(setupTimes_df, "\n")

        print("[SEQ] the sequence for each job is: ")

        trabajo_list = []
        seq_list = []
        for trabajo in self.seq:
            print(trabajo, "|", self.seq[trabajo])
            trabajo_list.append(trabajo)
            seq_list.append(self.seq[trabajo])

        seq_df = pd.DataFrame({"trabajo": trabajo_list, "seq": seq_list})

        # save into an excel
        # combine all dataframes
        combined_df = pd.concat([processTimes_df, setupTimes_df, seq_df], axis=1)

        # File path
        (
            n_machines,
            n_jobs,
            maxlots,
            seed,
        ) = (
            len(self.machines),
            len(self.jobs),
            len(self.lots),
            self.seed,
        )
        file_path = f"v5_m{n_machines}_j{n_jobs}_u{maxlots}_s{seed}_data.xlsx"
        # Save to excel
        combined_df.to_excel(file_path, index=False, sheet_name="Sheet1")

    def sequence_dependent_setup(self):
        """Generate  and print sequence dependent setup times for jobs on machines"""
        self.sd_setup = self._random_sequence_dependent_setup(
            self.machines, self.jobs, self.t_span_setup
        )
        self._print_sequence_dependent_setup()

    def to_dict(self):
        """Convert class attributes to dictionary"""
        return {
            "machines": self.machines.astype(int).tolist(),
            "jobs": self.jobs.astype(int).tolist(),
            "lots": self.lots.astype(int).tolist(),
            "seed": self.seed,
            "seq": self.seq,
            "p_times": self.p_times,
            "setup": self.setup,
            "t_span": self.t_span,
            "t_span_setup": self.t_span_setup,
        }

    def patch_dict(self):
        """Patch the dictionary to avoid serialization issues"""
        data = self.to_dict()

        """patch seq"""
        # Create a list of keys to iterate over
        keys_to_update = list(data["seq"].keys())

        for key in keys_to_update:
            # Update the key
            new_key = int(key)
            for i, j in enumerate(data["seq"][key]):
                new_j = int(j)
                data["seq"][key][i] = new_j

            # Update the dictionary with the new key
            data["seq"][new_key] = data["seq"].pop(key)

        """patch p_times"""
        keys_to_update = list(
            data["p_times"].keys()
        )  # Create a list of keys to avoid modification during iteration
        for key in keys_to_update:
            new_key = str(key)
            data["p_times"][new_key] = int(data["p_times"].pop(key))

        """patch setup"""
        keys_to_update = list(data["setup"].keys())
        for key in keys_to_update:
            new_key = str(key)
            data["setup"][new_key] = int(data["setup"].pop(key))

        return data

    def save_to_json(self, filename, data):
        """Save the parameters to a json file"""
        with open(filename, "w") as file:
            file.write(json.dumps(data, indent=2, default=custom_serializer))


class JobShopRandomParamsSeqDep(JobShopParams):
    """Generate parameters for job-shop scheduling problems with sequence dependent
    setup times.

    This class creates randomized parameters for job-shop scenarios with sequence
    dependent setup times, including processing times for jobs on different machines.

    Attributes:
        n_machines (int): The number of machines in the job shop.
        n_jobs (int): The number of jobs to be scheduled.
        n_lots (int): The number of max lots for splitting each job.
        seed (int | None): Seed for random number generation.
        t_span (tuple): Range of possible processing times (min, max).
        t_span_setup (tuple): Range of possible setup times (min, max).

    Example:
        >>> myParams = JobShopRandomParams(3, 3, 3, seed=42)
    """

    def __init__(
        self,
        n_machines: int,
        n_jobs: int,
        n_lots: int,
        seed=0,
        t_span=(1, 20),
        t_span_setup=(50, 100),
    ):
        self.t_span = t_span
        self.seed = seed
        self.t_span_setup = t_span_setup

        demand = {product: 50 for product in range(0, n_jobs + 1)}
        machines = np.arange(n_machines, dtype=int)
        jobs = np.arange(n_jobs + 2)
        p_times = self._random_times(machines, jobs, t_span)
        lots = np.arange(n_lots, dtype=int)
        seq = self._random_sequences(machines, jobs)
        setup = self._random_setup(machines, jobs, t_span_setup)
        super().__init__(machines, jobs, p_times, seq, setup, lots, demand)

    def _random_times(self, machines, jobs, t_span):
        np.random.seed(self.seed)
        t = np.arange(t_span[0], t_span[1])
        random_times = {}
        for m in machines:
            for j in jobs:
                if j == 0 or j == jobs[-1]:
                    random_times[(m, j)] = 0
                else:
                    random_times[(m, j)] = np.random.choice(t)

        return random_times

    def _random_sequences(self, machines, jobs):
        np.random.seed(self.seed)
        random_sequence = {}
        for j in jobs:
            if j != 0 and j != jobs[-1]:
                random_sequence[j] = self._generate_random_sequence(machines)
            else:
                random_sequence[j] = list(machines)
        return random_sequence

    def _generate_random_sequence(self, machines):
        # Decide on the length of the sequence (integer between 1 and len(machines))
        sequence_length = np.random.randint(1, len(machines) + 1)

        # Randomly select machines for the sequence
        sequence = np.random.choice(machines, size=sequence_length, replace=False)
        sequence = sequence.astype(int)

        return JobSequence(sequence)

    def _random_setup(self, machines, jobs, t_span_setup):
        np.random.seed(self.seed)
        t = np.arange(
            t_span_setup[0], t_span_setup[1]
        )  # meto en un vector todos los tiempos posibles
        setup_times = {}
        for m in machines:
            for j in jobs:
                if j == 0 or j == jobs[-1]:  # for dummy jobs
                    for k in jobs:
                        setup_times[m, j, k] = 0
                else:
                    setup_times[m, j, 0] = 50
                    for k in jobs:
                        if j == k:
                            setup_times[m, j, k] = 0
                        elif k != 0:
                            setup_times[m, j, k] = np.random.choice(t)

        return setup_times

    def _printMachines(self):
        print("[MACHINES]: \n", self.machines, "\n")

    def _printJobs(self):
        print("[JOBS]: \n", self.jobs, "\n")

    def _printLots(self):
        print("[BATCHES]: \n", self.lots, "\n")

    def _printProcessTimes(self):
        print(
            "[PROCESS TIMES] working time associated with each job on each machine:"
        )
        # Determine the dimensions of the matrix
        n_columns = len(self.jobs) - 2
        n_rows = len(self.machines)

        # Create an empty matrix filled with zeros
        matrix = np.zeros((n_rows, n_columns), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.p_times.items():
            if key[1] != 0 and key[1] != self.jobs[-1]:
                matrix[key[0]][key[1] - 1] = value

        # Transpose the matrix to have jobs as rows and machines as columns
        transposed_matrix = matrix.T

        # Create a DataFrame with row and column labels
        jobs = [f"Job {i+1}" for i in range(n_rows)]
        machines = [f"Machine {j}" for j in range(n_columns)]

        df = pd.DataFrame(transposed_matrix, columns=jobs, index=machines)

        # Print the DataFrame
        print(df, "\n")

    def _printSetupTimes(self):
        print(
            "[SETUP TIMES] the setup time associated with each job on each machine is:"
        )
        for m in self.machines:
            print("Machine ", m)
            n_columns = len(self.jobs) - 2
            n_rows = len(self.jobs) - 1
            matrix = np.zeros((n_rows, n_columns), dtype=int)
            for key, value in self.setup.items():
                if key[0] == m:
                    if key[1] != 0 and key[1] != self.jobs[-1]:
                        if key[2] != self.jobs[-1]:
                            matrix[key[2]][key[1] - 1] = value

            # Create a DataFrame with row and column labels
            setup_jobs = [f"Job {i+1}" for i in range(n_columns)]
            precedence_jobs = [f"Job {j}" for j in range(n_rows)]

            df = pd.DataFrame(matrix, columns=setup_jobs, index=precedence_jobs)

            # Print the DataFrame
            print(df, "\n")

    def _printSequence(self):
        print("[SEQ] the sequence for each job is: ")
        for trabajo in self.seq:
            print(trabajo, "|", self.seq[trabajo])

    def printParams(self):
        self._printMachines()
        self._printJobs()
        self._printLots()
        self._printSequence()
        self._printProcessTimes()
        self._printSetupTimes()


def convert_keys_to_tuples(dictionary):
    """Convert dictionary keys from string to tuple"""
    return {
        tuple(int(k) for k in key[1:-1].split(", ")): value
        for key, value in dictionary.items()
    }


def convert_keys_to_integers(dictionary):
    """Convert dictionary keys from string to integer"""
    return {int(key): value for key, value in dictionary.items()}


def job_params_from_json(filename: str):
    """Returns a JobShopParams instance from a json file containing
    the parameters of a job-shop scheduling problem.

    - 'machines': list with index of machines
    - 'jobs: list with index of jobs
    - 'lots': list with index of lots
    - 'seed': seed used to generate parameters
    - 'seq': sequence of each job
    - 'p_times': unitary processing times of each job in each machine
    - 'setup': setup time of each job in each machine

    Args:
        filename (str): Name of the json file containing the parameters

    Returns
        JobShopParams: Instance of JobShopParams with the parameters
    """
    # get the json path providing its name (respect relative position)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    json_file_path = os.path.join(parent_dir, "instances", filename)

    # open json and convert str to dictionary
    with open(json_file_path, "r") as file:
        json_data = file.read()
    data = json.loads(json_data)

    # get parameters from data dict
    machines = data["machines"]
    jobs = data["jobs"]
    lots = data["lots"]
    seq = data["seq"]
    seq = convert_keys_to_integers(seq)
    p_times = data["p_times"]
    p_times = convert_keys_to_tuples(p_times)
    setup = data["setup"]
    setup = convert_keys_to_tuples(setup)

    return JobShopParams(machines, jobs, p_times, seq, setup, lots)


def main():
    myParams = JobShopRandomParams(3, 3, 3, seed=42)
    print(myParams.demand)


if __name__ == "__main__":
    main()

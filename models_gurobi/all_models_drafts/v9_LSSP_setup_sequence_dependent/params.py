'''
DEFINITION OF CLASSES TO HANDLE JOBSHOP PARAMETERS
----------------------------------------------------------------------
- Classes, methods and functions are defined to facilitate the handling of jobshop parameters in the script where the model is defined.

- Two functions are included 
    - JobShopRandomParams.save_to_json to export the instance parameters to a .json file. 
    - job_params_from_json to get the parameters of the problem from a .json file
'''

import numpy as np
from typing import Iterable, Any
import json
import pandas as pd
import os

def custom_serializer(obj):
    if isinstance(obj, tuple):
        return str(obj)
    return obj

class JobSequence(list):
    
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
    
    def __init__(self, machines: Iterable, jobs: Iterable, p_times: dict, seq: dict, setup: dict, lotes: Iterable):
        """White label class for job-shop parameters

        Parameters
        ----------
        machines : Iterable
            Set of machines
            
        jobs : Iterable
            Set of jobs
        
        p_times : dict
            Processing times indexed by pairs machine, job
        
        seq : dict
            Sequence of operations (machines) of each job
        """
        self.machines = machines
        self.jobs = jobs
        self.p_times = p_times
        self.seq = seq
        self.setup = setup
        self.lotes = lotes

class JobShopRandomParams(JobShopParams):
    
    def __init__(self, n_machines: int, n_jobs: int, n_lotes: int, t_span=(1, 20), seed=None, t_span_setup=(50,100)):
        """Class for generating job-shop parameters

        Parameters
        ----------
        n_machines : int
            Number of machines
        
        n_jobs : int
            Number of jobs
        
        t_span : tuple, optional
            Processing times range, by default (1, 20)
        
        seed : int | None, optional
            numpy random seed, by default None
        """
        self.t_span = t_span
        self.seed = seed
        self.t_span_setup = t_span_setup
        
        machines = np.arange(n_machines, dtype=int)
        jobs = np.arange(n_jobs+2)
        p_times = self._random_times(machines, jobs, t_span)
        lotes=np.arange(n_lotes,dtype=int)
        # p_times_lotes = self.p_times_lotes(machines, jobs, p_times, lotes)
        seq = self._random_sequences(machines, jobs)
        setup = self._random_setup(machines, jobs, t_span_setup)
        super().__init__(machines, jobs, p_times, seq, setup, lotes)
    
    def _random_times(self, machines, jobs, t_span):
        np.random.seed(self.seed)
        t = np.arange(t_span[0], t_span[1])
        random_times={}
        for m in machines:
                for j in jobs:
                    if j ==0 or j==jobs[-1]:
                        random_times[(m,j)]=0
                    else:
                        random_times[(m,j)]=np.random.choice(t)
        
        return random_times       
        
    def _random_sequences(self, machines, jobs):
        np.random.seed(self.seed)
        random_sequence={}
        for j in jobs:
            if j !=0 and j!=jobs[-1]:
                random_sequence[j]=self._generate_random_sequence(machines)
            else:
                random_sequence[j]=list(machines)
        return  random_sequence    

    def _generate_random_sequence(self, machines):
        # Decide on the length of the sequence (can be any number between 1 and len(machines))
        sequence_length = np.random.randint(1, len(machines) + 1)

        # Randomly select machines for the sequence
        sequence = np.random.choice(machines, size=sequence_length, replace=False)
        sequence =sequence.astype(int)

        return JobSequence(sequence)
    
    def _random_setup(self, machines, jobs, t_span_setup):
        np.random.seed(self.seed)
        t=np.arange(t_span_setup[0],t_span_setup[1]) # meto en un vector todos los tiempos posibles
        setup_times={}
        for m in machines:
            for j in jobs:
                if j==0 or j==jobs[-1]: # for dummy jobs
                    for k in jobs:
                        setup_times[m,j,k]=0
                else:
                    setup_times[m,j,0] = 50
                    for k in jobs:
                        if j==k:
                            setup_times[m,j,k]=0
                        elif k!=0:
                            setup_times[m,j,k] = np.random.choice(t)
        
        return setup_times

    def printMachines(self):
        print("[MACHINES]: \n", self.machines, "\n")
    
    def printJobs(self):
        print("[JOBS]: \n", self.jobs, "\n")

    def printLotes(self):
        print("[BATCHES]: \n", self.lotes, "\n")
    
    def printProcessTimes(self):
        print("[PROCESS TIMES]the working time associated with each job on each machine is:")
        # Determine the dimensions of the matrix
        n_columns = len(self.jobs)-2
        n_rows = len(self.machines)

        # Create an empty matrix filled with zeros
        matrix = np.zeros((n_rows,n_columns), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.p_times.items():
            if key[1]!=0 and key[1]!=self.jobs[-1]:
                matrix[key[0]][key[1]-1] = value
        
                
        # Transpose the matrix to have jobs as rows and machines as columns
        transposed_matrix = matrix.T

        # Create a DataFrame with row and column labels
        jobs = [f'Job {i+1}' for i in range(n_rows)]
        machines = [f'Machine {j}' for j in range(n_columns)]

        df = pd.DataFrame(transposed_matrix, columns=jobs, index=machines)

        # Print the DataFrame
        print(df, "\n")

    def printSetupTimes(self):
        print("[SETUP TIMES] the setup time associated with each job on each machine is:")
        for m in self.machines:
            print("Machine ", m)
            n_columns=len(self.jobs)-2
            n_rows=len(self.jobs)-1
            matrix = np.zeros((n_rows, n_columns), dtype=int)
            for key, value in self.setup.items():
                if key[0]==m:
                    if key[1]!=0 and key[1]!=self.jobs[-1]:
                        if key[2]!=self.jobs[-1]:
                            matrix[key[2]][key[1]-1] = value

            # Create a DataFrame with row and column labels
            setup_jobs = [f'Job {i+1}' for i in range(n_columns)]
            precedence_jobs = [f'Job {j}' for j in range(n_rows)]

            df = pd.DataFrame(matrix, columns=setup_jobs, index=precedence_jobs)

            # Print the DataFrame
            print(df, "\n")

    def printSequence(self):
        print("[SEQ] the sequence for each job is: ")
        for trabajo in self.seq:
            print(trabajo, "|", self.seq[trabajo])
    
    def printParams(self):
        self.printMachines()
        self.printJobs()
        self.printLotes()
        self.printSequence()
        self.printProcessTimes()
        self.printSetupTimes()

    def to_dict(self):
        """Convert class attributes to dictionary"""
        return {
            'machines': self.machines.astype(int).tolist(),
            'jobs': self.jobs.astype(int).tolist(),
            'lotes': self.lotes.astype(int).tolist(),
            'seed': self.seed,
            'seq': self.seq,
            'p_times': self.p_times,
            'setup': self.setup,
            't_span': self.t_span,
            't_span_setup': self.t_span_setup,     
        }
    
    def patch_dict(self):
        data = self.to_dict()
        
        '''patch seq'''
        # Create a list of keys to iterate over
        keys_to_update = list(data['seq'].keys())

        for key in keys_to_update:
            # Update the key
            new_key = int(key)
            for i,j in enumerate(data['seq'][key]):
                new_j=int(j)
                data['seq'][key][i] = new_j

            # Update the dictionary with the new key
            data['seq'][new_key] = data['seq'].pop(key)
        
        '''patch p_times'''
        keys_to_update = list(data['p_times'].keys())  # Create a list of keys to avoid modification during iteration
        for key in keys_to_update:
            new_key = str(key)
            data['p_times'][new_key] = int(data['p_times'].pop(key))

        '''patch setup'''
        keys_to_update = list(data['setup'].keys())
        for key in keys_to_update:
            new_key=str(key)
            data['setup'][new_key]= int(data['setup'].pop(key))

        return data
    
    def save_to_json(self, filename, data):
        with open(filename, 'w') as file:
            file.write(json.dumps(data, indent=2, default=custom_serializer))
    
def convert_keys_to_tuples(dictionary):
    return {tuple(int(k) for k in key[1:-1].split(', ')): value for key, value in dictionary.items()}

def convert_keys_to_integers(dictionary):
    return {int(key): value for key, value in dictionary.items()}

def job_params_from_json(filename: str):
    """Returns a JobShopParams instance from a json file containing
    - 'machines': list with index of machines
    - 'jobs: list with index of jobs
    - 'lotes': list with index of lots
    - 'seed': seed used to generate parameters
    - 'seq': sequence of each job
    - 'p_times': unitary processing times of each job in each machine
    - 'setup': setup time of each job in each machine

    Parameters
    ----------
    filename : str
        Filename of json

    Returns
    -------
    JobShopParams
        Parameters of problem
    """
    # get the json path providing its name (respect relative position)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    json_file_path = os.path.join(parent_dir, 'instances', filename)

    # open json and convert str to dictionary
    with open(json_file_path, 'r') as file:
        json_data = file.read()
    data = json.loads(json_data)

    # get parameters from data dict
    machines = data['machines']
    jobs = data['jobs']
    lotes = data['lotes']
    seq = data['seq']
    seq = convert_keys_to_integers(seq)
    p_times = data['p_times']
    p_times = convert_keys_to_tuples(p_times)
    setup = data['setup']
    setup = convert_keys_to_tuples(setup)

    return JobShopParams(machines, jobs, p_times, seq, setup, lotes)


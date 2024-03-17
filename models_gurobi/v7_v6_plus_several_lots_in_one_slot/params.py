'''
DEFINICION DE CLASES PARA MANEJAR PARÁMETROS DEL JOBSHOP
----------------------------------------------------------------------Se definen clases, métodos y funciones para facilitar el manejo de los parámetros del jobshop en el script donde se define el modelo
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
    
    def __init__(self, n_machines: int, n_jobs: int, n_lotes: int, t_span=(1, 10), seed=None, t_span_setup=(0,40)):
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
        jobs = np.arange(n_jobs)
        p_times = self._random_times(machines, jobs, t_span)
        lotes=np.arange(n_lotes,dtype=int)
        # p_times_lotes = self.p_times_lotes(machines, jobs, p_times, lotes)
        seq = self._random_sequences(machines, jobs)
        setup = self._random_setup(machines, jobs, t_span_setup)
        super().__init__(machines, jobs, p_times, seq, setup, lotes)
    
    def _random_times(self, machines, jobs, t_span):
        np.random.seed(self.seed)
        t = np.arange(t_span[0], t_span[1])
        return {
            (m, j): np.random.choice(t)
            for m in machines
            for j in jobs
        }
    
    def _random_sequences(self, machines, jobs):
        np.random.seed(self.seed)
        return {
            j: self._generate_random_sequence(machines)
            for j in jobs
        }

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
        return{
            (m,j): np.random.choice(t)
            for m in machines
            for j in jobs
        }  

    # def p_times_lotes(self, machines, jobs, p_times, lotes):
    #     quantity={0:100, 1:200}
    #     return {
    #         (m, j, t): p_times.get((m,j))*(quantity[j]/len(lotes))
    #         for m in machines
    #         for j in jobs
    #         for t in lotes
    #     }

        
    def printParams(self):
        print("las máquinas son: \n", self.machines, "\n")
        print("los trabajos son: \n", self.jobs, "\n")
        print("las lotes son: \n", self.lotes, "\n")
        print("el tiempo de trabajo asociado a cada trabajo en cada máquina es: \n")
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
        jobs = [f'Job {i}' for i in range(max_job + 1)]
        machines = [f'Machine {j}' for j in range(max_machine + 1)]

        df = pd.DataFrame(transposed_matrix, columns=jobs, index=machines)

        # Print the DataFrame
        print(df, "\n")

        print("el tiempo de setup asociado a cada trabajo en cada máquina es: \n")
        # Determine the dimensions of the matrix
        max_job = max(key[0] for key in self.setup.keys())
        max_machine = max(key[1] for key in self.setup.keys())

        # Create an empty matrix filled with zeros
        matrix = np.zeros((max_job + 1, max_machine + 1), dtype=int)

        # Fill the matrix with the given data
        for key, value in self.setup.items():
            matrix[key[0]][key[1]] = value

        # Transpose the matrix to have jobs as rows and machines as columns
        transposed_matrix = matrix.T

        # Create a DataFrame with row and column labels
        jobs = [f'Job {i}' for i in range(max_job + 1)]
        machines = [f'Machine {j}' for j in range(max_machine + 1)]

        df = pd.DataFrame(transposed_matrix, columns=jobs, index=machines)

        # Print the DataFrame
        print(df, "\n")

        print("la secuencia para cada trabajo es: ")
        for trabajo in self.seq:
            print(trabajo, "|", self.seq[trabajo])
        
        print("los tiempos de proceso unitarios son: ")
        print(self.p_times)
    
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


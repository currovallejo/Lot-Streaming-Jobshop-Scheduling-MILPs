import pyomo.environ as pyo
import matplotlib as mpl # para usar funciones matplotlib
import matplotlib.pyplot as plt # para pintar gráficas
import pandas as pd


class JobShopModel(pyo.ConcreteModel): # modelo tipo Concrete
    
    cmap = mpl.colormaps["Dark2"] # mapa de colores qualitative
    colors = cmap.colors #extraigo la lista de colores asociados al mapa
    
    def __init__(self, params, **kwargs): # constructor
        super().__init__() # llama al constructor de la clase padre
        self.params = params
        self._construct_sets() # construye sets machines & jobs
        self._construct_params() # 

        ''' **kwargs allows you to pass a variable number of keyword arguments to a function.
        The double asterisks (**) before kwargs indicate that it should collect keyword arguments.
        A keyword argument is a type of argument that is passed to a function or method using a keyword (parameter name) along with its associated value.
        EXAMPLE:
        my_function(param1=1, param2=2, param3=3)
        '''
        

    
    @property # used to define a method as a "getter" for a property
    def seq(self): 
        return self.params.seq
    
    @property
    def tandas(self):
        return self.params.tandas
    
    @property
    def jobs(self):
        return self.params.jobs
    
    @property
    def machines(self):
        return self.params.machines
    
    def _construct_sets(self): # defino sets (constantes)
        self.M = pyo.Set(initialize=self.params.machines)
        self.J = pyo.Set(initialize=self.params.jobs)
        self.T = pyo.Set(initialize=self.params.tandas)
    
    def _construct_params(self): # parametro p_ij asociado a sets M,N
        self.p = pyo.Param(self.M, self.J, self.T, initialize=self.params.p_times_tandas)
        self.s = pyo.Param(self.M, self.J, initialize=self.params.setup)
        self.V = sum(self.p[key] for key in self.p)
        '''
        Param en pyomo es: data that must be provided in order to find an optimal (or good) assignment of values to the decision variables.
        self.p --> Matriz donde se representa el tiempo de proceso de cada trabajo en cada máquina
        self.V --> suma todos los elementos de la matriz
        '''
    
    def plot(self, horizontal=True, figsize=[7, 3], dpi=100, colors=None): # method to create gantt charts
        if horizontal:
            self._plot_horizontal(figsize=figsize, dpi=dpi, colors=colors)
        else:
            self._plot_vertical(figsize=figsize, dpi=dpi, colors=colors)

    def _plot_vertical(self, figsize=[7, 3], dpi=100, colors=None):
        
        if colors is None:
            colors = self.colors
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi) 
        # crea una nueva figura y ejes para graficar
        

        for i, j in enumerate(self.J): # enumerate devuelve una tupla formada por el índice y el elemento --> (índice, elemento)
            for t in enumerate(self.T):
                machines, starts, spans, starts_setup, spans_setup = self._get_elements(j, t) # usado en disjunctive
                
                if i >= len(colors):
                    i = i % len(colors) # reusar colores cíclicamente
                
                color = colors[i]
                ax.bar(machines, spans_setup, bottom=starts_setup, color='white', edgecolor='black')
                ax.bar(machines, spans, bottom=starts, label=f"Job {j}", color=color)
            '''
            machines = x-coordinates
            spans = height of each bar
            bottom = y-coordinate for the bootom of the bar
            label = label each bar with each job j
            color = color of the bar
            '''

        ax.set_xticks(self.M) # x-axis ticks 
        ax.set_xlabel("Machine") # x-axis label
        ax.set_ylabel("Time") # y-axis label
        y_axe = [i for i in range (61)]
        ax.set_yticks(y_axe)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) #adds the legend for each ax.bar
        fig.tight_layout() # ensures the layout is well-adjusted
        plt.show() # grafica el gantt

    def _plot_horizontal(self, figsize=[7, 3], dpi=100, colors=None):
        
        colors = self._get_colors(colors)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, j in enumerate(self.J):
            for k, t in enumerate(self.T):
                machines, starts, spans, starts_setup, spans_setup = self._get_elements(j, t) # usado en disjunctive
            
                if i >= len(colors):
                    i = i % len(colors)
                
                color = colors[i]
                ax.barh(machines, width=spans_setup, left=starts_setup, color='white', edgecolor='black', hatch="/")
                ax.barh(machines, width=spans, left=starts, color=color, edgecolor='black', label=f"Job {j}")

        ax.set_yticks(self.M)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        # x_axe = [i for i in range (0, 61, 2)]
        # ax.set_xticks(x_axe)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
        fig.tight_layout() # ensure
        plt.show()
    
    def plot_variables(self):
        ma,jo,ta,ss,se,ps,pe = [list() for _ in range(7)]
        for t in self.T:         
            for j in self.J:                  
                for m in self.M:
                    machines, starts, spans, starts_setup, spans_setup = self._get_elements(j, t)

                    # #start setup dates
                    # print("m =", m, "del j =", j, "en t =", t, "setup en:", starts_setup[m], "trabajo en:", starts[m])

                    ma.append(m)
                    jo.append(j)
                    ta.append(t)
                    ss.append(starts_setup[m])
                    se_value=starts_setup[m]+spans_setup[m]
                    se.append(se_value)
                    ps.append(starts[m])
                    pe_value=starts[m]+spans[m]
                    pe.append(pe_value)

    
        data = {'Machine':ma, 'job' :jo, 'tanda':ta, 'Setup_start':ss, 'setup_end':se, 'process_start':ps, 'process_end':pe}
        
        df=pd.DataFrame(data)
        print(df)
            
        # for j in self.J:
        #     for t in self.T:        
        #         for m in self.M:
        #             print("el proceso en la máquina ", m, "del trabajo ", j, "en la tanda ", t, "empieza en el segundo: ", starts[m])
                
        #         print("el tiempo de trabajo asociado a cada trabajo en cada máquina es: \n")
    
    def _get_colors(self, colors):
        if colors is None:
            colors = self.colors
        return colors

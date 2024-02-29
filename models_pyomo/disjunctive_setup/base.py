import pyomo.environ as pyo
import matplotlib as mpl # para usar funciones matplotlib
import matplotlib.pyplot as plt # para pintar gráficas


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
    
    def _construct_sets(self): # defino sets (constantes)
        self.M = pyo.Set(initialize=self.params.machines)
        self.J = pyo.Set(initialize=self.params.jobs)
    
    def _construct_params(self): # parametro p_ij asociado a sets M,N
        self.p = pyo.Param(self.M, self.J, initialize=self.params.p_times)
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
        

        for i, j in enumerate(self.J):
            machines, starts, spans, starts_setup, spans_setup = self._get_elements(j) # usado en disjunctive
            
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
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03)) #adds the legend for each ax.bar
        fig.tight_layout() # ensures the layout is well-adjusted
        plt.show() # grafica el gantt

    def _plot_horizontal(self, figsize=[7, 3], dpi=100, colors=None):
        
        colors = self._get_colors(colors)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for i, j in enumerate(self.J):
            machines, starts, spans, starts_setup, spans_setup = self._get_elements(j)
            
            if i >= len(colors):
                i = i % len(colors)
            
            color = colors[i]
            ax.barh(machines, width=spans_setup, left=starts_setup, color='white', edgecolor='black', hatch="/")
            ax.barh(machines, width=spans, left=starts, color=color, edgecolor='black', label=f"Job {j}")


        ax.set_yticks(self.M)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1.03))
        fig.tight_layout() # ensure
        plt.show()
    
    def plot_variables(self):

        # obtengo los datos de cada trabajo
        for j in self.J:
            machines, starts, spans, starts_setup, spans_setup = self._get_elements(j)

            # start setup dates
            for m in self.M:
                print("el start_setup del trabajo ", j, "en la máquina ", m, "es: ", starts_setup[m])
    
    def _get_colors(self, colors):
        if colors is None:
            colors = self.colors
        return colors
    

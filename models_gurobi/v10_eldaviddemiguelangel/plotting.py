import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
import plotly.express as px

#--------- PRINT PROBLEM DATA ---------
def printData(params, demand):
    print("--------- JOBSHOP DATA --------- \n")
    # job shop parameters
    params.printParams()
    
    # demand
    jobs = [f'Job {i}' for i in range(len(demand))]
    demandDf = pd.DataFrame(list(demand.values()), columns=['Demand'], index=jobs)
    print("[DEMAND] The current demand is: \n", demandDf, "\n")

#--------- PLOT SOLUTION EVOLUTION ---------
def plot_solution_evolution(elapsed_times, objectives, gaps, show=True):

    # Plot the objective value over time
    fig, ax1 = plt.subplots()

    # Plot the objective value with the primary y-axis
    ax1.plot(elapsed_times, objectives, marker='o', color='tab:blue', label='Objective Value')
    ax1.set_xlabel('Tiempo (minutos)', fontsize=14)
    ax1.set_ylabel('Makespan', color='tab:blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Set major and minor ticks on the x-axis
    ax1.xaxis.set_major_locator(MultipleLocator(300))
    ax1.xaxis.set_minor_locator(MultipleLocator(60))

    # Customize tick parameters
    ax1.tick_params(axis='x', which='minor', length=6, width=1)  # Increase minor tick size
    ax1.tick_params(axis='x', which='major', length=10, width=2)  # Customize major tick size
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x/60)))

    # Create a secondary y-axis for the MIP gap
    ax2 = ax1.twinx()
    ax2.plot(elapsed_times, [gap * 100 for gap in gaps], marker='x', color='tab:red', label='MIP Gap')
    ax2.set_ylabel('MIP Gap (%)', color='tab:red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Add a custom formatter to display percentages on the y-axis
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y / 100)))

    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plt.title('Evolución del Makespan y MIP Gap en el tiempo', fontsize=18)
    plt.grid(True)
    if show:
        plt.show()

#--------- PLOT GANTT CHART ---------
def plot_pxt_gantt(df_results,params,demand, shift_time:int, show=True, version=0):

    machines, jobs, maxlots, seed = len(params.machines), len(params.jobs), len(params.lotes), params.seed

    # converting all dataframe numbers to int format
    df = df_results.astype(int)

    # add "labels" for plotting discrete
    lst_aux=[]
    for j in df['Job'].tolist():
        lst_aux.append('P %s' %j)
    df['Products']=lst_aux
    lst_aux.clear()
    for j in df['Machine'].tolist():
        lst_aux.append('M %s' %j)
    df['Resources']=lst_aux
    lst_aux.clear()
    for i,j in enumerate(df['Job'].tolist()):
            u = df.loc[i,'Lote']
            lst_aux.append('P%s - L%s' %(j,u))
    df['Text']=lst_aux

    print('dataframe es: \n', df)

    # length of the bars
    df["delta"] = df["makespan"] - df['Start Time (x)']
    print(df['delta'])

    # Create a figure with Plotly colorscale
    fig = px.timeline(df,
                        x_start='Start Time (x)',
                        x_end='makespan', 
                        y='Resources', 
                        color='Products',
                        title='Job Shop Schedule with Lot Streaming',
                        text='Text',
                        hover_data={'Start Time (x)':True,'makespan':True,'Lotsize':True,'Text':False,'Resources':False}
                        # category_orders={'Machine': sorted(df['Machine'].unique())},
                        # labels={'y':'Machine', 'x':'Job'},
                        # range_x=[0,max(df['makespan'])]
                        )
                
    # Set the X-axis type to 'linear'
    fig.layout.xaxis.type = "linear"

    fig.update_xaxes(tick0=0, dtick=shift_time)
        
    for j,Bar in enumerate(fig.data):
        # columna de dataframe de filtrado
        filt = df["Products"] == Bar.name 

        #filtrado de la columna delta
        Bar.x = df[filt]["delta"].tolist() 

    # length of the setup bars 
    df["delta_s"] = df["Start Time (x)"] - df['Setup Start Time (y)']
    # print(df['delta_s'])

    fig_s = px.timeline(df,
                        x_start='Setup Start Time (y)',
                        x_end='Start Time (x)', 
                        y='Resources',
                        # category_orders={'Machine': sorted(df['Machine'].unique())},
                        # labels={'y':'Machine', 'x':'Job'},
                        # range_x=[0,max(df['makespan'])]
                        )

    # Set the X-axis type to 'linear'
    fig_s.layout.xaxis.type = "linear"
    fig_s.update_traces(marker_pattern_shape="/")
    fig_s.update_xaxes(tick0=0, dtick=200)
    fig_s.update_traces(marker_color='white')
        
    for j,Bar in enumerate(fig_s.data):

        Bar.x = df["delta_s"].tolist()
        Bar.legendgroup = "Setup"
        Bar.name = "Setup"
        Bar.showlegend = True
        fig.add_trace(Bar)

    fig.write_html(f'v10_m{machines}_j{jobs}_u{maxlots}_s{seed}_d{demand}_sht{shift_time}_setup_no_pmtn_v{version}.html', (machines,jobs,maxlots, seed,demand, version), auto_open=True if show else False)

if __name__=='__main__':
    pass
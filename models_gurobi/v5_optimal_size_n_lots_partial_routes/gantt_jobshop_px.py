'''
DEVELOPMENT OF A GANTT PLOTTING FUNCTION USING PLOTLY EXPRESS
----------------------------------------------------------------------
- The gantt in plotly is clearer, more presentable and allows to see more data.
- The plotly express timeline library is used.
'''

import plotly.express as px
import pandas as pd
# from model_v5 import params, p, x, y, q, results_df

def print_pxt_gantt(df_results):
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
                        title='Job Shop Schedule',
                        text='Text',
                        hover_data={'Start Time (x)':True,'makespan':True,'Lotsize':True,'Text':False,'Resources':False}
                        # category_orders={'Machine': sorted(df['Machine'].unique())},
                        # labels={'y':'Machine', 'x':'Job'},
                        # range_x=[0,max(df['makespan'])]
                        )
                
    # Set the X-axis type to 'linear'
    fig.layout.xaxis.type = "linear"

    fig.update_xaxes(tick0=0, dtick=200)
        
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

    # print(fig_s.data)

    # fig.show()
    fig.write_html('jobshop_gantt_' + 'plot.html', auto_open=True)


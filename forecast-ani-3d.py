import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

data = {
    'Month': ['January 2022', 'February 2022', 'March 2022', 'April 2022', 'May 2022', 'June 2022',
              'July 2022', 'August 2022', 'September 2022', 'October 2022', 'November 2022', 'December 2022',
              'January 2023', 'February 2023', 'March 2023', 'April 2023', 'May 2023', 'June 2023',
              'July 2023', 'August 2023', 'September 2023'],
    'Sales': [10000, 12000, 11500, 13000, 14500, 15000, 16000, 18000, 17500, 19000, 20500, 22000,
              23500, 24000, 25000, 26500, 27000, 29000, 30500, 31000, 32500]
}

df = pd.DataFrame(data)

df['Month'] = pd.to_datetime(df['Month'], format='%B %Y')

df['Months_Since_Start'] = np.arange(len(df)) 

X = df['Months_Since_Start'].values.reshape(-1, 1)  
y = df['Sales'].values  

model = LinearRegression()
model.fit(X, y)

forecast_periods = 6
future_months = np.arange(len(df), len(df) + forecast_periods).reshape(-1, 1)  
future_sales = model.predict(future_months)  

future_dates = pd.date_range(df['Month'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq='ME')
df_future = pd.DataFrame({'Month': future_dates, 'Sales': future_sales})

df_combined = pd.concat([df, df_future], ignore_index=True)

fig = make_subplots(
    rows=1, cols=1, specs=[[{'type': 'scatter3d'}]],
    subplot_titles=['Forecasted Sales & Trendline']
)

X_vals = np.arange(len(df))  
Y_vals = df['Sales']  
Z_vals = np.zeros_like(X_vals)  

future_X_vals = np.arange(len(df_combined)) 
future_Y_vals = model.predict(future_X_vals.reshape(-1, 1))  
future_Z_vals = np.zeros_like(future_X_vals) 

fig.add_trace(go.Scatter3d(x=future_X_vals, y=future_Y_vals, z=future_Z_vals,
                           mode='markers+lines', marker=dict(color='red', size=8),
                           name='Forecasted Sales & Trendline'))

frames = []
for i in range(1, len(df_combined)+1):
    frames.append(go.Frame(
        data=[ 
            go.Scatter3d(x=future_X_vals[:i], y=future_Y_vals[:i], z=future_Z_vals[:i],
                         mode='markers+lines', marker=dict(color='red', size=8),
                         line=dict(color='green', dash='dash'),
                         name='Forecasted Sales & Trendline')
        ],
        name=f'Frame {i}'
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='Months',
        yaxis_title='Sales',
        zaxis_title='Z',
    ),
    title='Sales Over Time with Dynamic Forecast',
    margin=dict(l=0, r=200, b=100, t=40),  
    height=800,
    showlegend=True,
    autosize=True, 
    updatemenus=[],  
    xaxis=dict(range=[0, len(df) + forecast_periods]),
    yaxis=dict(range=[0, max(df['Sales']) + 5000]),
    dragmode='pan',  
    sliders=[{
        'steps': [{'args': [[f'Frame {i}'], {'frame': {'duration': 200, 'redraw': True}, 'mode': 'immediate'}], 'label': f'Frame {i}', 'method': 'animate'} for i in range(1, len(df_combined)+1)],
        'currentvalue': {'visible': True, 'prefix': 'Month: ', 'font': {'size': 16}},
        'len': 1.0,
    }]
)

fig.frames = frames

fig.add_annotation(
    text="<b>SUBJECT</b>: Financial Management<br><b>PROFESSOR</b>: Mary Jane S. Legaspi <br><b>SECTION</b>: BSIS 303 <br> &#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;&#8212;",
    x=1.1, y=0.8, showarrow=False,
    font=dict(size=16, color="black", family="Arial"), align="left",
    xref="paper", yref="paper", 
    ax=0, ay=0
)

fig.add_annotation(
    text="<b>GROUP MEMBERS</b>:<br><br>MARTILLOS, JOHN NEMUEL<br>AQUINO, DIANA ROSE<br>SAMSON, BENCH<br>PERIODICO, ANGELO",
    x=1.1, y=0.4, showarrow=False,
    font=dict(size=16, color="black", family="Arial"), align="left",
    xref="paper", yref="paper",  
    ax=0, ay=0
)

fig.write_html("index.html")

print("3D dynamic and looping forecast plot saved to 'index.html'")

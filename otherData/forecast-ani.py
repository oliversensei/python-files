from flask import Flask, render_template, send_file 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
import io

app = Flask(__name__)

@app.route('/')
def home():
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

    plt.figure(figsize=(10, 6))
    plt.plot(df['Month'], df['Sales'], marker='o', linestyle='-', color='b', label='Actual Sales')
    plt.plot(df_combined['Month'], model.predict(np.arange(len(df_combined)).reshape(-1, 1)), 
             linestyle='--', color='r', label='Trendline (Forecasted)')
    plt.plot(df_future['Month'], df_future['Sales'], marker='x', linestyle='', color='r', label='Forecasted Sales')

    plt.title('Sales Over Time with Forecast (January 2022 - March 2024)', fontsize=16, pad=20)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Sales (P)', fontsize=14)

    plt.xticks(rotation=45)

    plt.grid(True)

    plt.legend()

    plt.figtext(0.5, -0.10, 'Group Members', ha='center', fontsize=18, fontweight='bold', color='darkblue')
    plt.figtext(0.5, -0.15, 'Martillos, Periodico, Samson, Periodico', ha='center', fontsize=14, fontweight='normal', color='black')
    plt.figtext(0.5, -0.20, 'Section: BSIS 303', ha='center', fontsize=14, fontweight='normal', color='black')

    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')  
    img.seek(0)

    plt.close()

    return send_file(img, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)

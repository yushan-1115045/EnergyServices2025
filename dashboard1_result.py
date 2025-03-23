import pickle
import urllib
import dash
from dash import dcc,html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn import metrics
import numpy as np

df = pd.read_excel('forecast_data.xlsx')
df['Date'] = pd.to_datetime(df['Date'])  # create a new column 'data time' of datetime type
df2 = df.iloc[:, 1:8]
X2 = df2.values
# Use Melt to convert multiple columns of data into a long format
df_melted = df.melt(id_vars=['Date'], value_vars=df.columns[1:8], var_name='variable', value_name='value')
# Draw an overlay line chart
fig1 = px.line(df_melted, x='Date', y='value', color='variable', title='Multi-line overlay chart')
# fig1.show()
df_real = pd.read_excel('testData_2019_SouthTower.xlsx')
y2 = df_real['South Tower (kWh)'].values

# load RF model
with open('south_tower_model_hour.plk', "rb") as f:
    RF_model2 = pickle.load(f)

y2_pred_RF = RF_model2.predict(X2)
#Evaluate errors
MAE_RF = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_RF = np.mean(y2 - y2_pred_RF)
MSE_RF = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_RF = np.sqrt(metrics.mean_squared_error(y2, y2_pred_RF))
cvRMSE_RF = RMSE_RF/np.mean(y2)
NMBE_RF = MBE_RF/np.mean(y2)
# print("MAE_RF:",MAE_RF)
# print("MBE_RF:",MBE_RF)
# print("MSE_RF:",MSE_RF)
# print("RMSE_RF:",RMSE_RF)
# print("cvRMSE_RF:",cvRMSE_RF)
# print("NMBE_RF:",MBE_RF)


# load LSTM model
with open('LR_model.plk', "rb") as f:
    LR_model2 = pickle.load(f)
y2_pred_LR = LR_model2.predict(X2)
#Evaluate errors
MAE_LR = metrics.mean_absolute_error(y2, y2_pred_LR)
MBE_LR = np.mean(y2 - y2_pred_LR)
MSE_LR = metrics.mean_squared_error(y2, y2_pred_LR)
RMSE_LR = np.sqrt(metrics.mean_squared_error(y2, y2_pred_LR))
cvRMSE_LR = RMSE_LR/np.mean(y2)
NMBE_LR = MBE_LR/np.mean(y2)
# print("MAE_LR:",MAE_LR)
# print("MBE_LR:",MBE_LR)
# print("MSE_LR:",MSE_LR)
# print("RMSE_LR:",RMSE_LR)
# print("cvRMSE_LR:",cvRMSE_LR)
# print("NMBE_LR:",MBE_LR)

# Create data frames with prediction results and error metrics
d={'Methods':['RandomForest','LR'],'MAE':[MAE_RF,MAE_LR],'MBE':[MBE_RF,MBE_LR],'MSE':[MSE_RF,MSE_LR],'RMSE':[RMSE_RF,RMSE_LR],'cvRMSE':[cvRMSE_RF,cvRMSE_LR],'NMBE':[NMBE_RF,NMBE_LR]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'RandomForest':y2_pred_RF,'LR':y2_pred_LR}
df_forecast = pd.DataFrame(data=d)

# merge real and forecast results and creantes a figure with it
df_results = pd.merge(df_real,df_forecast,on='Date')
fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:12])
# fig2.show()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H2('IST Energy Forecast tool (kWh)'),
    dcc.Tabs(id='tabs',value='tab-1',children=[
        dcc.Tab(label='📊 Raw Data', value='tab-1'),
        dcc.Tab(label='🔮 Forecast', value='tab-2'),
        dcc.Tab(label='User Input Forecast', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

@app.callback(Output('tabs-content', 'children'),Input('tabs','value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Dropdown(
                id='chart-type-selector',
                options=[
                    {'label': 'Line chart', 'value': 'line'},
                    {'label': 'histogram', 'value': 'bar'},
                    {'label': 'Scatter plot', 'value': 'scatter'}
                ],
                value='line',
                style={'width': '50%'}
            ),
            dcc.Graph(id='raw-data-graph')
        ])
    # Add missing Forecast page logic
    elif tab == 'tab-2':
     return html.Div([
        html.H3('IST Electricity Forecast (kWh)'),
        dcc.Checklist(
            id='model-selector',
            options=[
                {'label': ' Random Forest', 'value': 'RandomForest'},
                {'label': ' Linear Regression', 'value': 'LR'},
                {'label': ' Show All', 'value': 'All'}
            ],
            value=['All'],
            inline=True,
            style={'margin-bottom': '20px'}
        ),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=df_results['Date'].min(),
            max_date_allowed=df_results['Date'].max(),
            initial_visible_month=df_results['Date'].min(),
            start_date=df_results['Date'].min(),
            end_date=df_results['Date'].max()
        ),
        dcc.Graph(id='forecast-graph'),
        #Added download data link
        html.A(
             'Download Data',
             id='download-link',
             download="forecast_data.csv",
             href="",
             target="_blank",
             style={'fontSize': '25px'}
        ),
        html.Div(id='metrics-table'),
    ])
    elif tab == 'tab-3':
      return html.Div([
          html.H3('Forecasting for Next Hour Based on User Inputs',style={'color':'#52C2C6'}),
          html.Div([
              html.Label('lag1'),
              dcc.Input(id='lag1-input', type='number', placeholder='Lag 1', value=88),
              html.Label('rolling24h'),
              dcc.Input(id='rolling24h-input', type='number', placeholder='Rolling 24h', value=88),
              html.Label('temp_C'),
              dcc.Input(id='tempC-input', type='number', placeholder='Temperature (°C)', value=10),
              html.Label('hour'),
              dcc.Input(id='hour-input', type='number', min=0, max=23, placeholder='Hour (0-23)', value=0),
              html.Label('lag24'),
              dcc.Input(id='lag24-input', type='number', placeholder='Lag 24', value=88),
              html.Label('rolling3h'),
              dcc.Input(id='rolling3h-input', type='number', placeholder='Rolling 3h', value=88),
              html.Label('weekday'),
              dcc.Input(id='weekday-input', type='number', min=0, max=6, placeholder='Weekday (0-6)', value=1),
              html.Button('Predict', id='predict-button', n_clicks=0,style={'background-color': 'blue', 'color': 'white'})
          ]),
          html.Div(id='prediction-output')
      ])
@app.callback(
    Output('raw-data-graph', 'figure'),
    Input('chart-type-selector', 'value')
)
def update_chart_type(selected_type):
    if selected_type == 'line':
        return px.line(df_melted, x='Date', y='value', color='variable',
                     title='Multivariate data visualization')
    elif selected_type == 'bar':
        return px.bar(df_melted, x='Date', y='value', color='variable',
                    title='Multivariate data visualization')
    elif selected_type == 'scatter':
        return px.scatter(df_melted, x='Date', y='value', color='variable',
                        title='Multivariate data visualization')
@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('metrics-table', 'children')],
    [Input('model-selector', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_forecast(selected_models, start_date, end_date):
    # Filter the data frames to include data from the selected date range
    filtered_df_results = df_results[(df_results['Date'] >= start_date) & (df_results['Date'] <= end_date)]

    # 处理预测图表
    if 'All' in selected_models:
        display_columns = filtered_df_results.columns[1:]
    else:
        display_columns = ['South Tower (kWh)'] + selected_models

    fig = px.line(filtered_df_results,
                  x=filtered_df_results.columns[0],
                  y=display_columns,
                  title='Comparison of predictions')

    # Calculate the error metric for the selected date range
    y2_filtered = df_real[(df_real['Date'] >= start_date) & (df_real['Date'] <= end_date)]['South Tower (kWh)'].values
    y2_pred_RF_filtered = filtered_df_results['RandomForest'].values
    y2_pred_LR_filtered = filtered_df_results['LR'].values

    MAE_RF_filtered = metrics.mean_absolute_error(y2_filtered, y2_pred_RF_filtered)
    MBE_RF_filtered = np.mean(y2_filtered - y2_pred_RF_filtered)
    MSE_RF_filtered = metrics.mean_squared_error(y2_filtered, y2_pred_RF_filtered)
    RMSE_RF_filtered = np.sqrt(metrics.mean_squared_error(y2_filtered, y2_pred_RF_filtered))
    cvRMSE_RF_filtered = RMSE_RF_filtered / np.mean(y2_filtered)
    NMBE_RF_filtered = MBE_RF_filtered / np.mean(y2_filtered)

    MAE_LR_filtered = metrics.mean_absolute_error(y2_filtered, y2_pred_LR_filtered)
    MBE_LR_filtered = np.mean(y2_filtered - y2_pred_LR_filtered)
    MSE_LR_filtered = metrics.mean_squared_error(y2_filtered, y2_pred_LR_filtered)
    RMSE_LR_filtered = np.sqrt(metrics.mean_squared_error(y2_filtered, y2_pred_LR_filtered))
    cvRMSE_LR_filtered = RMSE_LR_filtered / np.mean(y2_filtered)
    NMBE_LR_filtered = MBE_LR_filtered / np.mean(y2_filtered)

    d_filtered = {
        'Methods': ['RandomForest', 'LR'],
        'MAE': [MAE_RF_filtered, MAE_LR_filtered],
        'MBE': [MBE_RF_filtered, MBE_LR_filtered],
        'MSE': [MSE_RF_filtered, MSE_LR_filtered],
        'RMSE': [RMSE_RF_filtered, RMSE_LR_filtered],
        'cvRMSE': [cvRMSE_RF_filtered, cvRMSE_LR_filtered],
        'NMBE': [NMBE_RF_filtered, NMBE_LR_filtered]
    }
    df_metrics_filtered = pd.DataFrame(data=d_filtered)

    # Work with the indicator table
    if 'All' in selected_models:
        filtered_df = df_metrics_filtered
    else:
        filtered_df = df_metrics_filtered[df_metrics_filtered['Methods'].isin(selected_models)]

    return fig, generate_table(filtered_df)
     # Add a download link to the Forecast page layout
@app.callback(
           Output('download-link', 'href'),
           [Input('model-selector', 'value'),
            Input('date-picker-range', 'start_date'),
            Input('date-picker-range', 'end_date')]
       )
def update_download_link(selected_models, start_date, end_date):
           filtered_df_results = df_results[(df_results['Date'] >= start_date) & (df_results['Date'] <= end_date)]
           if 'All' in selected_models:
               display_columns = filtered_df_results.columns[1:]
           else:
               display_columns = ['South Tower (kWh)'] + selected_models

           csv_string = filtered_df_results.to_csv(index=False, encoding='utf-8')
           csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
           return csv_string
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('lag1-input', 'value'),
     State('rolling24h-input', 'value'),
     State('tempC-input', 'value'),
     State('hour-input', 'value'),
     State('lag24-input', 'value'),
     State('rolling3h-input', 'value'),
     State('weekday-input', 'value')]
)
def update_prediction(n_clicks, lag1, rolling24h, tempC, hour, lag24, rolling3h, weekday):
    if n_clicks > 0:
        # Convert user input to the format required by the model
        input_data = np.array([[lag1, rolling24h, tempC, hour, lag24, rolling3h, weekday]])

        # Predictions are made using the loaded model
        lr_prediction = LR_model2.predict(input_data)[0]
        rf_prediction = RF_model2.predict(input_data)[0]

        return html.Div([
    html.P(f'Linear Regression Prediction: {lr_prediction:.3f} kW',
           style={'fontSize': '20px', 'fontWeight': 'bold'}),
    html.P(f'Random Forest Prediction: {rf_prediction:.3f} kW',
           style={'fontSize': '20px', 'fontWeight': 'bold'})
             ])

    else:
        return ''



import threading
import time

def open_browser():
    time.sleep(2)  # Wait for the app to start
    import webbrowser
    webbrowser.open('http://127.0.0.1:8050')

if __name__ == '__main__':
    threading.Thread(target=open_browser).start()  # Open the browser asynchronously
    app.run_server(debug=True, use_reloader=False)  # Launch the Dash app


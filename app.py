import base64
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet import Prophet
from flask import Flask, render_template, request, jsonify, Response, json, send_file
from flask_cors import CORS, cross_origin
import numpy as np  
import pandas as pd
from pandas import Series
from flask_mysqldb import MySQL
import sys
import matplotlib.pyplot as plt
import joblib
from fbprophet.plot import plot_cross_validation_metric

pd.Series(dtype='m8[ns]')
pd.Series(dtype=np.timedelta64(0, 'ns').dtype)

app = Flask(__name__)

CORS(app, resources={r"*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# MySQL configurations
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'prophet'
app.config['MYSQL_HOST'] = '127.0.0.1'

mysql = MySQL(app)

# Add a single endpoint that we can use for testing
@app.route('/prophet', methods=['GET'])
def getData():
    
    df = pd.read_sql("SELECT LCLid,tstp,energy FROM pzem", mysql.connection)

    df = df.set_index("tstp")
    df.index = df.index.astype("datetime64[ns]")

    # set energy consumption data to float type
    df = df[df["energy"] != "Null"]
    df["energy"] = df["energy"].astype("float64")

    # Choose only 1 house by LCLid "NF0001"
    df = df[df["LCLid"] == "NF0001" ]

    # plot energy consumption data with dataframe module
    df.plot(y="energy", figsize=(15, 4))
    plt.savefig('./img/energy_consumption.png')

    train_size = int(0.8 * len(df))
    X_train, X_test = df[:train_size].index, df[train_size:].index
    y_train, y_test = df[:train_size]["energy"].values, df[train_size:]["energy"].values

    train_df = pd.concat([pd.Series(X_train), pd.Series(y_train)], axis=1, keys=["ds", "y"])
    test_df = pd.concat([pd.Series(X_test), pd.Series([0]*len(y_test))], axis=1, keys=["ds", "y"])
    answer_df = pd.concat([pd.Series(X_test), pd.Series(y_test)], axis=1, keys=["ds", "y"])

    model = Prophet()
    model.fit(train_df)

    model = joblib.load("models/finalized_model.sav", 'r')
    predict = model.predict(test_df)

    model.plot(predict)
    plt.savefig('./img/forecast.png')

    model.plot_components(predict)
    plt.savefig('./img/forecast_component.png')

    # Analysis with cross validation method
    # This cell takes some minutes.
    # cutoffs = pd.to_datetime(['2022-06-12', '2022-07-12'])
    df_cv = cross_validation(model, horizon="1 days")
    df_cv.head()

    # # # With performance_metrics, we can visualize the score
    df_p = performance_metrics(df_cv)
    df_p
    plot4 = plot_cross_validation_metric(df_cv, metric='mape')
    plot4.savefig('./img/mape.png')
    # plt.savefig('./img/final.png')

    plt.figure(figsize=(12, 4))
    plt.plot(answer_df['ds'], answer_df['y'])
    plt.plot(predict['ds'], predict['yhat'])
    plt.savefig('./img/final.png')

    response = {"response": "OK"}
    return jsonify(response)

@app.route('/get_image_energy_consumption', methods=['GET'])
def get_image1():
    return send_file('./img/energy_consumption.png', mimetype='image/gif')

@app.route('/get_image_forecast', methods=['GET'])
def get_image2():
    return send_file('./img/forecast.png', mimetype='image/gif')

@app.route('/get_image_forecast_component', methods=['GET'])
def get_image3():
    return send_file('./img/forecast_component.png', mimetype='image/gif')

@app.route('/get_image_forecast_final', methods=['GET'])
def get_image4():
    return send_file('./img/final.png', mimetype='image/gif')

@app.route('/get_image_mape', methods=['GET'])
def get_image5():
    return send_file('./img/mape.png', mimetype='image/gif')

# When run from command line, start the server
if __name__ == '__main__':
    app.run(debug=True)

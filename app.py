from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis
#import predict_exited_from_saved_model
import json
import os
from diagnostics import dataframe_summary, execution_time, na_data, model_predictions
from scoring import score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS', 'GET'])
def prediction():
    dataset_path = request.args.get('filename')
    X = pd.read_csv(dataset_path)
    X = X.drop('exited', axis = 1)
    X = X.drop('corporation', axis = 1)
    preds = model_predictions(X)
    return str(preds)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    score = score_model()
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    summary_stats = dataframe_summary()
    return str(summary_stats)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    diagnostic_data = na_data()
    diagnostic_data.append(execution_time())
    return str(diagnostic_data)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

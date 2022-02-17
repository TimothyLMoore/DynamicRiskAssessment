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
model_path = os.path.join(config['prod_deployment_path'])

with open(os.path.join(os.getcwd(), model_path, "trainedmodel.pkl"), 'rb') as file:
        model = pickle.load(file)
test_data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS', 'GET'])
def prediction():
    dataset_path = request.args.get('filename')
    print(dataset_path)
    X = pd.read_csv(os.path.join(os.getcwd(),dataset_path))
    X = X.drop('exited', axis = 1)
    X = X.drop('corporation', axis = 1)
    preds = model_predictions(X)
    return jsonify({"predictions": preds})

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    return jsonify({"f1_score": score_model(model, test_data)})

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    return jsonify({"stats": dataframe_summary()})

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    return jsonify({"execution time": execution_time(),
        "missing_data": na_data(),})

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)

from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    with open(os.path.join(os.getcwd(), model_path, "trainedmodel.pkl"), 'rb') as file:
        model = pickle.load(file)

    with open(os.path.join(os.getcwd(), model_path, "latestscore.txt"), "r") as r:
        latestscore = r.read()

    with open(os.path.join(os.getcwd(), dataset_csv_path, "ingestedfiles.txt"), "r") as r:
        ingestedfiles = r.read().splitlines()

    path = os.path.join(os.getcwd(), prod_deployment_path)
    os.makedirs(path, exist_ok=True)

    pickle.dump(model, open(os.path.join(os.getcwd(), prod_deployment_path, "trainedmodel.pkl"), 'wb'))

    with open(os.path.join(os.getcwd(), prod_deployment_path, "latestscore.txt"), 'w') as f:
        f.write(str(latestscore))


    with open(os.path.join(os.getcwd(), prod_deployment_path, "ingestedfiles.txt"), 'w') as f:
        f.write(str(ingestedfiles))

if __name__ == '__main__':
    store_model_into_pickle()

        
        
        


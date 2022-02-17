
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
prod_path = os.path.join(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(X):
    with open(os.path.join(os.getcwd(), prod_path, "trainedmodel.pkl"), 'rb') as file:
        model = pickle.load(file)

    predicted=model.predict(X)

    return predicted.tolist()

##################Function to get summary statistics
def dataframe_summary():
    X = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))
    X = X.drop('corporation', axis = 1)
    X = X.drop('exited', axis = 1)
    summaryStats= { }
    for (columnName, columnData) in X.iteritems():
        summaryStats[columnName+"_mean"] = columnData.mean()
        summaryStats[columnName+"_median"] = columnData.median()
        summaryStats[columnName+"_std.dev."] = columnData.std()



    return summaryStats


##################Function to get NA items
def na_data():
    X = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "finaldata.csv"))

    percent_na = []
    for col, col_data in X.iteritems():
        percent_na.append(col_data.isna().sum()/len(col_data))
    return percent_na


##################Function to get timings
def execution_time():
    times = []

    starttime = timeit.default_timer()
    os.system('python ingestion.py')
    timing=timeit.default_timer() - starttime
    times.append(timing)

    starttime = timeit.default_timer()
    os.system('python training.py')
    timing=timeit.default_timer() - starttime
    times.append(timing)

    return times

##################Function to check dependencies
def outdated_packages_list():
    os.system('pip list > pip_list.txt')
    df = pd.read_csv('pip_list.txt', sep=r'\s+', skiprows=[1])
    os.remove('pip_list.txt')

    latest_list = []
    for i in df['Package']:
        try:
            temp = (subprocess.check_output(['pip-versions', 'latest', i]))
            latest_list.append(temp[:-2].decode('utf-8'))
        except:
            latest_list.append("N/A")
    df['Latest'] = latest_list
    return df

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(os.getcwd(), test_data_path, "testdata.csv"))
    df = df.drop('corporation', axis = 1)
    df = df.drop('exited', axis = 1)
    print(model_predictions(df))
    print(dataframe_summary())
    print(na_data())
    print(execution_time())
    print(outdated_packages_list())





    

import training
from scoring import score_model
import deployment
import diagnostics
import reporting
import json
import os
import sys
from ingestion import merge_multiple_dataframe
from datetime import datetime
import pandas as pd
import pickle

dateTimeObj=datetime.now()
thetimenow=str(dateTimeObj.year)+ '/'+str(dateTimeObj.month)+ '/'+str(dateTimeObj.day)

with open('config.json','r') as f:
    config = json.load(f)

prod_path = config["prod_deployment_path"]
source_path = config['input_folder_path']
output_data_path = config["output_folder_path"]

##################Check and read new data
#first, read ingestedfiles.txt
with open(os.path.join(os.getcwd(), prod_path, 'ingestedfiles.txt')) as f:
    ingested = f.read().splitlines()
ingestedfiles = []
ingested_split = []
for i in ingested:
    ingested_split.append(i.split(", "))

for i in ingested_split:
    ingestedfiles.append(i[1].replace("'",""))
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
source_files = os.listdir(os.getcwd()+'\\'+source_path)
has_new_files = False
for i in source_files:
    if i in ingestedfiles:
        pass
    else:
        has_new_files = True
        break


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if has_new_files:
    print("New files to ingest")
    os.system('python ingestion.py')
else:
    print("There are no new files")
    exit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(os.getcwd(), prod_path, "latestscore.txt"), "r") as f:
    prev_f1 = float(f.read())

new_data = pd.read_csv(os.path.join(os.getcwd(), output_data_path, "finaldata.csv"))
with open(os.path.join(os.getcwd(), prod_path, "trainedmodel.pkl"), 'rb') as file:
        model = pickle.load(file)

new_f1 = score_model(model, new_data)

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_f1 > prev_f1:
    print("There is no model drift")
    exit()
else:
    print("There is model drift")


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
os.system('python training.py')
os.system('python scoring.py')
os.system('python deployment.py')
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system('python apicalls.py')
os.system('python reporting.py')








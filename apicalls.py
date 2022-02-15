import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://10.50.3.89:8000/"

path = ("C:/Users/tmoore/PycharmProjects/DynamicRiskAssessment/testdata/testdata.csv")

#Call each API endpoint and store the responses
response1 = subprocess.run(['curl', 'http://10.50.3.89:8000/prediction?filename='+path],capture_output=True).stdout
response2 = subprocess.run(['curl', 'http://10.50.3.89:8000/scoring'],capture_output=True).stdout
response3 = subprocess.run(['curl', 'http://10.50.3.89:8000/summarystats'],capture_output=True).stdout
response4 = subprocess.run(['curl', 'http://10.50.3.89:8000/diagnostics'],capture_output=True).stdout

#combine all API responses
responses = [response1, response2, response3, response4]
#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

MyFile=open(os.getcwd()+'\\'+model_path+'\\'+'apireturns.txt','w+')

for element in responses:
     MyFile.write(str(element))
MyFile.close()




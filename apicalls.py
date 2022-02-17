import subprocess
from subprocess import DEVNULL, STDOUT, check_call
import requests
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://10.50.3.89:8000"

path = ("testdata/testdata.csv")

#Call each API endpoint and store the responses
response1 = requests.post(URL+'/prediction?filename='+path)
response2 = requests.get(URL+'/scoring')
response3 = requests.get(URL+'/summarystats')
response4 = requests.get(URL+'/diagnostics')

#combine all API responses
#print(response1)
#print(response2)
#print(response3)
#print(response4)

responses = [response1.json(), response2.json(), response3.json(), response4.json()]
#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

MyFile=open(os.getcwd()+'\\'+model_path+'\\'+'apireturns2.txt','w+')

for element in responses:
     MyFile.write(str(element))
MyFile.close()




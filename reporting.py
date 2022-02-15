import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['test_data_path'])
output_path = os.path.join(config['output_model_path'])




##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    preds = model_predictions()
    X = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, "testdata.csv"))
    y = X.pop('exited')
    cf_matrix = metrics.confusion_matrix(y, preds)


    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues')

    ax.set_title('Dynamic Risk Assessment\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');


    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])


    fig=ax.get_figure()
    fig.savefig(os.path.join(os.getcwd(), output_path, "confusionmatrix.png"))





if __name__ == '__main__':
    score_model()

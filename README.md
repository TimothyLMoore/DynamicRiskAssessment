# Dynamic Risk Assesment

- Project **Dynamic Risk Assesment** of ML DevOps Engineer Nanodegree Udacity

## Project Description

Using the starting code provided, we created an automated pipeline to read in new data and update the model based on the data is model decay exists

## Running Files

> git clone https://github.com/TimothyLMoore/DynamicRiskAssessment

> cd DynamicRiskAssessment

> conda env create -n [env name] -f requirments.txt

> conda activate [env name]

python fullprocess.py

This will run the entire system, there is also a CRON file that will check the run the fullprocess every 10 minutes

## Model Details

Logistic Regression

Max iter = 100

tol = 0.0001

Scores:

F1 Score = 0.7647

## Submission Details

-Github Repository: https://github.com/TimothyLMoore/DynamicRiskAssessment


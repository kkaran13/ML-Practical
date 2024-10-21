# Practical 5
"""Aim: Write a program to construct a Bayesian network considering medical data. Use 
this model to demonstrate the diagnosis of heart patients using standard Heart Disease 
Data Set. """

# Import necessary libraries
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

# Load the Dataset
url = 'https://raw.githubusercontent.com/vtucs/Machine_Learning_Laboratory/master/ds4.csv'
data = pd.read_csv(url)
heart_disease = pd.DataFrame(data)

# Check the column names
print("Columns in the dataset:", heart_disease.columns)

# Define the Bayesian Network model with correct column names
model = BayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),  # Note the spelling here
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),  # Use 'cholestrol'
    ('diet', 'cholestrol')
])

# Fit the model using MLE
model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

# Create a Variable Elimination object
HeartDisease_infer = VariableElimination(model)

# Instructions for entering values
print('For Age enter: SuperSeniorCitizen: 0, SeniorCitizen: 1, MiddleAged: 2, Youth: 3, Teen: 4')
print('For Gender enter: Male: 0, Female: 1')
print('For Family History enter: Yes: 1, No: 0')
print('For Diet enter: High: 0, Medium: 1, Low: 2')
print('For Lifestyle enter: Athlete: 0, Active: 1, Moderate: 2, Sedentary: 3')
print('For Cholesterol enter: High: 0, Borderline: 1, Normal: 2')

# Perform heart disease diagnosis based on user-provided attributes
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
    'age': int(input('Enter Age: ')),
    'Gender': int(input('Enter Gender: ')),
    'Family': int(input('Enter Family History: ')),
    'diet': int(input('Enter Diet: ')),
    'Lifestyle': int(input('Enter Lifestyle: ')),
    'cholestrol': int(input('Enter Cholesterol: '))  # Use 'cholestrol'
})

# Print the results
print(q)


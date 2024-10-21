# Practicle 3
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset from the provided URL
url = 'https://raw.githubusercontent.com/M-Arashi/DataSets/main/playgolf.csv'
playgolf_data = pd.read_csv(url)

# Display the first few rows of the dataset
print(playgolf_data.head())

# Assume the last column is the target variable and the rest are features
X = playgolf_data.iloc[:, :-1].values
y = playgolf_data.iloc[:, -1].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Encode categorical features
encoder = OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Create a Categorical Naive Bayes classifier
classifier = CategoricalNB()

# Train the model
classifier.fit(X_train_encoded, y_train)

# Make predictions
y_pred = classifier.predict(X_test_encoded)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)

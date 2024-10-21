# Task-1: Import Libraries, preprocess the training dataset and select required features.

# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Import the dataset
url = 'https://raw.githubusercontent.com/rcdbe/bigdatacertification/master/dataset/churn_trasnsformed_new.csv'
df_csv = pd.read_csv(url, sep=',')
print(df_csv.head())

# Step 3: Remove "Unnamed: 0" Column
df = df_csv.drop("Unnamed: 0", axis=1)
print(df.head())
print(df.info())

# Step 4: Normalize features using MinMaxScaler
mm_scaler = MinMaxScaler()
column_names = df.columns.tolist()
column_names.remove('Churn')
df[column_names] = mm_scaler.fit_transform(df[column_names])
df.sort_index(inplace=True)
print(df.head())

# Step 5: Select features and target variable
features = ['Churn']
train_feature = df.drop(features, axis=1)
train_target = df["Churn"]
print(train_feature.head(5))

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, shuffle=True, 
                                                    test_size=0.3, random_state=1)
print(X_train.head())

# Task-2: Train the ANN Model

# Step 1: Initialize the MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', solver='adam', max_iter=10000, verbose=True)

# Step 2: Fit the model
mlp.fit(X_train, y_train)

# Prediction on test dataset
y_pred_mlp = mlp.predict(X_test)

# Step 3: Print the model details
print('Number of Layers =', mlp.n_layers_)
print('Number of Iterations =', mlp.n_iter_)
print('Current loss computed with the loss function =', mlp.loss_)

# Step 4: Compute the confusion matrix
cnf_matrix_mlp = metrics.confusion_matrix(y_test, y_pred_mlp)
print("Confusion Matrix:\n", cnf_matrix_mlp)

# Step 5: Calculate metrics
acc_mlp = metrics.accuracy_score(y_test, y_pred_mlp)
prec_mlp = metrics.precision_score(y_test, y_pred_mlp)
rec_mlp = metrics.recall_score(y_test, y_pred_mlp)
f1_mlp = metrics.f1_score(y_test, y_pred_mlp)
kappa_mlp = metrics.cohen_kappa_score(y_test, y_pred_mlp)

print(f'Accuracy: {acc_mlp}')
print(f'Precision: {prec_mlp}')
print(f'Recall: {rec_mlp}')
print(f'F1 Score: {f1_mlp}')
print(f'Cohen Kappa: {kappa_mlp}')

# Task-3: Create the confusion matrix using seaborn and matplotlib

# Step 1: Create a heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cnf_matrix_mlp, annot=True, fmt='d', cmap='Blues', cbar=False)

# Step 2: Set labels, title, and axis ticks
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn'])
plt.yticks(ticks=[0, 1], labels=['No Churn', 'Churn'])
plt.title('Confusion Matrix')

# Step 3: Display the plot
plt.show()

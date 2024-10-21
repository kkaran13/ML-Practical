import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics

# # Load the Iris dataset
# iris = load_iris()
# X = pd.DataFrame(iris.data, columns=iris.feature_names)
# y = X['sepal length (cm)']  # Target: sepal length

# # Drop the target feature from X
# X = X.drop(columns=['sepal length (cm)'])

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create and fit the Linear Regression model
# linear_model = LinearRegression()
# linear_model.fit(X_train, y_train)

# # Make predictions
# y_pred = linear_model.predict(X_test)

# # Evaluate the model
# mse = metrics.mean_squared_error(y_test, y_pred)
# r2 = metrics.r2_score(y_test, y_pred)

# print("Mean Squared Error:", mse)
# print("R^2 Score:", r2)

# # Visualizing the results for one feature
# plt.scatter(X_test['sepal width (cm)'], y_test, color='blue', label='Actual')
# plt.scatter(X_test['sepal width (cm)'], y_pred, color='red', label='Predicted')
# plt.xlabel('Sepal Width (cm)')
# plt.ylabel('Sepal Length (cm)')
# plt.title('Linear Regression Predictions')
# plt.legend()
# plt.show()
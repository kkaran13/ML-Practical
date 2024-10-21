# Compare various unsupervised learning algorithms using appropriate data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load the Iris dataset
data = load_iris()
X = data.data[:, [0, 1]]  # Using Sepal Length and Sepal Width features

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define clustering algorithms
algorithms = {
    'K-Means': KMeans(n_clusters=3, random_state=42),
    'Hierarchical': AgglomerativeClustering(n_clusters=3),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'GMM': GaussianMixture(n_components=3, random_state=42)
}

# Function to visualize results
def visualize_results(name, labels):
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
    plt.title(f"{name} Clustering Results")
    plt.xlabel("Sepal Length (scaled)")
    plt.ylabel("Sepal Width (scaled)")
    plt.show()

# Apply each algorithm and visualize results
for name, algorithm in algorithms.items():
    algorithm.fit(X_scaled)
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_
    else:
        labels = algorithm.predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, labels)
    print(f"{name} Silhouette Score: {silhouette_avg:.3f}")
    
    # Visualize results
    visualize_results(name, labels)
    print("===")

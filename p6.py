# # Aim: Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same 
# # data set for clustering using k-Means algorithm.
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler

# # Step 1: Load the Dataset from GitHub
# url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv'
# data = pd.read_csv(url)
# print(data.head())

# # Step 2: Preprocessing
# # Select the 'Age' and 'Income' columns for clustering
# X = data[['Age', 'Income($)']].values  # Use the relevant columns for clustering

# # Scale the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Step 3: Clustering using EM Algorithm (GMM)
# gmm = GaussianMixture(n_components=3, random_state=42)  # Adjust n_components as needed
# gmm.fit(X_scaled)
# gmm_labels = gmm.predict(X_scaled)

# # Step 4: Clustering using k-Means
# kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters as needed
# kmeans.fit(X_scaled)
# kmeans_labels = kmeans.predict(X_scaled)

# # Step 5: Visualizing the Results
# plt.figure(figsize=(12, 5))

# # GMM Clustering
# plt.subplot(1, 2, 1)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
# plt.title('GMM Clustering')
# plt.xlabel('Scaled Age')
# plt.ylabel('Scaled Income')

# # k-Means Clustering
# plt.subplot(1, 2, 2)
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
# plt.title('k-Means Clustering')
# plt.xlabel('Scaled Age')
# plt.ylabel('Scaled Income')

# plt.tight_layout()
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Step 1: Load the Income Dataset from the URL
url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv'
df = pd.read_csv(url)

# Step 2: Visualize Income Data
plt.scatter(df['Age'], df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.title('Age vs Income')
plt.show()

# Step 3: K-Means Clustering on Income Data
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])
df['cluster'] = y_predicted

# Step 4: Plotting K-Means Clustering Results
plt.figure(figsize=(8, 6))
plt.scatter(df[df.cluster == 0]['Age'], df[df.cluster == 0]['Income($)'], color="green", label='Cluster 0')
plt.scatter(df[df.cluster == 1]['Age'], df[df.cluster == 1]['Income($)'], color='red', label='Cluster 1')
plt.scatter(df[df.cluster == 2]['Age'], df[df.cluster == 2]['Income($)'], color="black", label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroids')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend()
plt.title('K-Means Clustering on Income Data')
plt.show()

# Step 5: Scaling Data for Iris Dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['True Label'] = iris.target

# Scaling the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.iloc[:, :-1])  # Exclude the true label for scaling

# Step 6: K-Means Clustering on Iris Dataset
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(data_scaled)
data['KMeans_Cluster'] = kmeans.labels_

# Step 7: Gaussian Mixture Model Clustering on Iris Dataset
gmm = GaussianMixture(n_components=num_clusters, random_state=42)
gmm.fit(data_scaled)
data['EM_Cluster'] = gmm.predict(data_scaled)

# Step 8: Print Cluster Centers
print("K-Means Cluster Centers:\n", kmeans.cluster_centers_)
print("EM Means:\n", gmm.means_)
print("EM Covariances:\n", gmm.covariances_)

# Step 9: Plot K-Means and EM Clusters
plt.figure(figsize=(14, 6))

# Plot K-Means clusters
plt.subplot(1, 2, 1)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data['KMeans_Cluster'], cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')

# Plot EM clusters
plt.subplot(1, 2, 2)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=data['EM_Cluster'], cmap='plasma')
plt.title('EM (Gaussian Mixture) Clustering')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')

plt.show()

# Step 10: Evaluate Clustering Performance
ari_kmeans = adjusted_rand_score(data['True Label'], data['KMeans_Cluster'])
print("ARI for K-Means: ", ari_kmeans)

ari_em = adjusted_rand_score(data['True Label'], data['EM_Cluster'])
print("ARI for EM: ", ari_em)

nmi_kmeans = normalized_mutual_info_score(data['True Label'], data['KMeans_Cluster'])
print("NMI for K-Means: ", nmi_kmeans)

nmi_em = normalized_mutual_info_score(data['True Label'], data['EM_Cluster'])
print("NMI for EM: ", nmi_em)

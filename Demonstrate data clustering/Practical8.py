import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Applying K-Means clustering with K=3 (as there are three classes in Iris dataset)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster centers and labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualizing the clusters (considering only the first two features for simplicity)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8, edgecolors='w')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='o', label='Centroids')
plt.title('K-Means Clustering of Iris Dataset (First two features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

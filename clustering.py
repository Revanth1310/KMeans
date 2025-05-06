import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load dataset
iris = load_iris()
X = iris.data

# Optional: reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 2. Elbow Method to choose K
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 3. Fit KMeans with optimal K (say, K=3 from Elbow)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 4. Visualize Clusters (2D PCA)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.title('K-Means Clustering (PCA-reduced Data)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 5. Evaluate using Silhouette Score
score = silhouette_score(X, labels)
print("Silhouette Score:", score)

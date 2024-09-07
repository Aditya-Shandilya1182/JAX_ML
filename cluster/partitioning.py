import jax.numpy as jnp
from jax import random

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def initialize_centroids(self, X):
        key = random.PRNGKey(self.random_state) if self.random_state else random.PRNGKey(0)
        indices = random.choice(key, X.shape[0], shape=(self.n_clusters,), replace=False)
        return X[indices]

    def assign_clusters(self, X, centroids):
        def distance(point, centroid):
            return jnp.sum((point - centroid) ** 2)
        
        def assign(point):
            distances = jnp.array([distance(point, c) for c in centroids])
            return jnp.argmin(distances)
        
        return jnp.array([assign(point) for point in X])

    def update_centroids(self, X, labels):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if cluster_points.shape[0] == 0:
                new_centroids.append(jnp.zeros(X.shape[1]))
            else:
                new_centroids.append(jnp.mean(cluster_points, axis=0))
        return jnp.array(new_centroids)

    def fit(self, X):
        centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            labels = self.assign_clusters(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            if jnp.allclose(new_centroids, centroids, atol=self.tol):
                break
            centroids = new_centroids
        self.centroids = centroids

    def predict(self, X):
        return self.assign_clusters(X, self.centroids)

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

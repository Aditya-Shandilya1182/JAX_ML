import jax.numpy as jnp

class AgglomerativeClustering:
    
    def __init__(self, n_clusters=2, linkage='single', metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def euclidean_dist(self, X):
        return jnp.sqrt(jnp.sum((X[:, None] - X[None, :]) ** 2, axis=-1))
    
    def linkage_dist(self, cluster1, cluster2, distances):
        cluster1 = jnp.array(cluster1)
        cluster2 = jnp.array(cluster2)

        if self.linkage == 'single':
            return jnp.min(distances[cluster1][:, cluster2])
        elif self.linkage == 'complete':
            return jnp.max(distances[cluster1][:, cluster2])
        elif self.linkage == 'average':
            return jnp.mean(distances[cluster1][:, cluster2])
        else:
            raise ValueError(f"Unknown linkage: {self.linkage}")
        
    def find_closest_clusters(self, clusters, distances):
        min_distance = jnp.inf
        pair = (-1, -1)
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters):
                if i != j:
                    dist = self.linkage_dist(cluster1, cluster2, distances)
                    if dist < min_distance:
                        min_distance = dist
                        pair = (i, j)
        return pair

    def fit(self, X):
        distances = self.euclidean_dist(X)

        clusters = [[i] for i in range(len(X))]

        while len(clusters) > self.n_clusters:
            i, j = self.find_closest_clusters(clusters, distances)

            clusters[i].extend(clusters[j])
            clusters.pop(j)

        labels = jnp.zeros(len(X), dtype=int)
        for idx, cluster in enumerate(clusters):
            cluster = jnp.array(cluster)  
            labels = labels.at[cluster].set(idx)
        
        self.labels_ = labels

        return self


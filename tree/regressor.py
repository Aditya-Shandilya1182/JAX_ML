import jax.numpy as jnp

class DecisionTreeRegressor:
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        mean = jnp.mean(y)
        return jnp.mean((y - mean) ** 2)

    def split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def best_split(self, X, y):
        m, n = X.shape
        best_mse = float('inf')
        best_split = None

        for feature_index in range(n):
            thresholds = jnp.unique(X[:, feature_index]) 
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse_left = self.mse(y_left)
                mse_right = self.mse(y_right)
                mse_total = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)

                if mse_total < best_mse:
                    best_mse = mse_total
                    best_split = (feature_index, threshold)

        return best_split

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape

        if depth >= self.max_depth or num_samples < self.min_samples_split:
            leaf_value = jnp.mean(y)
            return leaf_value

        best_split = self.best_split(X, y)
        if not best_split:
            leaf_value = jnp.mean(y)
            return leaf_value

        feature_index, threshold = best_split
        X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)

        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)

        return {"feature_index": feature_index, "threshold": threshold, "left": left_subtree, "right": right_subtree}

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)
        return self

    def predict_sample(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        feature_index = tree["feature_index"]
        threshold = tree["threshold"]

        if x[feature_index] <= threshold:
            return self.predict_sample(x, tree["left"])
        else:
            return self.predict_sample(x, tree["right"])

    def predict(self, X):
        return jnp.array([self.predict_sample(x, self.tree) for x in X])



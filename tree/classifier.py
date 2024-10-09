import jax.numpy as jnp
from collections import Counter

class DecisionTreeClassifier():

    def __init__(self, max_depth = 10, min_samples_split = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def gini_idx(self, y):
        m = len(y)
        counts = Counter(y.tolist())
        gini = 1.0 - sum((count / m) ** 2 for count in counts.values())
        return gini
    
    def split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
    
    def best_split(self, X, y):
        m, n = X.shape
        best_gini = float('inf')
        best_split = None

        for feature_index in range(n):
            thresholds = jnp.unique(X[:, feature_index])  
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split(X, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini_idx(y_left)
                gini_right = self.gini_idx(y_right)
                gini_total = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if gini_total < best_gini:
                    best_gini = gini_total
                    best_split = (feature_index, threshold)

        return best_split
    
    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(jnp.unique(y))

        if depth >= self.max_depth or num_classes == 1 or num_samples < self.min_samples_split:
            leaf_value = Counter(y.tolist()).most_common(1)[0][0]
            return leaf_value

        best_split = self.best_split(X, y)
        if not best_split:
            leaf_value = Counter(y).most_common(1)[0][0]
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
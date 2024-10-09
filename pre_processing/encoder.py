import numpy as np

class NumericalEncoder:
    
    def __init__(self):
        self.mapping = {}
    
    def fit(self, X):
        unique_values = list(set(X))
        self.mapping = {value: idx for idx, value in enumerate(unique_values)}
        return self
    
    def transform(self, X):
        return np.array([self.mapping[value] for value in X])
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class OneHotEncoder:
    
    def __init__(self):
        self.categories_ = None
    
    def fit(self, X):
        self.categories_ = np.unique(X)
        return self
    
    def transform(self, X):
        one_hot_encoded = np.zeros((len(X), len(self.categories_)))
        for i, value in enumerate(X):
            category_index = np.where(self.categories_ == value)[0][0]
            one_hot_encoded[i, category_index] = 1
        return one_hot_encoded
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)



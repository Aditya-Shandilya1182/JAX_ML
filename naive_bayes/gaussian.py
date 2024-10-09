import jax.numpy as jnp
from jax.scipy.stats import norm

class GaussianNaiveBayes:
    
    def __init__(self):
        self.classes = None
        self.class_prior_ = None
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y):
        self.classes = jnp.unique(y) 
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.mean_ = jnp.zeros((n_classes, n_features))
        self.var_ = jnp.zeros((n_classes, n_features))
        self.class_prior_ = jnp.zeros(n_classes)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean_ = self.mean_.at[idx, :].set(jnp.mean(X_c, axis=0))
            self.var_ = self.var_.at[idx, :].set(jnp.var(X_c, axis=0))
            self.class_prior_ = self.class_prior_.at[idx].set(X_c.shape[0] / float(n_samples))

        return self

    def calculate_likelihood(self, X, class_idx):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        return norm.logpdf(X, mean, jnp.sqrt(var)).sum(axis=1)

    def calculate_posterior(self, X):
        posteriors = []

        for idx, _ in enumerate(self.classes):
            prior = jnp.log(self.class_prior_[idx])
            likelihood = self.calculate_likelihood(X, idx)
            posterior = prior + likelihood
            posteriors.append(posterior)

        return jnp.stack(posteriors, axis=1)

    def predict(self, X):
        posteriors = self.calculate_posterior(X)
        return self.classes[jnp.argmax(posteriors, axis=1)]

    def predict_proba(self, X):
        posteriors = self.calculate_posterior(X)
        exp_posteriors = jnp.exp(posteriors)
        return exp_posteriors / exp_posteriors.sum(axis=1, keepdims=True)




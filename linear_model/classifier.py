import jax
import jax.numpy as jnp
from jax import grad

class LogisticRegression:
    
    def __init__(self, input_dim, learning_rate=0.01, seed=42):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.weights = jnp.zeros(input_dim)
        self.bias = jnp.array(0.0)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + jnp.exp(-z))
    
    def predict_probability(self, X):
        z = jnp.dot(X, self.weights) + self.bias  
        return self.sigmoid(z)
    
    def predict(self, X):
        probability = self.predict_probability(X)
        return jnp.where(probability >= 0.5, 1, 0)
    
    def loss(self, X, y):
        predictions = self.predict_probability(X)
        return -jnp.mean(y * jnp.log(predictions) + (1 - y) * jnp.log(1 - predictions))
    
    def update_weights(self, X, y):
        loss_fn = lambda weights: self.loss(X, y)
        grads = grad(loss_fn)(self.weights)  
        self.weights -= self.learning_rate * grads
        self.bias -= self.learning_rate * grad(lambda b: self.loss(X, y))(self.bias)

    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.update_weights(X, y)
            if epoch % 10 == 0:
                current_loss = self.loss(X, y)
                print(f"Epoch {epoch}, Loss: {current_loss}")

    

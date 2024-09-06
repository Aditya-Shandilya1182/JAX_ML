import jax.numpy as jnp
from jax import grad, jit, value_and_grad

class LinearRegression:
    def __init__(self, input_dim, learning_rate=0.01, seed=42):
        self.learning_rate = learning_rate
        self.weights = jnp.zeros(input_dim)
        self.bias = jnp.array(0.0)

    def predict(self, X, weights, bias):
        return jnp.dot(X, weights) + bias

    def loss(self, params, X, y):
        weights, bias = params
        predictions = self.predict(X, weights, bias)
        return jnp.mean((predictions - y) ** 2)

    def update(self, params, X, y):
        loss_fn = lambda p: self.loss(p, X, y)
        grads = grad(loss_fn)(params)
        weights, bias = params
        new_weights = weights - self.learning_rate * grads[0]
        new_bias = bias - self.learning_rate * grads[1]
        return new_weights, new_bias

    def fit(self, X, y, epochs): 
        params = (self.weights, self.bias)
        for epoch in range(epochs):
            params = self.update(params, X, y)
            if epoch % 100 == 0:
                loss_value = self.loss(params, X, y)
                print(f"Epoch: {epoch}, Loss: {loss_value}")
        self.weights, self.bias = params


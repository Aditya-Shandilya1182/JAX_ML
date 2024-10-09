import jax.numpy as jnp

def accuracy_score(y_true, y_pred):
    return jnp.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = jnp.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def recall_score(y_true, y_pred):
    true_positives = jnp.sum((y_pred == 1) & (y_true == 1))
    actual_positives = jnp.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

def mean_squared_error(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    total_variance = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    explained_variance = jnp.sum((y_true - y_pred) ** 2)
    return 1 - (explained_variance / total_variance)

def confusion_matrix(y_true, y_pred):
    true_positive = jnp.sum((y_true == 1) & (y_pred == 1))
    true_negative = jnp.sum((y_true == 0) & (y_pred == 0))
    false_positive = jnp.sum((y_true == 0) & (y_pred == 1))
    false_negative = jnp.sum((y_true == 1) & (y_pred == 0))
    return jnp.array([[true_positive, false_positive], [false_negative, true_negative]])


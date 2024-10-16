import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return jnp.tanh(x)

def tanh_derivative(x):
    return 1 - jnp.tanh(x)**2

def relu(x):
    return jnp.maximum(0, x)

def relu_derivative(x):
    return jnp.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return jnp.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return jnp.where(x > 0, 1, alpha)

def parametric_relu(x, alpha=0.25):
    return jnp.where(x > 0, x, alpha * x)

def parametric_relu_derivative(x, alpha=0.25):
    return jnp.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return jnp.where(x >= 0, x, alpha * (jnp.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return jnp.where(x >= 0, 1, alpha * jnp.exp(x))

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * jnp.where(x >= 0, x, alpha * (jnp.exp(x) - 1))

def selu_derivative(x, alpha=1.67326, scale=1.0507):
    return scale * jnp.where(x >= 0, 1, alpha * jnp.exp(x))

def gelu(x):
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    sqrt_2_over_pi = jnp.sqrt(2 / jnp.pi)
    term = sqrt_2_over_pi * (x + 0.044715 * x**3)
    tanh_term = jnp.tanh(term)
    sech_squared = 1 - tanh_term**2
    return 0.5 * (1 + tanh_term) + 0.5 * x * sech_squared * (sqrt_2_over_pi * (1 + 3 * 0.044715 * x**2))

def silu(x):
    return x * sigmoid(x)

def silu_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(x):
    # numerically stable softmax
    shiftx = x -jnp.max(x, axis=0, keepdims=True)
    exps = jnp.exp(shiftx)
    return exps / jnp.sum(exps, axis=0, keepdims=True)

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

def softplus(x):
    return jnp.log1p(jnp.exp(x))  # log(1 + e^x) for numerical stability

def softplus_derivative(x):
    return sigmoid(x)

def mish(x):
    return x * jnp.tanh(softplus(x))

def mish_derivative(x):
    sp = softplus(x)
    tanh_sp = jnp.tanh(sp)
    sech_sp_sq = 1 - tanh_sp**2
    return tanh_sp + x * sech_sp_sq * sigmoid(x)
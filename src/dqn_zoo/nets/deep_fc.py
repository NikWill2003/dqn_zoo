import numpy as np
import time
from functools import wraps
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import pdb

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__!r} executed in {end-start:.6f}s")
        return result
    return wrapper

''' TODO 
- implement classes for parameters, include the param and the gradient within it
- implement the optimisers as a class
- implement hooks/ understand what they are
''' 

RNG = np.random.default_rng()

class Parameter():
    def __init__(self, weights):
        self.weights = weights # (out dim, in dim, ...)
        self.gradient = np.zeros_like(weights)
    
    def zero_grad(self):
        self.gradient.fill(0)

    @property
    def shape(self):
        return self.weights.shape
    
    @property
    def size(self):
        return self.weights.size
    
    @classmethod
    def kaiming(cls, shape):
        std = np.sqrt(2/shape[1])
        weights = RNG.normal(0, std, shape)
        return cls(weights)
    
    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype="float32"))

class Layer(ABC):
    def __init__(self, params):
        self.input = None
        self.output = None
        self.params = params 

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self):
        pass

    def clear(self):
        self.input = None 
        self.output = None

class Affine(Layer):
    def __init__(self, indim, outdim):
        self.A = Parameter.kaiming((outdim, indim))
        self.b = Parameter.zeros((outdim))
        params = [self.A, self.b]
        super().__init__(params)

    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        self.input = x # (batch, indim)
        A, b = self.A.weights, self.b.weights
        self.output = np.einsum('ij,kj->ki', A, x, optimize=True) + b
        return self.output
        
    def backward(self, out_grad):
        self.A.gradient = np.einsum('ki,kj->ij', out_grad, self.input)
        self.b.gradient = np.einsum('kj->j', out_grad)
        in_grad =  out_grad @ self.A.weights
        self.clear()
        return in_grad

class Relu(Layer):
    def __init__(self):
        super().__init__([])
    
    def __call__(self, x):
        self.forward(x)

    def forward(self, x):
        self.input = x
        # pdb.set_trace() 
        self.output = np.maximum(x,0)
        return self.output

    def backward(self, out_grad):
        in_grad = np.einsum('ki,ki->ki', out_grad, np.where(self.input>0, 1, 0))
        self.clear()
        return in_grad
    
class Sequential():
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, input):
        return self.forward(input)
    
    @property
    def params(self):
        params = [p for layer in self.layers for p in layer.params]
        return params
    
    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        g = loss_grad
        for layer in reversed(self.layers):
            g = layer.backward(g)
    
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
    
class Loss(ABC):
    def __init__(self):
        self.input = None
        self.output = None
    
    def __call__(self, x, y):
        loss = self.compute_loss(x, y)
        return loss
    
    @abstractmethod
    def compute_loss(self, x, y):
        pass
    
    @abstractmethod
    def backward(self):
        pass

class RMS(Loss):

    def compute_loss(self, x, y):
        self.input = (x, y)
        self.output = np.mean((x-y)**2)/2
        return self.output
    
    def backward(self):
        x, y = self.input
        return (x-y)/x.shape[0]

class Optimiser(ABC):
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    @abstractmethod
    def step(self):
        pass

class SGD(Optimiser):
    
    def step(self):
        for param in self.params:
            param.weights += -self.lr * param.gradient

W = RNG.random((5,5))
b = RNG.random(5)
x = RNG.random((400,5)) 
y = np.einsum('ij,ki->ki', W, x) + b

relu = Relu()
relu.forward(x)

fc = Sequential([Affine(5, 64), Relu(), Affine(64, 64),
                    Relu(), Affine(64, 5)])
ell = RMS()
optimiser = SGD(fc.params, 0.03)
losses = []

@timeit
def train():
    for epoch in range(100):
        perm = RNG.permutation(400)

        for start in range(0, 400, 20):
            batch_idx = perm[start:start+20]
            x_batch = x[batch_idx]
            y_batch = y[batch_idx]

            output = fc(x_batch)
            loss = ell.compute_loss(output, y_batch)
            fc.backward(ell.backward())
            optimiser.step()

        full_output = fc(x)
        full_loss = ell.compute_loss(full_output, y)
        losses.append(full_loss)

train()

plt.plot(losses)
plt.show()
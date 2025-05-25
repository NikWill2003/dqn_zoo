import numpy as np
import time
from functools import wraps
import matplotlib.pyplot as plt

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
-need to add in a way to do backpropogation
-need to add in the optimiser
 ''' 

RNG = np.random.default_rng()

class Layer():
    def __init__(self, indim, outdim, params, function, derivative):
        self.input = None
        self.output = None
        self.gradient = None 
        self.partial = None
        self.indim = indim
        self.outdim = outdim 
        self.params = params 
        self.function = function 
        self.derivative = derivative 

    def __call__(self, input):
        out = self.forward(input)
        self.partial = self.derivative(input)
        return out

    def forward(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def reset(self):
        self.input = None 
        self.output = None
        self.gradient = None 
        self.partial = None

    def get_params(self):
        return self.params
    
    def optimise(self):
        pass

class Affine(Layer):
    def __init__(self, indim, outdim):
        # Kaiming init 
        std = (2/indim) ** 0.5
        self.A = RNG.normal(0, std, (outdim, indim))
        self.b = RNG.normal(0, std, (outdim))
        self.A_grad = None
        self.b_grad = None
        params = [self.A, self.b]
        function = lambda x : np.einsum('ij,kj->ki', self.A, x) + self.b
        derivative = lambda x : self.A
        super().__init__(indim, outdim, params, function, derivative)
    
    def backward(self, out_grad):
        self.A_grad = np.einsum('ki,kj->ij', out_grad, self.input)
        self.b_grad = np.einsum('kj->j', out_grad)
        in_grad = np.einsum('ki,ij->kj', out_grad, self.A)
        return in_grad
    
    def optimise(self, lr=0.0001):
        self.A -= lr * self.A_grad
        self.b -= lr * self.b_grad

    def reset(self):
        super().reset()
        self.A_grad = None
        self.b_grad = None 

class Relu(Layer):
    def __init__(self, dim):
        params = []
        function = lambda x : np.maximum(x, 0)
        derivative = lambda x : np.where(x>0, 1, 0)
        super().__init__(dim, dim, params, function, derivative)

    def backward(self, out_grad):
        return np.einsum('ki,ki->ki', out_grad, self.partial)
    
class FullyConnected():
    def __init__(self, indim, outdim):
        self.seq = [Affine(indim, 64), Relu(64), Affine(64, 64),
                    Relu(64), Affine(64, outdim)]
    
    def __call__(self, input):
        return self.forward(input)
    
    @timeit
    def forward(self, input):
        x = input
        for layer in self.seq:
            x = layer(x)
        return x

    def backward(self, loss_grad):
        g = loss_grad
        for layer in reversed(self.seq):
            g = layer.backward(g)

    def optimise(self):
        for layer in self.seq:
            layer.optimise()
            layer.reset()
    
class Loss():
    def __init__(self, function, derivative):
        self.input = None
        self.output = None
        self.grad = None
        self.function = function
        self.derivative = derivative
    
    def __call__(self, x, y):
        loss = self.compute_loss(x, y)
        self.grad = self.derivative(x, y)
        return loss
    
    def compute_loss(self, x, y):
        loss = self.function(x, y)
        return loss
    
    def backward(self):
        return self.grad

class RMS(Loss):
    def __init__(self):
        function = lambda x, y : np.mean((x-y)**2)/2
        derivative = lambda x, y : (x-y)/x.shape[0]
        super().__init__(function, derivative)

# fc = FullyConnected(10, 5)
# ell = RMS()

# input = RNG.random((20,10))
# target = RNG.random((20,5))

# output = fc(input)

# loss = ell(output, target)

# fc.backward(ell.backward())

# fc.optimise()

W = RNG.random((5,5))
b = RNG.random(5)
x = RNG.random((400,5)) 
y = np.einsum('ij,ki->ki', W, x) + b

ell = RMS()
fc = FullyConnected(5,5)

losses = []

for epoch in range(100):
    perm = RNG.permutation(400)

    for start in range(0, 400, 20):
        batch_idx = perm[start:start+20]
        x_batch = x[batch_idx]
        y_batch = y[batch_idx]

        output = fc(x_batch)
        loss = ell(output, y_batch)
        fc.backward(ell.backward())
        fc.optimise()

    full_output = fc(x)
    full_loss = ell(full_output, y)
    losses.append(full_loss)

plt.plot(losses)
plt.show()
import numpy as np

RNG = np.random.default_rng()

class Layer():
    def __init__(self, indim, outdim, params, function, derivative):
        self.input = None
        self.output = None
        self.indim = indim
        self.outdim = outdim
        self.gradient = None 
        self.params = params 
        self.function = function 
        self.derivative = derivative 

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def backward(self):
        self.gradient = self.derivative(self.input)
        return self.gradient

    def reset(self):
        self.input = None 
        self.gradient = None 

    def get_params(self):
        return self.params

class Affine(Layer):
    def __init__(self, indim, outdim):
        # Kaiming init 
        std = (2/indim) ** 0.5
        A = RNG.normal(0, std, (outdim, indim))
        b = RNG.normal(0, std, (outdim))
        params = [A, b]
        function = lambda x : np.matmul(A, x) + b
        derivative = lambda x : A
        super().__init__(indim, outdim, params, function, derivative)
        

class Relu(Layer):
    def __init__(self, dim):
        params = []
        function = lambda x : np.maximum(x, 0)
        derivative = lambda x : 1 if x > 0 else 0
        super().__init__(dim, dim, params, function, derivative)

class FullyConnected():
    def __init__(self, indim, outdim):
        self.a1 = Affine(indim, 64)
        self.r1 = Relu(64)
        self.a2 = Affine(64, 64)
        self.r2 = Relu(64)
        self.a3 = Affine(64, outdim)
    
    def __call__(self, input):
        return self.forward(input)
    
    def forward(self, input):
        x = self.r1(self.a1(input))
        x = self.r2(self.a2(x))
        logits = self.a3(x)
        return logits

affine = Affine(2,3)
print(affine.get_params())
input = np.ones(2)
print(affine(input))

# input = 10* RNG.random(3) - 5
# relu = Relu(3)
# print(input)
# print(relu.forward(input))
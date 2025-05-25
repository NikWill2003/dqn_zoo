import numpy as np

RNG = np.random.default_rng()

x = RNG.random((3,5))
y = RNG.random((10,5))

z = np.einsum('ij,kj->ki', x, y)
w = y @ x.T

print(np.allclose(z, w, atol=0, rtol=1e-12))



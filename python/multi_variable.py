import numpy as np
import numpy.typing as npt

# Takes in a numpy array [x_n, ..., x_0] and computes the sum of the squares - 1
def func(params: npt.NDArray):
    return np.sum(params**2) - 1

def func2(params: npt.NDArray):
    return params[0] * params[0] * np.sin(params[1]) + params[1] * params[1] * np.cos(params[0])

# Computes the derivative of func at func(params)
def gradient(func, params: npt.NDArray, step=0.0001, delta=None):
    if delta is None:
        delta = np.full(params.shape, step, dtype=np.float64)
    if params.shape != delta.shape:
        raise Exception("Parameters and Delta have different shapes")
    # The gradient is (del f / del x_i) * e_i
    gradient = np.zeros(params.shape, dtype=np.float64)
    for i in range(len(params)):
        gradient[i] = partial_derivative(func, params, delta[i], i)
    return gradient

# Computes the Hessian (second derivative matrix) of func at func(params)
def hessian(func, params: npt.NDArray, step=0.0001, delta=None):
    if delta is None:
        delta = np.full(params.shape, step, dtype=np.float64)
    if params.shape != delta.shape:
        raise Exception("Parameters and Delta have different shapes for Hessian")
    hessian = np.zeros((params.shape[0], params.shape[0]), dtype=np.float64)
    for i in range(len(params)):
        for j in range(len(params)):
            hessian[i][j] = second_partial_derivative(func, params, delta[i], delta[j], i, j)
    return hessian

# Computes the inverse of the hessian
def inverse(hessian):
    return np.linalg.inv(hessian)

# Computes the second partial derivative (f_x)_y using central difference
def second_partial_derivative(func, params, delta_x, delta_y, x_idx, y_idx):
    e_x = np.zeros(params.shape, dtype=np.float64)
    e_y = np.zeros(params.shape, dtype=np.float64)
    e_x[x_idx] = delta_x
    e_y[y_idx] = delta_y
    
    return (func(params + e_x + e_y) - func(params + e_x) - func(params + e_y) + func(params)) / (delta_x * delta_y)

# Computes the partial derivative using central difference
def partial_derivative(func, params, delta_x_i, i):
    e_i = np.zeros(params.shape, dtype=np.float64)
    e_i[i] = delta_x_i
    return (func(params + e_i) - func(params - e_i)) / (2 * delta_x_i)

def iteration(func, vals, step=0.0000001):
    grad = gradient(func, vals, step=step)
    hess = hessian(func, vals, step=step)
    return vals - np.matmul(inverse(hess), grad)

def main():
    vals = np.asarray([5.0, -3.0], dtype=np.float64)
    prev = None
    for i in range(10000):
        output = func(vals)
        print(f"iter: {i}. vals={vals}, func(vals)={output}")
        vals = iteration(func2, vals)
        prev = output

if __name__ == '__main__':
    main()
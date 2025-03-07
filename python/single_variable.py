import numpy as np

def func(x):
    return (x**2 - 3) / (x**4 + 1) * np.exp(x) + np.sqrt(np.abs(x))

def approx_derivative(x, func, delta_x=0.00000001):
    return (func(x + delta_x) - func(x)) / delta_x

def iteration(x):
    return x - func(x) / approx_derivative(x, func)

def main():
    x = 100
    for i in range(1000):
        print(f"iter {i}: x = {x}, f(x) = {func(x)}")
        x = iteration(x)

if __name__ == '__main__':
    main()
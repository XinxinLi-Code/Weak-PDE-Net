"""Functions for use with symbolic regression.

These classes encapsulate SymPy, TensorFlow, and NumPy implementations of each
function so that the functions can be used in multiple contexts."""

import torch
# import tensorflow as tf
import numpy as np
import sympy as sp


class BaseFunction:
    """Abstract class for primitive functions."""

    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        """SymPy implementation."""
        return None

    def torch(self, x):
        """No base implementation is required."""
        return None

    def tf(self, x):
        """Automatically convert SymPy to TensorFlow."""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'tensorflow')(x)

    def np(self, x):
        """Automatically convert SymPy to NumPy."""
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)


class Constant(BaseFunction):
    def torch(self, x):
        return torch.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):
    def __init__(self):
        super(Identity, self).__init__()
        self.name = 'id'

    def torch(self, x):
        return x / self.norm  

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):
    def __init__(self):
        super(Square, self).__init__()
        self.name = 'pow2'

    def torch(self, x):
        return torch.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm


class Cube(BaseFunction):
    def __init__(self):
        super(Cube, self).__init__()
        self.name = 'pow3'

    def torch(self, x):
        return torch.pow(x, 3) / self.norm

    def sp(self, x):
        return x ** 3 / self.norm

    def np(self, x):
        return np.power(x, 3) / self.norm


class Pow(BaseFunction):
    def __init__(self, power, norm=1):
        BaseFunction.__init__(self, norm=norm)
        self.power = power
        self.name = 'pow{}'.format(int(power))

    def torch(self, x):
        return torch.pow(x, self.power) / self.norm

    def sp(self, x):
        return x ** self.power / self.norm




class Sin(BaseFunction):
    def __init__(self):
        super().__init__()
        self.name = 'sin'

    def torch(self, x):
        return torch.sin(x) / self.norm

    def sp(self, x):
        return sp.sin(x) / self.norm


class Cos(BaseFunction):
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'

    def torch(self, x):
        return torch.cos(x) / self.norm

    def sp(self, x):
        return sp.cos(x) / self.norm


class Tan(BaseFunction):
    def __init__(self):
        super(Tan, self).__init__()
        self.name = 'tan'

    def torch(self, x):
        return torch.tan(x) / self.norm

    def sp(self, x):
        return sp.tan(x) / self.norm



class Exp(BaseFunction):
    def __init__(self, theta_exp=1e2):
        super().__init__()
        self.name = 'exp'
        self.theta_exp = theta_exp

    def torch(self, x):
        # Create the result tensor.
        result = torch.zeros_like(x)
        
        # Handle entries where x != 0.
        mask_nonzero = (x != 0)
        x_nonzero = x[mask_nonzero]
        
        # Apply the original logic to nonzero values.
        if x_nonzero.numel() > 0:  # Ensure that nonzero elements are present.
            non_zero_result = torch.where(
                x_nonzero >= self.theta_exp,
                torch.exp(torch.tensor(self.theta_exp, dtype=x.dtype, device=x.device)),
                torch.exp(x_nonzero)
            ) / self.norm
            
            # Place the results back in their original positions.
            result[mask_nonzero] = non_zero_result
        return result

    def sp(self, x):
        return sp.exp(x)


class Log(BaseFunction):
    def __init__(self, theta_ln=1e-2, epsilon=1e-6):
        super().__init__()
        self.name = 'log'
        self.theta_ln = theta_ln
        self.epsilon = epsilon

    def torch(self, x):
        result = torch.where(
            x < self.theta_ln,
            torch.zeros_like(x),
            torch.log(x + self.epsilon)
        )
        return result / self.norm

    def sp(self, x):
        return sp.log(x)



class Sqrt(BaseFunction):
    def __init__(self):
        super(Sqrt, self).__init__()
        self.name = 'sqrt'

    def torch(self, x):
        return torch.sqrt(torch.abs(x)) / self.norm

    def sp(self, x):
        return sp.sqrt(sp.Abs(x)) / self.norm


class BaseFunction2:
    """Abstract class for primitive functions with two inputs."""

    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        """SymPy implementation."""
        return None

    def torch(self, x, y):
        return None

    def tf(self, x, y):
        """Automatically convert SymPy to TensorFlow."""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'tensorflow')(x, y)

    def np(self, x, y):
        """Automatically convert SymPy to NumPy."""
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)


class Product(BaseFunction2):
    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '*'

    def torch(self, x, y):
        return x * y / self.norm

    def sp(self, x, y):
        return x * y / self.norm


class Plus(BaseFunction2):
    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '+'

    def torch(self, x, y):
        return (x + y) / self.norm

    def sp(self, x, y):
        return (x + y) / self.norm


class Sub(BaseFunction2):
    def __init__(self, norm=1.0):
        super().__init__(norm=norm)
        self.name = '-'

    def torch(self, x, y):
        return (x - y) / self.norm

    def sp(self, x, y):
        return (x - y) / self.norm


class Div(BaseFunction2):
    def __init__(self):
        super(Div, self).__init__()
        self.name = '/'

    def torch(self, x, y):
        return x / (y + 1e-6)

    def sp(self, x, y):
        return x / (y + 1e-6)


def count_inputs(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction):
            i += 1
        elif isinstance(func, BaseFunction2):
            i += 2
    return i


def count_double(funcs):
    i = 0
    for func in funcs:
        if isinstance(func, BaseFunction2):
            i += 1
    return i


default_func = [
    Product(),
    Plus(),
    Sin(),
]

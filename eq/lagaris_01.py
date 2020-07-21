"""
Problem 1 in Lagaris et al (1998) (1st order ODE IVP)

Note that an upper-case 'Y' is used to represent the Greek psi from
the original equation.

The equation is defined on the domain [0,1].

The analytical form of the equation is:
    G(x, Y, dY/dx) = dY/dx + (x + (1 + 3*x**2)/(1 + x + x**3))*Y
                     - x**3 - 2*x - x**2*(1 + 3*x**2)/(1 + x + x**3) = 0

with initial condition:

Y(0) = 1

This equation has the analytical solution for the supplied initial conditions:

Ya(x) = exp(-x**2)/(1 + x + x**3) + x**2

Reference:

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis, "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999, 1998

"""


from math import exp
import numpy as np


# Specify the initial condition for this ODE: Y(0) = 1
ic = 1


def G(x, Y, dY_dx):
    """Code for differential equation"""
    return (
        dY_dx + (x + (1 + 3*x**2)/(1 + x + x**3))*Y
        - x**3 - 2*x - x**2*(1 + 3*x**2)/(1 + x + x**3)
    )

def dG_dY(x, Y, dY_dx):
    """Derivative of G wrt Y"""
    return x + (1 + 3*x**2)/(1 + x + x**3)

def dG_ddYdx(x, y, dY_dx):
    """Derivative of G wrt dy/dx"""
    return 1

def Ya(x):
    """Analytical solution"""
    return exp(-x**2/2)/(1 + x + x**3) + x**2

def dYa_dx(x):
    """Derivative of analytical solution"""
    return 2*x - exp(-x**2/2)*(1 + x + 4*x**2 + x**4)/(1 + x + x**3)**2

if __name__ == '__main__':
    assert G(0, 0, 0) == 0
    assert dG_dY(0, 0, 0) == 1
    assert dG_ddYdx(0, 0, 0) == 1
    assert np.isclose(Ya(0), ic)
    assert np.isclose(dYa_dx(0), -1)

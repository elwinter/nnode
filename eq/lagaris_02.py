"""
Problem 2 in Lagaris et al, 1998 (1st order ODE IVP)

Note that an upper-case 'Y' is used to represent the Greek psi from the original equation.

The equation is defined on the domain [0,1]:

The analytical form of the equation is:
    G(x, Y, dY/dx) = dY/dx + Y/5 - exp(-x/5)*cos(x) = 0

with initial condition:

Y(0) = 0

This equation has the analytical solution for the supplied initial conditions:

Ya(x) = exp(-x/5)*sin(x)

Reference:

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis, "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999, 1998

"""


from math import cos, exp, sin
import numpy as np


# Specify the initial condition for this ODE: Y(0) = 1
ic = 0


def Gf(x, Y, dY_dx):
    """Code for differential equation"""
    return dY_dx + Y/5 - exp(-x/5)*cos(x)

def dG_dYf(x, Y, dY_dx):
    """Derivative of G wrt Y"""
    return 1/5

def dG_ddYdxf(x, y, dY_dx):
    """Derivative of G wrt dy/dx"""
    return 1

def Yaf(x):
    """Analytical solution"""
    return exp(-x/5)*sin(x)

def dYa_dxf(x):
    """Derivative of analytical solution"""
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))

if __name__ == '__main__':
    assert Gf(0, 0, 0) == -1
    assert dG_dYf(0, 0, 0) == 1/5
    assert dG_ddYdxf(0, 0, 0) == 1
    assert np.isclose(Yaf(0), ic)
    assert np.isclose(dYa_dxf(0), 1)

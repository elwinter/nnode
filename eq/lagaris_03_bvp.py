"""
Problem 3 in Lagaris et al (2nd order ODE BVP)

Note that an upper-case 'Y' is used to represent the Greek psi from the original equation.

The equation is defined on the domain [0,1]:

The analytical form of the equation is:
    G(x, Y, dY/dx) = d2Y_dx2 + 1/5*dY_dx + Y + 1/5*exp(-x/5)*cos(x) = 0

with boundary condition:

Y(0) = 0
Y(1) = sin(1)*exp(-1/5) =  0.688938...

This equation has the analytical solution for the supplied initial conditions:

Ya(x) = exp(-x/5)*sin(x)

Reference:

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis, "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999, 1998

"""


from math import exp, sin, cos
import numpy as np


bc0 = 0
bc1 = sin(1)*exp(-1/5)  # 0.688938...


def Gf(x, Y, dY_dx, d2Y_dx2):
    """Code for differential equation"""
    return d2Y_dx2 + 1/5*dY_dx + Y + 1/5*exp(-x/5)*cos(x)


def dG_dYf(x, Y, dY_dx, d2Y_dx2):
    """Derivative of G wrt Y"""
    return 1


def dG_ddYdxf(x, Y, dY_dx, d2Y_dx2):
    """Derivative of G wrt dY/dx"""
    return 1/5


def dG_dd2Ydx2f(x, Y, dY_dx, d2Y_dx2):
    """Derivative of G wrt d2Y/dx2"""
    return 1


def Yaf(x):
    """Analytical solution"""
    return exp(-x/5)*sin(x)


def dYa_dxf(x):
    """Derivative of analytical solution"""
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))


def d2Ya_dx2f(x):
    """2nd derivative of analytical solution"""
    return -2/25*exp(-x/5)*(5*cos(x) + 12*sin(x))


if __name__ == '__main__':
    assert np.isclose(Gf(0, 0, 0, 0), 1/5)
    assert np.isclose(dG_dYf(0, 0, 0, 0), 1)
    assert np.isclose(dG_ddYdxf(0, 0, 0, 0), 1/5)
    assert np.isclose(dG_dd2Ydx2f(0, 0, 0, 0), 1)

"""Sample 1st-order ODE IVP

The equation is defined on the domain [0,1], with initial conditions defined
at x=0.

The analytical form of the equation is:
    G(x,y,dy/dx) = dy/dx - y + x**2 - 1 = 0
"""


from inspect import getsource
from math import exp


# Initial condition
ic = 0


def Gf(x, y, dy_dx):
    """Code for differential equation"""
    return dy_dx - y + x**2 - 1


def dG_dyf(x, y, dy_dx):
    """Derivative of G wrt y"""
    return -1


def dG_dydxf(x, y, dy_dx):
    """Derivative of G wrt dy/dx"""
    return 1


def yaf(x):
    """Analytical solution"""
    return (x + 1)**2 - exp(x)


def dya_dxf(x):
    """Derivative of analytical solution"""
    return 2*(x + 1) - exp(x)


if __name__ == '__main__':
    print(getsource(Gf))
    print('Gf(0) = ', Gf(0, 0, 0))
    print()
    print(getsource(dG_dyf))
    print('dG_dyf(0) = ', dG_dyf(0, 0, 0))
    print()
    print(getsource(dG_dydxf))
    print('dG_dydxf(0) = ', dG_dydxf(0, 0, 0))
    print()
    print(getsource(yaf))
    print('yaf(0) = ', yaf(0))
    assert yaf(0) == ic
    print()
    print('dya_dxf(0) = ', dya_dxf(0))

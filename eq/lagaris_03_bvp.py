"""Problem 3 in Lagaris et al (2nd order ODE BVP)

The equation is defined on the domain [0,1], with boundary conditions defined
at x=0.

The analytical form of the equation is:
    G(x,y,dy/dx) = d2y_dx2 - dy_dx + 2x - 1 = 0
"""


from inspect import getsource
from math import exp, sin, cos


bc0 = 0
bc1 = sin(1)*exp(-1/5)  # 0.688938...


def Gf(x, y, dy_dx, d2y_dx2):
    """Code for differential equation"""
    return d2y_dx2 + 1/5*dy_dx + y + 1/5*exp(-x/5)*cos(x)


def dG_dyf(x, y, dy_dx, d2y_dx2):
    """Derivative of G wrt y"""
    return 1


def dG_dydxf(x, y, dy_dx, d2y_dx2):
    """Derivative of G wrt dy/dx"""
    return 1/5


def dG_d2ydx2f(x, y, dy_dx, d2y_dx2):
    """Derivative of G wrt d2y/dx2"""
    return 1


def yaf(x):
    """Analytical solution"""
    return exp(-x/5)*sin(x)


def dya_dxf(x):
    """Derivative of analytical solution"""
    return 1/5*exp(-x/5)*(5*cos(x) - sin(x))


def d2ya_dx2f(x):
    """2nd derivative of analytical solution"""
    return -2/25*exp(-x/5)*(5*cos(x) + 12*sin(x))


if __name__ == '__main__':
    print(getsource(Gf))
    print('Gf(0) = ', Gf(0, 0, 0, 0))
    print()
    print(getsource(dG_dyf))
    print('dG_dyf(0) = ', dG_dyf(0, 0, 0, 0))
    print()
    print(getsource(dG_dydxf))
    print('dG_dydxf(0) = ', dG_dydxf(0, 0, 0, 0))
    print()
    print(getsource(dG_d2ydx2f))
    print('dG_d2ydx2f(0) = ', dG_d2ydx2f(0, 0, 0, 0))
    print()
    print(getsource(yaf))
    print('yaf(0) = ', yaf(0))
    assert yaf(0) == bc0
    assert yaf(1) == bc1
    print()
    print('dya_dxf(0) = ', dya_dxf(0))
    print()
    print('d2ya_dx2f(0) = ', d2ya_dx2f(0))

"""Sample 2nd-order ODE IVP

The equation is defined on the domain [0,1], with the initial conditions
defined at x=0.

The analytical form of the equation is:
    G(x,y,dy/dx) = d2y_dx2 - dy_dx + 2x - 1 = 0
"""


from inspect import getsource


ic = 1
ic1 = 1


def Gf(x, y, dy_dx, d2y_dx2):
    """Code for differential equation"""
    return d2y_dx2 - dy_dx + 2*x - 1


def dG_dyf(x, y, dy_dx, d2y_dx2):
    """Derivative of G wrt y"""
    return 0


def dG_dydxf(x, y, dy_dx, d2y_dx2):
    """Derivative of G wrt dy/dx"""
    return -1


def dG_d2ydx2f(x, y, dy_dx, d2y_dx2):
    """Derivative of G wrt d2y/dx2"""
    return 1


def yaf(x):
    """Analytical solution"""
    return x**2 + x + 1


def dya_dxf(x):
    """Derivative of analytical solution"""
    return 2*x + 1


def d2ya_dx2f(x):
    """2nd derivative of analytical solution"""
    return 2


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
    print()
    print('dya_dxf(0) = ', dya_dxf(0))
    print()
    print('d2ya_dx2f(0) = ', d2ya_dx2f(0))
    assert yaf(0) == ic
    assert dya_dxf(0) == ic1

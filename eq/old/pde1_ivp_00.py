"""Sample 1st-order PDE IVP

The equation is defined on the domain [0,1], with the initial conditions
defined at (x,y)=(0,0).

The analytical form of the equation is:
    G(x,y,dy/dx) = x*y - Y = 0
"""


from inspect import getsource


def Gf(xy, Y, delY):
    """Code for differential equation"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return x * y - Y


def dG_dYf(xy, Y, delY):
    """Derivative of G wrt Y"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return -1


def dG_dY_dxf(xy, Y, delY):
    """Derivative of G wrt x"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return y


def dG_dY_dyf(xy, Y, delY):
    """Derivative of G wrt y"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return x


dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)


def f0f(y):
    """Initial condition at (x,y)=(0,y)"""
    return 0


def g0f(x):
    """Initial condition at (x,y)=(x,0)"""
    return 0


bcf = (f0f, g0f)


def df0_dyf(y):
    """Derivative of initial condition at (x,y)=(0,y)"""
    return 0


def dg0_dxf(x):
    """Derivative of initial condition at (x,y)=(x,0)"""
    return 0


bcdf = (df0_dyf, dg0_dxf)


def Yaf(xy):
    """Analytical solution"""
    (x, y) = xy
    return x * y


def dYa_dxf(xy):
    """Analytical derivative wrt x"""
    (x, y) = xy
    return y


def dYa_dyf(xy):
    """Analytical derivative wrt y"""
    (x, y) = xy
    return x


delYaf = (dYa_dxf, dYa_dyf)


if __name__ == '__main__':
    print(getsource(Gf))
    print('Gf(0, 0], 0, [0, 0]) = ', Gf([0, 0], 0, [0, 0]))
    print()
    print(getsource(dG_dYf))
    print('dG_dYf(0, 0], 0, [0, 0]) = ', dG_dYf([0, 0], 0, [0, 0]))
    print()
    for i in range(2):
        print(getsource(dG_ddelYf[i]))
        print('dG_ddelYf[%d]([0, 0], 0, [0, 0]) = ' % i,
              dG_ddelYf[i]([0, 0], 0, [0, 0]))
        print()
    for i in range(2):
        print(getsource(bcf[i]))
        print('bcf[%d](0) = ' % i, bcf[i](0))
        print()
    for i in range(2):
        print(getsource(bcdf[i]))
        print('bcdf[%d](0) = ' % i, bcdf[i](0))
        print()
    print(getsource(Yaf))
    print('Yaf([0, 0]) = ', Yaf([0, 0]))
    assert Yaf([0, 0]) == bcf[0](0)
    assert Yaf([0, 0]) == bcf[1](0)
    print()

    for i in range(2):
        print(getsource(delYaf[i]))
        print('delYaf[%d]([0,0]) = ' % i, delYaf[i]([0, 0]))
        print()

"""
Problem 5 in Lagaris et al (2nd order PDE)

The equation is defined on the domain [[0,1],[0,1]], with the
Dirichlet boundary conditions specified at x=0|1 and y=0|1.

The analytical form of the equation is:
    G(x,y,dy/dx) = G(x,y,Y,delY,deldelY) =
    d2Y_dx2 + d2Y_dy2 - exp(-x)*(x - 2 + y**3 + 6*y) = 0
"""


from inspect import getsource
from math import exp
import numpy as np


def Gf(xy, Y, delY, deldelY):
    """Code for differential equation"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return d2Y_dxdx + d2Y_dydy - exp(-x)*(x - 2 + y**3 + 6*y)


def dG_dYf(xy, Y, delY, deldelY):
    """Partial of PDE wrt Y"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0


def dG_dY_dxf(xy, Y, delY, deldelY):
    """Partial of PDE wrt dY/dx"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0


def dG_dY_dyf(xy, Y, delY, deldelY):
    """Partial of PDE wrt dY/dy"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0


dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)


def dG_d2Y_dxdxf(xy, Y, delY, deldelY):
    """Partial of PDE wrt d2Y/dx2"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 1


def dG_d2Y_dxdyf(xy, Y, delY, deldelY):
    """Partial of PDE wrt d2Y/dxdy"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 0


def dG_d2Y_dydxf(xy, Y, delY, deldelY):
    """Partial of PDE wrt d2Y/dydx"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 0


def dG_d2Y_dydyf(xy, Y, delY, deldelY):
    """Partial of PDE wrt d2Y/dy2"""
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 1


dG_ddeldelYf = ((dG_d2Y_dxdxf, dG_d2Y_dxdyf),
                (dG_d2Y_dydxf, dG_d2Y_dydyf))


def f0f(y):
    """Boundary condition at (x,y) = (0,y)"""
    return y**3


def f1f(y):
    """Boundary condition at (x,y) = (1,y)"""
    return (1 + y**3)/np.e


def g0f(x):
    """Boundary condition at (x,y) = (x,0)"""
    return x*exp(-x)


def g1f(x):
    """Boundary condition at (x,y) = (x,1)"""
    return exp(-x)*(x + 1)


bcf = ((f0f, g0f), (f1f, g1f))


def df0_dyf(y):
    """1st derivative of BC function at (x,y) = (0,y)"""
    return 3*y**2


def df1_dyf(y):
    """1st derivative of BC function at (x,y) = (1,y)"""
    return 3*y**2/np.e


def dg0_dxf(x):
    """1st derivative of BC function at (x,y) = (x,0)"""
    return exp(-x)*(1 - x)


def dg1_dxf(x):
    """1st derivative of BC function at (x,y) = (x,1)"""
    return -x*exp(-x)


bcdf = ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf))


def d2f0_dy2f(y):
    """2nd derivative of BC function at (x,y) = (0,y)"""
    return 6*y


def d2f1_dy2f(y):
    """2nd derivative of BC function at (x,y) = (1,y)"""
    return 6*y/np.e


def d2g0_dx2f(x):
    """2nd derivative of BC function at (x,y) = (x,0)"""
    return exp(-x)*(x - 2)


def d2g1_dx2f(x):
    """2nd derivative of BC function at (x,y) = (x,1)"""
    return exp(-x)*(x - 1)


bcd2f = ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f))


def Yaf(xy):
    """Analytical solution"""
    (x, y) = xy
    return exp(-x)*(x + y**3)


def dYa_dxf(xy):
    """Analytical dY/dx"""
    (x, y) = xy
    return exp(-x)*(1 - x - y**3)


def dYa_dyf(xy):
    """Analytical dY/dy"""
    (x, y) = xy
    return 3*exp(-x)*y**2


delYaf = (dYa_dxf, dYa_dyf)


def d2Ya_dxdxf(xy):
    """Analytical d2Y/dx2"""
    (x, y) = xy
    return exp(-x)*(x + y**3 - 2)


def d2Ya_dxdyf(xy):
    """Analytical d2Y/dxdy"""
    (x, y) = xy
    return -3*exp(-x)*y**2


def d2Ya_dydxf(xy):
    """Analytical d2Y/dydx"""
    (x, y) = xy
    return -3*exp(-x)*y**2


def d2Ya_dydyf(xy):
    """Analytical d2Y/dy2"""
    (x, y) = xy
    return 6*exp(-x)*y


deldelYaf = ((d2Ya_dxdxf, d2Ya_dxdyf),
             (d2Ya_dydxf, d2Ya_dydyf))


if __name__ == '__main__':
    print(getsource(Gf))
    print('Gf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          Gf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_dYf))
    print('dG_dYf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_dYf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_dY_dxf))
    print('dG_dY_dxf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_dY_dxf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_dY_dyf))
    print('dG_dY_dyf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_dY_dyf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_d2Y_dxdxf))
    print('dG_d2Y_dxdxf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_d2Y_dxdxf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_d2Y_dxdyf))
    print('dG_d2Y_dxdyf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_d2Y_dxdyf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_d2Y_dydxf))
    print('dG_d2Y_dydxf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_d2Y_dydxf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(dG_d2Y_dydyf))
    print('dG_d2Y_dydyf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]) = ',
          dG_d2Y_dydyf([0, 0], 0, [0, 0], [[0, 0], [0, 0]]))
    print()
    print(getsource(f0f))
    print('f0f(0) = ', f0f(0))
    print()
    print(getsource(f1f))
    print('f1f(0) = ', f1f(0))
    print()
    print(getsource(g0f))
    print('g0f(0) = ', g0f(0))
    print()
    print(getsource(g1f))
    print('g1f(0) = ', g1f(0))
    print()
    print(getsource(df0_dyf))
    print('df0_dyf(0) = ', df0_dyf(0))
    print()
    print(getsource(df1_dyf))
    print('df1_dyf(0) = ', df1_dyf(0))
    print()
    print(getsource(dg0_dxf))
    print('dg0_dxf(0) = ', dg0_dxf(0))
    print()
    print(getsource(dg1_dxf))
    print('dg1_dxf(0) = ', dg1_dxf(0))
    print()
    print(getsource(d2f0_dy2f))
    print('d2f0_dy2f(0) = ', d2f0_dy2f(0))
    print()
    print(getsource(d2f1_dy2f))
    print('d2f1_dy2f(0) = ', d2f1_dy2f(0))
    print()
    print(getsource(d2g0_dx2f))
    print('d2g0_dx2f(0) = ', d2g0_dx2f(0))
    print()
    print(getsource(d2g1_dx2f))
    print('d2g1_dx2f(0) = ', d2g1_dx2f(0))
    print()
    print(getsource(Yaf))
    print('Yaf([0, 0]) = ', Yaf([0, 0]))
    assert np.isclose(f0f(0), g0f(0))
    assert np.isclose(f1f(0), g0f(1))
    assert np.isclose(f0f(1), g1f(0))
    assert np.isclose(f1f(1), g1f(1))
    assert np.isclose(Yaf([0, 0]), f0f(0))
    assert np.isclose(Yaf([0, 1]), f0f(1))
    assert np.isclose(Yaf([1, 0]), f1f(0))
    assert np.isclose(Yaf([1, 1]), f1f(1))
    assert np.isclose(Yaf([0.5, 0]), g0f(0.5))
    assert np.isclose(Yaf([0.5, 1]), g1f(0.5))

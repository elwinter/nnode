"""
Problem 5 in Lagaris et al (1998) (2nd order PDE)

Note that an upper-case 'Y' is used to represent the Greek psi from
the original equation.

The equation is defined on the domain [[0,1],[0,1]].

The analytical form of the equation is:
    G(xv, Y, delY, deldelY) = d2Y_dx2 + d2Y_dy2 - exp(-x)*(x - 2 + y**3 + 6*y) = 0

The boundary conditions are:

Y(0, y) = f0(0, y) = y**3
Y(1, y) = f1(1, y) = (1 + y**3)/e
Y(x, 0) = g0(x, 0) = x*exp(-x)
Y(x, 1) = g1(x, 1) = exp(-x)*(x + 1)

The analytical solution is:

Ya(x, y) = exp(-x)*(x + y**3)

Reference:

Isaac Elias Lagaris, Aristidis Likas, and Dimitrios I. Fotiadis, "Artificial Neural Networks for Solving Ordinary and Partial Differential Equations", *IEEE Transactions on Neural Networks* **9**(5), pp. 987-999, 1998
"""


from math import exp
import numpy as np


def Gf(xv, Y, delY, deldelY):
    """Code for differential equation"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return d2Y_dxdx + d2Y_dydy - exp(-x)*(x - 2 + y**3 + 6*y)


def dG_dYf(xv, Y, delY, deldelY):
    """dG/dY"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0


def dG_ddYdxf(xv, Y, delY, deldelY):
    """dG/(dY/dx)"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0

def dG_ddYdyf(xv, Y, delY, deldelY):
    """dG/(dY/dy)"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0

dG_ddelYf = [dG_ddYdxf, dG_ddYdyf]


def dG_dd2Ydxdxf(xv, Y, delY, deldelY):
    """dG/(d2Y/dx2)"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 1

def dG_dd2Ydxdyf(xv, Y, delY, deldelY):
    """dG/(d2Y/dxdy)"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 0

def dG_dd2Ydydxf(xv, Y, delY, deldelY):
    """dG/(d2Y/dydx)"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 0

def dG_dd2Ydydyf(xv, Y, delY, deldelY):
    """dG/(d2Y/dy2)"""
    (x, y) = xv
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 1

dG_ddeldelYf = [[dG_dd2Ydxdxf, dG_dd2Ydxdyf],
                [dG_dd2Ydydxf, dG_dd2Ydydyf]]


def f0f(xv):
    """Boundary condition at (x,y) = (0,y)"""
    (x, y) = xv
    return y**3

def f1f(xv):
    """Boundary condition at (x,y) = (1,y)"""
    (x, y) = xv
    return (1 + y**3)/np.e

def g0f(xv):
    """Boundary condition at (x,y) = (x,0)"""
    (x, y) = xv
    return x*exp(-x)

def g1f(xv):
    """Boundary condition at (x,y) = (x,1)"""
    (x, y) = xv
    return exp(-x)*(x + 1)

bcf = [[f0f, f1f], [g0f, g1f]]


def df0_dxf(xv):
    """df0/dx"""
    (x, y) = xv
    return 0

def df0_dyf(xv):
    """df0/dy"""
    (x, y) = xv
    return 3*y**2

def df1_dxf(xv):
    """df1/dx"""
    (x, y) = xv
    return 0

def df1_dyf(xv):
    """df1/dy"""
    (x, y) = xv
    return 3*y**2/np.e

def dg0_dxf(xv):
    """dg0/dx"""
    (x, y) = xv
    return exp(-x)*(1 - x)

def dg0_dyf(xv):
    """dg0/dy"""
    (x, y) = xv
    return 0

def dg1_dxf(xv):
    """dg1/dx"""
    (x, y) = xv
    return -x*exp(-x)

def dg1_dyf(xv):
    """dg1/dy"""
    (x, y) = xv
    return 0

delbcf = [[[df0_dxf, df0_dyf], [df1_dxf, df1_dyf]],
          [[dg0_dxf, dg0_dyf], [dg1_dxf, dg1_dyf]]]


def d2f0_dxdxf(xv):
    """d2f0/dxdx"""
    x, y = xv
    return 0

def d2f0_dxdyf(xv):
    """d2f0/dxdy"""
    x, y = xv
    return 0

def d2f0_dydxf(xv):
    """d2f0/dydx"""
    x, y = xv
    return 0

def d2f0_dydyf(xv):
    x, y = xv
    return 6*y

def d2f1_dxdxf(xv):
    x, y = xv
    return 0

def d2f1_dxdyf(xv):
    x, y = xv
    return 0

def d2f1_dydxf(xv):
    x, y = xv
    return 0

def d2f1_dydyf(xv):
    x, y = xv
    return 6*y/np.e

def d2g0_dxdxf(xv):
    x, y = xv
    return exp(-x)*(x - 2)

def d2g0_dxdyf(xv):
    x, y = xv
    return 0

def d2g0_dydxf(xv):
    x, y = xv
    return 0

def d2g0_dydyf(xv):
    x, y = xv
    return 0

def d2g1_dxdxf(xv):
    x, y = xv
    return exp(-x)*(x - 1)

def d2g1_dxdyf(xv):
    x, y = xv
    return 0

def d2g1_dydxf(xv):
    x, y = xv
    return 0

def d2g1_dydyf(xv):
    x, y = xv
    return 0

deldelbcf = [[[[d2f0_dxdxf, d2f0_dxdyf], [d2f0_dydxf, d2f0_dydyf]],
              [[d2f1_dxdxf, d2f1_dxdyf], [d2f1_dydxf, d2f1_dydyf]]],
             [[[d2g0_dxdxf, d2g0_dxdyf], [d2g0_dydxf, d2g0_dydyf]],
              [[d2g1_dxdxf, d2g1_dxdyf], [d2g1_dydxf, d2g1_dydyf]]]]


def Yaf(xv):
    """Analytical solution"""
    (x, y) = xv
    return exp(-x)*(x + y**3)


def dYa_dxf(xv):
    """Analytical dY/dx"""
    (x, y) = xv
    return exp(-x)*(1 - x - y**3)

def dYa_dyf(xv):
    """Analytical dY/dy"""
    (x, y) = xv
    return 3*exp(-x)*y**2

delYaf = (dYa_dxf, dYa_dyf)


def d2Ya_dxdxf(xv):
    """Analytical d2Y/dx2"""
    (x, y) = xv
    return exp(-x)*(x + y**3 - 2)

def d2Ya_dxdyf(xv):
    """Analytical d2Y/dxdy"""
    (x, y) = xv
    return -3*exp(-x)*y**2

def d2Ya_dydxf(xv):
    """Analytical d2Y/dydx"""
    (x, y) = xv
    return -3*exp(-x)*y**2

def d2Ya_dydyf(xv):
    """Analytical d2Y/dy2"""
    (x, y) = xv
    return 6*exp(-x)*y

deldelYaf = ((d2Ya_dxdxf, d2Ya_dxdyf),
             (d2Ya_dydxf, d2Ya_dydyf))


if __name__ == '__main__':

    # Values to use in computing test values
    xv_test = (0, 0)
    Y_test = 0
    m = 2
    delY_test = [0, 0]
    deldelY_test = [[0, 0], [0, 0]]

    # Reference values for tests.
    G_ref = 2
    dG_dY_ref = 0
    dG_ddelY_ref = [0, 0]
    dG_ddeldelY_ref = [[1, 0], [0, 1]]
    bc_ref = [[0, 1/np.e], [0, 1]]
    delbc_ref = [[[0, 0], [0, 0]],
                 [[1, 0], [0, 0]]]
    deldelbc_ref = [[[[ 0, 0], [0, 0]],
                     [[ 0, 0], [0, 0]]],
                    [[[-2, 0], [0, 0]],
                     [[-1, 0], [0, 0]]]]
    Ya_ref = 0
    delYa_ref = [1, 0]
    deldelYa_ref = [[-2, 0], [0, 0]]

    print('Testing differential equation and derivatives.')

    assert np.isclose(Gf(xv_test, Y_test, delY_test, deldelY_test), G_ref)

    assert np.isclose(dG_dYf(xv_test, Y_test, delY_test, deldelY_test), dG_dY_ref)

    for j in range(m):
        assert np.isclose(dG_ddelYf[j](xv_test, Y_test, delY_test, deldelY_test),
                          dG_ddelY_ref[j])

    for j1 in range(m):
        for j2 in range(m):
            assert np.isclose(dG_ddeldelYf[j1][j2](xv_test, Y_test, delY_test,
                                                  deldelY_test),
                              dG_ddeldelY_ref[j1][j2])

    print('Testing boundary conditions and derivatives.')

    for j in range(m):
        assert np.isclose(bcf[j][0](xv_test), bc_ref[j][0])
        assert np.isclose(bcf[j][1](xv_test), bc_ref[j][1])

    for j1 in range(m):
        for j2 in range(2):
            for j3 in range(m):
                assert np.isclose(delbcf[j1][j2][j3](xv_test), delbc_ref[j1][j2][j3])

    for j1 in range(m):
        for j2 in range(2):
            for j3 in range(m):
                for j4 in range(m):
                    assert np.isclose(deldelbcf[j1][j2][j3][j4](xv_test),
                                      deldelbc_ref[j1][j2][j3][j4])

    print('Testing analytical solution and derivatives.')

    assert np.isclose(Yaf(xv_test), Ya_ref)

    for j in range(m):
        assert np.isclose(delYaf[j](xv_test), delYa_ref[j])

    for j1 in range(m):
        for j2 in range(m):
            assert np.isclose(deldelYaf[j1][j2](xv_test), deldelYa_ref[j1][j2])

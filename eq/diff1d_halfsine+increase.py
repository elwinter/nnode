"""
1-D diffusion PDE

The analytical form of the equation is:
  G(x,t,Y,delY,del2Y) = dY_dt - D*d2Y_dx2 = 0

The equation is defined on the domain [[0,1],[0,]]. The boundary
conditions are:

Y(0,t) = C = 0
Y(1,t) = C = a*t
Y(x,0) = C = sin(pi*x)
"""


from math import cos, cosh, exp, pi, sin, sinh
import numpy as np


# Diffusion coefficient
D = 0.1

# Boundary increase rate at x=1
a = 0.1

# Number of terms in analytical summation
kmax = 800


def Gf(xt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return dY_dt - D*d2Y_dx2

def dG_dYf(xt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dxf(xt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dtf(xt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return 1

dG_ddelYf = [dG_dY_dxf, dG_dY_dtf]


def dG_d2Y_dx2f(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return -D

def dG_d2Y_dt2f(xt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    (d2Y_dx2, d2Y_dt2) = del2Y
    return 0

dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dt2f]


def f0f(xt):
    """Boundary condition at (x,t) = (0,t)"""
    (x, t) = xt
    return 0

def f1f(xt):
    """Boundary condition at (x,t) = (1,t)"""
    (x, t) = xt
    return a*t

def Y0f(xt):
    """Boundary condition at (x,t) = (x,0)"""
    (x, t) = xt
    return sin(pi*x)

def Y1f(xt):
    """Boundary condition at (x,t) = (x,1) NOT USED"""
    (x, t) = xt
    return None

bcf = [[f0f, f1f], [Y0f, Y1f]]


def df0_dxf(xt):
    """1st derivative of BC wrt x at (x,t) = (0,t)"""
    (x, t) = xt
    return 0

def df0_dtf(xt):
    """1st derivative of BC wrt t at (x,t) = (0,t)"""
    (x, t) = xt
    return 0

def df1_dxf(xt):
    """1st derivative of BC wrt x at (x,t) = (1,t)"""
    (x, t) = xt
    return 0

def df1_dtf(xt):
    """1st derivative of BC wrt t at (x,t) = (1,t)"""
    (x, t) = xt
    return a

def dY0_dxf(xt):
    """1st derivative of BC wrt x at (x,t) = (x,0)"""
    (x, t) = xt
    return pi*cos(pi*x)

def dY0_dtf(xt):
    """1st derivative of BC wrt t at (x,t) = (x,0)"""
    (x, t) = xt
    return 0

def dY1_dxf(xt):
    """1st derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
    (x, t) = xt
    return None

def dY1_dtf(xt):
    """1st derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
    (x, t) = xt
    return None

delbcf = [[[df0_dxf, df0_dtf], [df1_dxf, df1_dtf]],
          [[dY0_dxf, dY0_dtf], [dY1_dxf, dY1_dtf]]]


def d2f0_dx2f(xt):
    """2nd derivative of BC wrt x at (x,t) = (0,t)"""
    (x, t) = xt
    return 0

def d2f0_dt2f(xt):
    """2nd derivative of BC wrt t at (x,t) = (0,t)"""
    (x, t) = xt
    return 0

def d2f1_dx2f(xt):
    """2nd derivative of BC wrt x at (x,t) = (1,t)"""
    (x, t) = xt
    return 0

def d2f1_dt2f(xt):
    """2nd derivative of BC wrt t at (x,t) = (1,t)"""
    (x, t) = xt
    return 0

def d2Y0_dx2f(xt):
    """2nd derivative of BC wrt x at (x,t) = (x,0)"""
    (x, t) = xt
    return -pi**2*sin(pi*x)

def d2Y0_dt2f(xt):
    """2nd derivative of BC wrt t at (x,t) = (x,0)"""
    (x, t) = xt
    return 0

def d2Y1_dx2f(xt):
    """2nd derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
    (x, t) = xt
    return None

def d2Y1_dt2f(xt):
    """2nd derivative of BC wrt x at (x,t) = (x,1) NOT USED"""
    (x, t) = xt
    return None

del2bcf = [[[d2f0_dx2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dt2f]],
           [[d2Y0_dx2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dt2f]]]


def Af(xt):
    """Optimized version of boundary condition function"""
    (x, t) = xt
    A = a*t*x + (1 - t)*sin(pi*x)
    return A

def delAf(xt):
    """Optimized version of boundary condition function gradient"""
    (x, t) = xt
    dA_dx = a*t + pi*(1 - t)*cos(pi*x)
    dA_dt = a*x - sin(pi*x)
    return [dA_dx, dA_dt]

def del2Af(xt):
    """Optimized version of boundary condition function Laplacian"""
    (x, t) = xt
    d2A_dx2 = pi**2*(t - 1)*sin(pi*x)
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dt2]


def Yaf(xt):
    """Analytical solution"""
    (x, t) = xt
    Ya = a*t*x + sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        Ya += 2*(-1)**k*a*(1 - exp(-pi**2*t*D*k**2))*sin(pi*x*k)/ \
              (pi**3*k**3)
    return Ya

def dYa_dxf(xt):
    """Analytical x-gradient"""
    (x, t) = xt
    dYa_dx = a*t + pi*cos(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        dYa_dx += 2*(-1)**k*a*(1 - exp(-k**2*pi**2*t*D))*cos(k*pi*x)/ \
              (k**2*pi**2)
    return dYa_dx

def dYa_dtf(xt):
    """Analytical t-gradient"""
    (x, t) = xt
    dYa_dt = a*x + sin(pi*x)*pi**2*D*(sinh(pi**2*t*D) - cosh(pi**2*t*D))
    for k in range(1, kmax + 1):
        dYa_dt += 2*(-1)**k*a*exp(-k**2*pi**2*t*D)*D*sin(k*pi*x)/ \
                  (k*pi)
    return dYa_dt

delYaf = [dYa_dxf, dYa_dtf]


def d2Ya_dx2f(xt):
    """Analytical x-Laplacian"""
    (x, t) = xt
    d2Ya_dx2 = -pi**2*sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        d2Ya_dx2 += -2*(-1)**k*a*(1 - exp(-pi**2*t*D*k**2))*sin(pi*x*k)/ \
              (pi*k)
    return d2Ya_dx2

def d2Ya_dt2f(xt):
    """Analytical t-Laplacian"""
    (x, t) = xt
    d2Ya_dt2 = sin(pi*x)*pi**4*D**2*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        d2Ya_dt2 += -2*(-1)**k*a*exp(-pi**2*t*D*k**2)*k*pi*D**2*sin(pi*x*k)
    return d2Ya_dt2

del2Yaf = [d2Ya_dx2f, d2Ya_dt2f]


if __name__ == '__main__':

    # Test values
    xt = [0.4, 0.5]

    # Reference values for tests.
    G_ref = 0.0360028908174169   # Non-zero due to roundoff error?
    dG_dY_ref = 0
    dG_ddelY_ref = (0, 1)
    dG_ddel2Y_ref = (-D, 0)
    bc_ref = [[0, 0.05],
              [0.951056516295154, None]]
    delbc_ref = [[[0, 0], [0, 0.1]],
                 [[0.970805519362733, 0], [None, None]]]
    del2bc_ref = [[[0, 0], [0, 0]],
                  [[-9.38655157891136, 0], [None, None]]]
    A_ref = 0.4955282581475768
    delA_ref = [0.5354027596813666, -0.911056516295153]
    del2A_ref = [-4.69327578945568, 0]
    Ya_ref = 0.5986958383019955
    delYa_ref =  [ 0.6383788665776866, -0.536469420056056]
    del2Ya_ref = [-5.724723108734729,   0.5680753052926439]

    print("Testing differential equation.")
    assert np.isclose(Gf(xt, Ya_ref, delYa_ref, del2Ya_ref), G_ref)

    print("Testing differential equation Y-derivative.")
    assert np.isclose(dG_dYf(xt, Ya_ref, delYa_ref, del2Ya_ref), dG_dY_ref)

    print("Testing differential equation gradient-derivatives.")
    for (i, f) in enumerate(dG_ddelYf):
        assert np.isclose(f(xt, Ya_ref, delYa_ref, del2Ya_ref),
                          dG_ddelY_ref[i])

    print("Testing differential equation Laplacian-derivatives.")
    for (i, f) in enumerate(dG_ddel2Yf):
        assert np.isclose(f(xt, Ya_ref, delYa_ref, del2Ya_ref),
                          dG_ddel2Y_ref[i])

    print("Testing boundary conditions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            if bc_ref[i][j] is None:
                assert f(xt) is None
            else:
                assert np.isclose(f(xt), bc_ref[i][j])

    print("Testing boundary condition gradients.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                if delbc_ref[i][j][k] is None:
                    assert f(xt) is None
                else:
                    assert np.isclose(f(xt), delbc_ref[i][j][k])

    print("Testing boundary condition Laplacians.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                if del2bc_ref[i][j][k] is None:
                    assert f(xt) is None
                else:
                    assert np.isclose(f(xt), del2bc_ref[i][j][k])

    print("Verifying BC continuity constraints.")
    assert np.isclose(f0f([0, 0]), Y0f([0, 0]))
    assert np.isclose(f1f([1, 0]), Y0f([1, 0]))
    # t=1 not used

    print("Testing optimized BC function.")
    assert np.isclose(Af(xt), A_ref)

    print("Testing optimized BC function gradient.")
    delA = delAf(xt)
    for i in range(len(delA_ref)):
        assert np.isclose(delA[i], delA_ref[i])

    print("Testing optimized BC function Laplacian.")
    del2A = del2Af(xt)
    for i in range(len(del2A_ref)):
        assert np.isclose(del2A[i], del2A_ref[i])

    print("Testing analytical solution.")
    assert np.isclose(Yaf(xt), Ya_ref)

    print("Testing analytical solution gradient.")
    for (i, f) in enumerate(delYaf):
        if delYa_ref[i] is None:
            assert f(xt) is None
        else:
            assert np.isclose(f(xt), delYa_ref[i])

    print("Testing analytical solution Laplacian.")
    for (i, f) in enumerate(del2Yaf):
        if del2Ya_ref[i] is None:
            assert f(xt) is None
        else:
            assert np.isclose(f(xt), del2Ya_ref[i])

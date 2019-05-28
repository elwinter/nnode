###############################################################################
"""
1-D diffusion PDE

The analytical form of the equation is:
  G(x,t,Y,delY,del2Y) = dY_dt - D*d2Y_dx2 = 0

The equation is defined on the domain [[0,1],[0,]]. The
initial profile is:

Y(x,0) = sin(pi*x)

with the x=1 value increasing linearly at rate a.
"""


from math import cos, cosh, exp, pi, sin, sinh
import numpy as np


# Diffusion coefficient
D = 0.1

# Boundary increase rate at x=1
a = 0.1

# Number of terms in analytical summation
kmax = 400


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


def Yaf(xt):
    """Analytical solution"""
    (x, t) = xt
    Ya = t*x*a + sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        Ya += 2*(-1)**k*(1 - exp(-pi**2*t*D*k**2))*a*sin(pi*x*k)/ \
              (pi**3*k**3)
    return Ya

def dYa_dxf(xt):
    """Analytical x-gradient"""
    (x, t) = xt
    dYa_dx = t*a + pi*cos(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        dYa_dx += 2*(-1)**k*(1 - exp(-pi**2*t*D*k**2))*a*cos(pi*x*k)/ \
              (pi**2*k**2)
    return dYa_dx

def dYa_dtf(xt):
    """Analytical t-gradient"""
    (x, t) = xt
    dYa_dt = x*a + sin(pi*x)*(-pi**2*D*cosh(pi**2*t*D) + pi**2*D*sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        dYa_dt += 2*(-1)**k*exp(-pi**2*t*D*k**2)*D*a*sin(pi*x*k)/ \
                  (pi*k)
    return dYa_dt

delYaf = [dYa_dxf, dYa_dtf]


def d2Ya_dx2f(xt):
    """Analytical x-Laplacian"""
    (x, t) = xt
    d2Ya_dx2 = -pi**2*sin(pi*x)*(cosh(pi**2*t*D) - sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        d2Ya_dx2 += -2*(-1)**k*(1 - exp(-pi**2*t*D*k**2))*a*sin(pi*x*k)/ \
              (pi*k)
    return d2Ya_dx2

def d2Ya_dt2f(xt):
    """Analytical t-Laplacian"""
    (x, t) = xt
    d2Ya_dt2 = sin(pi*x)*(pi**4*D**2*cosh(pi**2*t*D) - pi**4*D**2*sinh(pi**2*t*D))
    for k in range(1, kmax + 1):
        d2Ya_dt2 += -2*(-1)**k*exp(-pi**2*t*D*k**2)*k*pi*D**2*a*sin(pi*x*k)
    return d2Ya_dt2

del2Yaf = [d2Ya_dx2f, d2Ya_dt2f]


if __name__ == '__main__':

    # Test values
    xt = [0.4, 0.5]

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    (dG_dY_dx_ref, dG_dY_dt_ref) = (0, 1)
    (dG_d2Y_dx2_ref, dG_d2Y_dt2_ref) = (-D, 0)
    bc_ref = [[0, 0.05],
              [0.951057, None]]
    delbc_ref = [[[0, 0], [0, 0.1]],
                 [[0.970806, 0], [None, None]]]
    del2bc_ref = [[[0, 0], [0, 0]],
                  [[-9.38655, 0], [None, None]]]
    Ya_ref = 0.598696
    delYa_ref = [0.638379, -0.536469]
    del2Ya_ref = [-5.72475, 0.568075]

    print("Testing differential equation.")
    G = Gf(xt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(G, G_ref):
        print("ERROR: G = %s, vs ref %s" % (G, G_ref))

    print("Testing differential equation Y-derivative.")
    dG_dY = dG_dYf(xt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY, dG_dY_ref):
        print("ERROR: dG_dY = %s, vs ref %s" % (dG_dY, dG_dY_ref))

    print("Testing differential equation dY/dx-derivative.")
    dG_dY_dx = dG_dY_dxf(xt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dx, dG_dY_dx_ref):
        print("ERROR: dG_dY_dx = %s, vs ref %s" % (dG_dY_dx, dG_dY_dx_ref))

    print("Testing differential equation dY/dt-derivative.")
    dG_dY_dt = dG_dY_dtf(xt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dt, dG_dY_dt_ref):
        print("ERROR: dG_dY_dt = %s, vs ref %s" % (dG_dY_dt, dG_dY_dt_ref))

    print("Testing differential equation d2Y/dx2-derivative.")
    dG_d2Y_dx2 = dG_d2Y_dx2f(xt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dx2, dG_d2Y_dx2_ref):
        print("ERROR: dG_d2Y_dx2 = %s, vs ref %s" % (dG_d2Y_dx2, dG_d2Y_dx2_ref))

    print("Testing differential equation d2Y/dt2-derivative.")
    dG_d2Y_dt2 = dG_d2Y_dt2f(xt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dt2, dG_d2Y_dt2_ref):
        print("ERROR: dG_d2Y_dt2 = %s, vs ref %s" % (dG_d2Y_dt2, dG_d2Y_dt2_ref))

    print("Testing boundary conditions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            bc = f(xt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                delbc = f(xt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                del2bc = f(xt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing analytical solution.")
    Ya = Yaf(xt)
    if not np.isclose(Ya, Ya_ref):
        print("ERROR: Ya = %s, vs ref %s" % (Ya, Ya_ref))

    print("Testing analytical solution gradient.")
    for (i, f) in enumerate(delYaf):
        delYa = f(xt)
        if ((delYa_ref[i] is not None and not np.isclose(delYa, delYa_ref[i]))
                or (delYa_ref[i] is None and delYa is not None)):
                print("ERROR: delYa[%d] = %s, vs ref %s" % (i, delYa, delYa_ref[i]))

    print("Testing analytical solution Laplacian.")
    for (i, f) in enumerate(del2Yaf):
        del2Ya = f(xt)
        if ((del2Ya_ref[i] is not None and not np.isclose(del2Ya, del2Ya_ref[i]))
                or (del2Ya_ref[i] is None and del2Ya is not None)):
                print("ERROR: del2Ya[%d] = %s, vs ref %s" % (i, del2Ya, del2Ya_ref[i]))

################################################################################
"""
2-D diffusion PDE

The analytical form of the equation is:
  G(x,y,t,Y,delY,del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2) = 0

The equation is defined on the domain (x,y,t)=([0,1],[0,1],[0,]). The
initial profile is:

Y(x,y,0) = sin(pi*x)*sin(pi*y)/2
"""


from math import cos, exp, pi, sin
import numpy as np


# Diffusion coefficient
D = 0.1


def Gf(xyt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2)

def dG_dYf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dxf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dyf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dy"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dtf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 1

dG_ddelYf = [dG_dY_dxf, dG_dY_dyf, dG_dY_dtf]


def dG_d2Y_dx2f(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return -D

def dG_d2Y_dy2f(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dy2"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return -D

def dG_d2Y_dt2f(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dy2f, dG_d2Y_dt2f]


def f0f(xyt):
    """Boundary condition at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def f1f(xyt):
    """Boundary condition at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def g0f(xyt):
    """Boundary condition at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def g1f(xyt):
    """Boundary condition at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def Y0f(xyt):
    """Boundary condition at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return sin(pi*x)*sin(pi*y)/2

def Y1f(xyt):
    """Boundary condition at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

bcf = [[f0f, f1f], [g0f, g1f], [Y0f, Y1f]]


def df0_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def df0_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def df0_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def df1_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def df1_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def df1_dtf(xyt):
    """1st derivative of BC wrt z at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def dg0_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def dg0_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def dg0_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def dg1_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def dg1_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def dg1_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def dY0_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return pi*cos(pi*x)*sin(pi*y)/2

def dY0_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return pi*sin(pi*x)*cos(pi*y)/2

def dY0_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return 0

def dY1_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def dY1_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def dY1_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

delbcf = [[[df0_dxf, df0_dyf, df0_dtf], [df1_dxf, df1_dyf, df1_dtf]],
          [[dg0_dxf, dg0_dyf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dtf]],
          [[dY0_dxf, dY0_dyf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dtf]]]


def d2f0_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f0_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f0_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f1_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f1_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f1_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def d2g0_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def d2g0_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def d2g0_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def d2g1_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def d2g1_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def d2g1_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def d2Y0_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return -pi**2*sin(pi*x)*sin(pi*y)/2

def d2Y0_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return -pi**2*sin(pi*x)*sin(pi*y)/2

def d2Y0_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return 0

def d2Y1_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def d2Y1_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def d2Y1_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dt2f]],
           [[d2g0_dx2f, d2g0_dy2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dt2f]],
           [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dt2f]]]


def Af(xyt):
    """Optimized version of boundary condition function"""
    (x, y, t) = xyt
    A = -1/2*(-1 + t)*sin(pi*x)*sin(pi*y)
    return A

def delAf(xyt):
    """Optimized version of boundary condition function gradient"""
    (x, y, t) = xyt
    dA_dx = -pi/2*(-1 + t)*cos(pi*x)*sin(pi*y)
    dA_dy = -pi/2*(-1 + t)*sin(pi*x)*cos(pi*y)
    dA_dt = -1/2*sin(pi*x)*sin(pi*y)
    return [dA_dx, dA_dy, dA_dt]

def del2Af(xyt):
    """Optimized version of boundary condition function Laplacian"""
    (x, y, t) = xyt
    d2A_dx2 = pi**2/2*(-1 + t)*sin(pi*x)*sin(pi*y)
    d2A_dy2 = pi**2/2*(-1 + t)*sin(pi*x)*sin(pi*y)
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dy2, d2A_dt2]


def Yaf(xyt):
    """Analytical solution"""
    (x, y, t) = xyt
    return exp(-2*pi**2*D*t)*sin(pi*x)*sin(pi*y)/2

def dYa_dxf(xyt):
    """Analytical x-gradient"""
    (x, y, t) = xyt
    return exp(-2*pi**2*D*t)*pi*cos(pi*x)*sin(pi*y)/2

def dYa_dyf(xyt):
    """Analytical y-gradient"""
    (x, y, t) = xyt
    return exp(-2*pi**2*D*t)*pi*sin(pi*x)*cos(pi*y)/2

def dYa_dtf(xyt):
    """Analytical t-gradient"""
    (x, y, t) = xyt
    return -exp(-2*pi**2*D*t)*pi**2*D*sin(pi*x)*sin(pi*y)

delYaf = [dYa_dxf, dYa_dyf, dYa_dtf]


def d2Ya_dx2f(xyt):
    """Analytical x-Laplacian"""
    (x, y, t) = xyt
    return -exp(-2*pi**2*D*t)*pi**2*sin(pi*x)*sin(pi*y)/2

def d2Ya_dy2f(xyt):
    """Analytical y-Laplacian"""
    (x, y, t) = xyt
    return -exp(-2*pi**2*D*t)*pi**2*sin(pi*x)*sin(pi*y)/2

def d2Ya_dt2f(xyt):
    """Analytical t-Laplacian"""
    (x, y, t) = xyt
    return 2*exp(-2*pi**2*D*t)*pi**4*D**2*sin(pi*x)*sin(pi*y)

del2Yaf = [d2Ya_dx2f, d2Ya_dy2f, d2Ya_dt2f]


if __name__ == '__main__':

    # Test values
    xyt = [0.4, 0.5, 0.6]

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    (dG_dY_dx_ref, dG_dY_dy_ref, dG_dY_dt_ref) = [0, 0, 1]
    (dG_d2Y_dx2_ref, dG_d2Y_dy2_ref, dG_d2Y_dt2_ref) = [-D, -D, 0]
    bc_ref = [[0, 0],
              [0, 0],
              [0.475528, None]]
    delbc_ref = [[[0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0]],
                 [[0.485403, 0, 0], [None, None, None]]]
    del2bc_ref = [[[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[-4.69328, -4.69328, 0], [None, None, None]]]
    Ya_ref = 0.145485
    delYa_ref = [0.148506, 0, -0.287176]
    del2Ya_ref = [-1.43588, -1.43588, 0.566863]

    print("Testing differential equation.")
    G = Gf(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(G, G_ref):
        print("ERROR: G = %s, vs ref %s" % (G, G_ref))

    print("Testing differential equation Y-derivative.")
    dG_dY = dG_dYf(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY, dG_dY_ref):
        print("ERROR: dG_dY = %s, vs ref %s" % (dG_dY, dG_dY_ref))

    print("Testing differential equation dY/dx-derivative.")
    dG_dY_dx = dG_dY_dxf(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dx, dG_dY_dx_ref):
        print("ERROR: dG_dY_dx = %s, vs ref %s" % (dG_dY_dx, dG_dY_dx_ref))

    print("Testing differential equation dY/dy-derivative.")
    dG_dY_dy = dG_dY_dyf(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dy, dG_dY_dy_ref):
        print("ERROR: dG_dY_dy = %s, vs ref %s" % (dG_dY_dy, dG_dY_dy_ref))

    print("Testing differential equation dY/dt-derivative.")
    dG_dY_dt = dG_dY_dtf(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dt, dG_dY_dt_ref):
        print("ERROR: dG_dY_dt = %s, vs ref %s" % (dG_dY_dt, dG_dY_dt_ref))

    print("Testing differential equation d2Y/dx2-derivative.")
    dG_d2Y_dx2 = dG_d2Y_dx2f(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dx2, dG_d2Y_dx2_ref):
        print("ERROR: dG_d2Y_dx2 = %s, vs ref %s" % (dG_d2Y_dx2, dG_d2Y_dx2_ref))

    print("Testing differential equation d2Y/dy2-derivative.")
    dG_d2Y_dy2 = dG_d2Y_dy2f(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dy2, dG_d2Y_dy2_ref):
        print("ERROR: dG_d2Y_dy2 = %s, vs ref %s" % (dG_d2Y_dy2, dG_d2Y_dy2_ref))

    print("Testing differential equation d2Y/dt2-derivative.")
    dG_d2Y_dt2 = dG_d2Y_dt2f(xyt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dt2, dG_d2Y_dt2_ref):
        print("ERROR: dG_d2Y_dt2 = %s, vs ref %s" % (dG_d2Y_dt2, dG_d2Y_dt2_ref))

    print("Testing boundary conditions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            bc = f(xyt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                delbc = f(xyt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                del2bc = f(xyt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing analytical solution.")
    Ya = Yaf(xyt)
    if not np.isclose(Ya, Ya_ref):
        print("ERROR: Ya = %s, vs ref %s" % (Ya, Ya_ref))

    print("Testing analytical solution gradient.")
    for (i, f) in enumerate(delYaf):
        delYa = f(xyt)
        if ((delYa_ref[i] is not None and not np.isclose(delYa, delYa_ref[i]))
                or (delYa_ref[i] is None and delYa is not None)):
                print("ERROR: delYa[%d] = %s, vs ref %s" % (i, delYa, delYa_ref[i]))

    print("Testing analytical solution Laplacian.")
    for (i, f) in enumerate(del2Yaf):
        del2Ya = f(xyt)
        if ((del2Ya_ref[i] is not None and not np.isclose(del2Ya, del2Ya_ref[i]))
                or (del2Ya_ref[i] is None and del2Ya is not None)):
                print("ERROR: del2Ya[%d] = %s, vs ref %s" % (i, del2Ya, del2Ya_ref[i]))

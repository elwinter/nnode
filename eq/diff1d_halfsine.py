################################################################################
"""
1-D diffusion PDE

The analytical form of the equation is:
  G(x,t,Y,delY,del2Y) = dY_dt - D*d2Y_dx2 = 0

The equation is defined on the domain [[0,1],[0,]]. The
initial profile is:

Y(x,0) = sin(pi*x)
"""


from math import cos, exp, pi, sin
import numpy as np


# Diffusion coefficient
D = 0.1


def Gf(xt, Y, delY, del2Y):
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

dG_ddelYf = (dG_dY_dxf, dG_dY_dtf)


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
    return 0

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
    return 0

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
    Ya = exp(-pi**2*D*t)*sin(pi*x)
    return Ya


if __name__ == '__main__':

    # Test values
    xt_ref = [0, 0]
    Y_ref = 0
    delY_ref = [0, 0]
    del2Y_ref = [0, 0]

    # Reference values for tests.
    G_ref = 0

    print("Testing differential equation.")
    G = Gf(xt_ref, Y_ref, delY_ref, del2Y_ref)
    if not np.isclose(G, G_ref):
        print("ERROR: G = %s, vs ref %s" % (G, G_ref))

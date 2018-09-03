"""1-D diffusion problem with sine wave initial condition

The equation is defined on the domain x=[0,1], t=[0,1], with initial
conditions defined at t=0, and boundary conditions at x=0,1.

The analytical form of the equation is:
    G(x,y,dy/dx) = dY_dt - D*d2Y_dx2 = 0
"""

from math import exp, pi, sin, cos


# Diffusion coefficient
D = 1


def Gf(xt, Y, delY, deldelY):
    """Code for differential equation"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return dY_dt - D*d2Y_dxdx


def dG_dYf(xt, Y, delY, deldelY):
    """Derivative of G wrt Y"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0


def dG_dY_dxf(xt, Y, delY, deldelY):
    """Derivative of G wrt dY/dx"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0


def dG_dY_dtf(xt, Y, delY, deldelY):
    """Derivative of G wrt dY/dt"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 1


dG_ddelYf = (dG_dY_dxf, dG_dY_dtf)


def dG_d2Y_dxdxf(xt, Y, delY, deldelY):
    """Derivative of G wrt d2Y/dx2"""
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return -D


def dG_d2Y_dxdtf(xt, Y, delY, deldelY):
    """Derivative of G wrt d2Y/dxdt"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0


def dG_d2Y_dtdxf(xt, Y, delY, deldelY):
    """Derivative of G wrt d2Y/dtdx"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0


def dG_d2Y_dtdtf(xt, Y, delY, deldelY):
    """Derivative of G wrt d2Y/dt2"""
    (x, t) = xt
    (dY_dx, dY_dt) = delY
    ((d2Y_dxdx, d2Y_dxdt), (d2Y_dtdx, d2Y_dtdt)) = deldelY
    return 0


dG_ddeldelYf = ((dG_d2Y_dxdxf, dG_d2Y_dxdtf),
                (dG_d2Y_dtdxf, dG_d2Y_dtdtf))


def f0f(t):
    """Boundary condition at (x,t)=(0,t)"""
    return 0.5 + 0.1*sin(2*pi*t)


def f1f(t):
    """Boundary condition at (x,t)=(1,t)"""
    return 0.5


def g0f(x):
    """Initial condition at (x,t)=(x,0)"""
    return 0.5


def g1f(x):
    """NOT USED - Final condition at (x,t)=(x,1)"""
    return None


bcf = ((f0f, g0f), (f1f, g1f))


def df0_dtf(t):
    """1st derivative of boundary condition at (x,t)=(0,t) wrt t"""
    return 0.2*pi*cos(2*pi*t)


def df1_dtf(t):
    """1st derivative of boundary condition at (x,t)=(1,t) wrt t"""
    return 0


def dg0_dxf(x):
    """1st derivative of initial condition at (x,t)=(x,0) wrt x"""
    return 0


def dg1_dxf(x):
    """NOT USED - 1st derivative of final condition at (x,t)=(x,0) wrt x"""
    return None


bcdf = ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf))


def d2f0_dt2f(t):
    """2nd derivative of boundary condition at (x,t)=(0,t) wrt t"""
    return -0.4*pi**2*sin(2*pi*t)


def d2f1_dt2f(t):
    """2nd derivative of boundary condition at (x,t)=(1,t) wrt t"""
    return 0


def d2g0_dx2f(x):
    """2nd derivative of initial condition at (x,t)=(x,0) wrt x"""
    return 0


def d2g1_dx2f(x):
    """NOT USED - 2nd derivative of final condition at (x,t)=(x,0) wrt x"""
    return None


bcd2f = ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f))


def Yaf(xt):
    """Analytical solution"""
    (x, t) = xt
    Ya = 0.5 + 0.1*(1 - x)*sin(2*pi*t)
    fsum = 0
    for k in range(1, 21):
        fsum += (-0.10132118364233778*exp(-9.869604401089358*t*D*k**2) +
                 0.10132118364233778*D*k**2*cos(2*pi*t) +
                 0.06450306886639899*sin(2*pi*t))*sin(pi*x*k) / \
                 (k*(0.40528473456935116 + D**2*k**4))
    fsum *= 0.4*D
    Ya -= fsum
    return Ya

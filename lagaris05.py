# This PDE is problem 5 in Lagaris et al.

from math import exp
import numpy as np

# A reasonable solution can be found using the following settings:
# ???

# The equation is defined on the domain [[0,1],[0,1]], with the
# Dirichlet boundary conditions specified at x=0|1 and y=0|1.

# The analytical form of the equation is:

# G(x,y,Y,delY,deldelY) = d2Y_dx2 + d2Y_dy2 - exp(-x)*(x - 2 + y**3 + 6*y) = 0

# The analytical solution is:

# Y(x,y) = exp(-x)*(x + y**3)

# Define the original differential equation.
def Gf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return d2Y_dxdx + d2Y_dydy - exp(-x)*(x - 2 + y**3 + 6*y)

# Partial of PDE wrt Y
def dG_dYf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0

# Partials of PDE wrt del Y (partials wrt gradient of Y)
def dG_dY_dxf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0

def dG_dY_dyf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return 0

dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)

# Partials of PDE wrt del del Y (partials wrt Hessian of Y)
def dG_d2Y_dxdxf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 1

def dG_d2Y_dxdyf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 0

def dG_d2Y_dydxf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 0

def dG_d2Y_dydyf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Y_dydy)) = deldelY
    return 1

dG_ddeldelYf = ((dG_d2Y_dxdxf, dG_d2Y_dxdyf),
                (dG_d2Y_dydxf, dG_d2Y_dydyf))

# Boundary conditions for x at 0 and 1
def f0f(y):
    return y**3

def f1f(y):
    return (1 + y**3)/np.e

# Boundary conditions for y at 0 and 1
def g0f(x):
    return x*exp(-x)

def g1f(x):
    return exp(-x)*(x + 1)

# Array of BC functions
bcf = ((f0f, g0f), (f1f, g1f))

# 1st derivative of BC functions for x at 0 and 1
def df0_dyf(y):
    return 3*y**2

def df1_dyf(y):
    return 3*y**2/np.e

# 1st derivative of BC functions for y at 0 and 1
def dg0_dxf(x):
    return exp(-x)*(1 - x)

def dg1_dxf(x):
    return -x*exp(-x)

# Array of BC 1st derivatives
bcdf = ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf))

# 2nd derivatives of BC functions for x at 0 and 1
def d2f0_dy2f(y):
    return 6*y

def d2f1_dy2f(y):
    return 6*y/np.e

# 2nd derivatives of BC functions for y at 0 and 1
def d2g0_dx2f(x):
    return exp(-x)*(x - 2)

def d2g1_dx2f(x):
    return exp(-x)*(x - 1)

# Array of BC 2nd derivatives
bcd2f = ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f))

# Define the analytical solution and its derivatives.
def Yaf(xy):
    (x, y) = xy
    return exp(-x)*(x + y**3)

def dYa_dxf(xy):
    (x, y) = xy
    return exp(-x)*(1 - x - y**3)

def dYa_dyf(xy):
    (x, y) = xy
    return 3*exp(-x)*y**2

delYaf = (dYa_dxf, dYa_dyf)

def d2Ya_dxdxf(xy):
    (x, y) = xy
    return exp(-x)*(x + y**3 - 2)

def d2Ya_dxdyf(xy):
    (x, y) = xy
    return -3*exp(-x)*y**2

def d2Ya_dydxf(xy):
    (x, y) = xy
    return -3*exp(-x)*y**2

def d2Ya_dydyf(xy):
    (x, y) = xy
    return 6*exp(-x)*y

deldelYaf = ((d2Ya_dxdxf, d2Ya_dxdyf),
             (d2Ya_dydxf, d2Ya_dydyf))

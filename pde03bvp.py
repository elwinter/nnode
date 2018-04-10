# Sample 2nd-order PDE BVP

from math import sqrt

# A reasonable solution can be found using the following settings:
# 

# The equation is defined on the domain [[0,1],[0,1]], with the
# Dirichlet boundary conditions specified at x=0|1 and y=0|1.

# The analytical form of the equation is:

# G(x,y,Y,dY/dx,dY/dy,d2Y_dx2,d2Y_dy2) =
# 1/2*(x*dY/dx + y*dY/dy) - 1/4*(d2Y/dx2 + d2Y/dy2)Y = 0

# The analytical solution is:

# Y(x,y) = sqrt(x**2 + y**2)

# Define the original differential equation.
def Gf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return (x*dY_dx + y*dY_dy)/2 - (d2Y_dx2 + d2Y_dy2)/4*Y

# Partial of PDE wrt Y
def dG_dYf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return -(d2Y_dx2 + d2Y_dy2)/4

# Partial of PDE wrt gradient of Y
def dG_dY_dxf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return x/2

def dG_dY_dyf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return y/2

dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)

# Partial of PDE wrt del**2 of Y
def dG_d2Y_dx2f(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return -Y/4

def dG_d2Y_dy2f(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return -Y/4

dG_ddel2Yf = (dG_d2Y_dx2f, dG_d2Y_dy2f)

# Boundary conditions for x at 0 and 1
def f0f(y):
    return y**2

def f1f(y):
    return 1 + y**2

# Boundary conditions for y at 0 and 1
def g0f(x):
    return x**2

def g1f(x):
    return x**2 + 1

# Array of BC functions
bcf = ((f0f, g0f), (f1f, g1f))

# 1st derivative of BC functions for x at 0 and 1
def df0_dyf(y):
    return 2*y

def df1_dyf(y):
    return 2*y

# 1st derivative of BC functions for y at 0 and 1
def dg0_dxf(x):
    return 2*x

def dg1_dxf(x):
    return 2*x

# Array of BC 1st derivatives
bcdf = ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf))

# 2nd derivatives of BC functions for x at 0 and 1
def d2f0_dy2f(y):
    return 2

def d2f1_dy2f(y):
    return 2

# 2nd derivatives of BC functions for y at 0 and 1
def d2g0_dx2f(x):
    return 2

def d2g1_dx2f(x):
    return 2

# Array of BC 2nd derivatives
bcd2f = ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f))

# Define the analytical solution and its derivatives.
def Yaf(xy):
    (x, y) = xy
    return x**2 + y**2

def dYa_dxf(xy):
    (x, y) = xy
    return 2*x

def dYa_dyf(xy):
    (x, y) = xy
    return 2*y

delYaf = (dYa_dxf, dYa_dyf)

def d2Ya_dx2f(xy):
    (x, y) = xy
    return 2

def d2Ya_dy2f(xy):
    (x, y) = xy
    return 2

del2Yaf = (d2Ya_dx2f, d2Ya_dy2f)

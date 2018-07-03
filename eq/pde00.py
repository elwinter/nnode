# Sample 1st-order PDE IVP

# A reasonable solution can be found using the following settings:
# All defaults

# The equation is defined on the domain [[0,1],[0,1]], with the
# initial conditions specified at x=0 and y=0.

# Define the original differential equation, assumed to be in the form
# G(x,y,Y,dY/dx,dY/dy) = x*y - Y = 0.
# The analytical solution is: Y(x,y) = x*y
def Gf(xy, Y, delY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    return x * y - Y

# First partials of the PDE

def dG_dYf(xy, Y, delY):
    return -1

def dG_dY_dxf(xy, Y, delY):
    return 0

def dG_dY_dyf(xy, Y, delY):
    return 0

dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)

# Boundary condition functions and derivatives
def f0f(y):
    return 0

def g0f(x):
    return 0

bcf = (f0f, g0f)

def df0_dyf(y):
    return 0

def dg0_dxf(x):
    return 0

bcdf = (df0_dyf, dg0_dxf)

# Define the analytical solution and its derivatives.
def Yaf(xy):
    (x, y) = xy
    return x * y

def dYa_dxf(xy):
    (x, y) = xy
    return y

def dYa_dyf(xy):
    (x, y) = xy
    return x

delYaf = (dYa_dxf, dYa_dyf)

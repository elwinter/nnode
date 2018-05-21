# Sample 1st-order PDE IVP

# A reasonable solution can be found using the following settings:
# 

# The equation is defined on the domain [[0,1],[0,1]], with the
# initial conditions specified at x=0 and y=0.

# Define the original differential equation, assumed to be in the form
# G(x,y,Y,dY/dx,dY/dy) = 1/2*(x*dY_dx + y*dY_dy) - Y = 0.
# The analytical solution is: Y(x,y) = x**2 + y**2
def Gf(xy, Y, del_Y):
    (x, y) = xy
    (dY_dx, dY_dy) = del_Y
    return 1/2*(x*dY_dx + y*dY_dy) - Y

# First partials of the PDE

def dG_dxf(xy, Y, del_Y):
    (x, y) = xy
    (dY_dx, dY_dy) = del_Y
    return 1/2*dY_dx

def dG_dyf(xy, Y, del_Y):
    (x, y) = xy
    (dY_dx, dY_dy) = del_Y
    return 1/2*dY_dy

del_Gf = (dG_dxf, dG_dyf)

def dG_dYf(xy, Y, del_Y):
    (x, y) = xy
    (dY_dx, dY_dy) = del_Y
    return -1

def dG_dY_dxf(xy, Y, del_Y):
    (x, y) = xy
    (dY_dx, dY_dy) = del_Y
    return x/2

def dG_dY_dyf(xy, Y, del_Y):
    (x, y) = xy
    (dY_dx, dY_dy) = del_Y
    return y/2

dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)

# Boundary condition functions and derivatives
def f0f(y):
    return y**2

def g0f(x):
    return x**2

bcf = (f0f, g0f)

def df0_dyf(y):
    return 2*y

def dg0_dxf(x):
    return 2*x

bcdf = (df0_dyf, dg0_dxf)

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

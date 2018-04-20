# Sample 2nd-order PDE BVP

# A reasonable solution can be found using the following settings:
# ALL DEFAULTS

# The equation is defined on the domain [[0,1],[0,1]], with the
# Dirichlet boundary conditions specified at x=0|1 and y=0|1.

# The analytical form of the equation is:

# G(x,y,Y,delY,deldelY) = x*y - Y = 0.

# The analytical solution is:

# Y(x,y) = x*y

# Define the original differential equation.
def Gf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return x*y - Y

# Partial of PDE wrt Y
def dG_dYf(xy, Y, delY, deldelY):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    ((d2Y_dxdx, d2Y_dxdy), (d2Y_dydx, d2Ydydy)) = deldelY
    return -1

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
    return 0

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
    return 0

dG_ddeldelYf = ((dG_d2Y_dxdxf, dG_d2Y_dxdyf),
                (dG_d2Y_dydxf, dG_d2Y_dydyf))

# Boundary conditions for x at 0 and 1
def f0f(y):
    return 0

def f1f(y):
    return y

# Boundary conditions for y at 0 and 1
def g0f(x):
    return 0

def g1f(x):
    return x

# Array of BC functions
bcf = ((f0f, g0f), (f1f, g1f))

# 1st derivative of BC functions for x at 0 and 1
def df0_dyf(y):
    return 0

def df1_dyf(y):
    return 1

# 1st derivative of BC functions for y at 0 and 1
def dg0_dxf(x):
    return 0

def dg1_dxf(x):
    return 1

# Array of BC 1st derivatives
bcdf = ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf))

# 2nd derivatives of BC functions for x at 0 and 1
def d2f0_dy2f(y):
    return 0

def d2f1_dy2f(y):
    return 0

# 2nd derivatives of BC functions for y at 0 and 1
def d2g0_dx2f(x):
    return 0

def d2g1_dx2f(x):
    return 0

# Array of BC 2nd derivatives
bcd2f = ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f))

# Define the analytical solution and its derivatives.
def Yaf(xy):
    (x, y) = xy
    return x*y

def dYa_dxf(xy):
    (x, y) = xy
    return y

def dYa_dyf(xy):
    (x, y) = xy
    return x

delYaf = (dYa_dxf, dYa_dyf)

def d2Ya_dxdxf(xy):
    return 0

def d2Ya_dxdyf(xy):
    return 1

def d2Ya_dydxf(xy):
    return 1

def d2Ya_dydyf(xy):
    return 0

deldelYaf = ((d2Ya_dxdxf, d2Ya_dxdyf),
             (d2Ya_dydxf, d2Ya_dydyf))

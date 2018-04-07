# Sample 2nd-order PDE BVP

# A reasonable solution can be found using the following settings:
# 

# The equation is defined on the domain [[0,1],[0,1]], with the
# Dirichlet boundary conditions specified at x=0|1 and y=0|1.

# The analytical form of the equation is:

# G(x,y,Y,dY/dx,dY/dy,d2Y_dx2,d2Y_dy2) = x*y - Y = 0.

# The analytical solution is:

# Y(x,y) = x*y

# Define the original differential equation.
def Gf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return x*y - Y

# Partial of PDE wrt Y
def dG_dYf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return -1

# Partial of PDE wrt gradient of Y
def dG_dY_dxf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return 0

def dG_dY_dyf(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return 0

dG_ddelYf = (dG_dY_dxf, dG_dY_dyf)

# Partial of PDE wrt del**2 of Y
def dG_d2Y_dx2f(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return 0

def dG_d2Y_dy2f(xy, Y, delY, del2Y):
    (x, y) = xy
    (dY_dx, dY_dy) = delY
    (d2Y_dx2, d2Y_dy2) = del2Y
    return 0

dG_ddel2Yf = (dG_d2Y_dx2f, dG_d2Y_dy2f)

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

def d2Ya_dx2f(xy):
    (x, y) = xy
    return 0

def d2Ya_dy2f(xy):
    (x, y) = xy
    return 0

del2Yaf = (d2Ya_dx2f, d2Ya_dy2f)

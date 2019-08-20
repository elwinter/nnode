"""
3-D diffusion PDE

The analytical form of the equation is:
  G(x,y,z,t,Y,delY,del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2) = 0

The equation is defined on the domain (x,y,z,t)=([0,1],[0,1],[0,1],[0,]). The
boundary conditions are:

Y(0,y,z,t) = 0
Y(1,y,z,t) = a*t*sin(pi*y)*sin(pi*z)
Y(x,0,z,t) = 0
Y(x,1,z,t) = 0
Y(x,y,0,t) = 0
Y(x,y,1,t) = 0
Y(x,y,z,0) = sin(pi*x)*sin(pi*y)*sin(pi*z)
"""


from math import cos, exp, pi, sin


# Diffusion coefficient
D = 0.1

# Boundary increase rate at x=1
a = 1.0


def Gf(xyzt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2)

def dG_dYf(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return 0

def dG_dY_dxf(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return 0

def dG_dY_dyf(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dy"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return 0

def dG_dY_dzf(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dz"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return 0

def dG_dY_dtf(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return 1

dG_ddelYf = [dG_dY_dxf, dG_dY_dyf, dG_dY_dzf, dG_dY_dtf]


def dG_d2Y_dx2f(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return -D

def dG_d2Y_dy2f(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dy2"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return -D

def dG_d2Y_dz2f(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dz2"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return -D

def dG_d2Y_dt2f(xyzt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
    return 0

dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dy2f, dG_d2Y_dz2f, dG_d2Y_dt2f]


def f0f(xyzt):
    """Boundary condition at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def f1f(xyzt):
    """Boundary condition at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return a*t*sin(pi*y)*sin(pi*z)

def g0f(xyzt):
    """Boundary condition at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def g1f(xyzt):
    """Boundary condition at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def h0f(xyzt):
    """Boundary condition at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def h1f(xyzt):
    """Boundary condition at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def Y0f(xyzt):
    """Boundary condition at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return sin(pi*x)*sin(pi*y)*sin(pi*z)

def Y1f(xyzt):
    """Boundary condition at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

bcf = [[f0f, f1f], [g0f, g1f], [h0f, h1f], [Y0f, Y1f]]


def df0_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def df0_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def df0_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def df0_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def df1_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def df1_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return a*pi*t*cos(pi*y)*sin(pi*z)

def df1_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return a*pi*t*sin(pi*y)*cos(pi*z)

def df1_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return a*sin(pi*y)*sin(pi*z)

def dg0_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg0_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg0_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg0_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg1_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg1_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg1_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dg1_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh0_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh0_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh0_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh0_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh1_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh1_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh1_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def dh1_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def dY0_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return pi*cos(pi*x)*sin(pi*y)*sin(pi*z)

def dY0_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return pi*sin(pi*x)*cos(pi*y)*sin(pi*z)

def dY0_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return pi*sin(pi*x)*sin(pi*y)*cos(pi*z)

def dY0_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return 0

def dY1_dxf(xyzt):
    """1st derivative of BC wrt x at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

def dY1_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

def dY1_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

def dY1_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

delbcf = [[[df0_dxf, df0_dyf, df0_dzf, df0_dtf], [df1_dxf, df1_dyf, df1_dzf, df1_dtf]],
          [[dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf]],
          [[dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf], [dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf]],
          [[dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf]]]


def d2f0_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2f0_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2f0_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2f0_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (0,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2f1_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2f1_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return -a*pi**2*t*sin(pi*y)*sin(pi*z)

def d2f1_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return -a*pi**2*t*sin(pi*y)*sin(pi*z)

def d2f1_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g0_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g0_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g0_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g0_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (x,0,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g1_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g1_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g1_dz2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g1_dt2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h0_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h0_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h0_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h0_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (x,y,0,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h1_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h1_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h1_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2h1_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (x,y,1,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2Y0_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

def d2Y0_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

def d2Y0_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

def d2Y0_dt2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return 0

def d2Y1_dx2f(xyzt):
    """2nd derivative of BC wrt x at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

def d2Y1_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

def d2Y1_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

def d2Y1_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (x,y,z,1) NOT USED"""
    (x, y, z, t) = xyzt
    return None

del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dz2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dz2f, d2f1_dt2f]],
           [[d2g0_dx2f, d2g0_dy2f, d2g0_dz2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dz2f, d2g1_dt2f]],
           [[d2h0_dx2f, d2h0_dy2f, d2h0_dz2f, d2h0_dt2f], [d2h1_dx2f, d2h1_dy2f, d2h1_dz2f, d2h1_dt2f]],
           [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dz2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dz2f, d2Y1_dt2f]]]


def Af(xyzt):
    """Optimized version of boundary condition function"""
    (x, y, z, t) = xyzt
    A = (a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    return A

def delAf(xyzt):
    """Optimized version of boundary condition function gradient"""
    (x, y, z, t) = xyzt
    dA_dx = (a*t + pi*(1 - t)*cos(pi*x))*sin(pi*y)*sin(pi*z)
    dA_dy = pi*cos(pi*y)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*z)
    dA_dz = pi*cos(pi*z)*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)
    dA_dt = (a*x - sin(pi*x))*sin(pi*y)*sin(pi*z)
    return [dA_dx, dA_dy, dA_dz, dA_dt]

def del2Af(xyzt):
    """Optimized version of boundary condition function Laplacian"""
    (x, y, z, t) = xyzt
    d2A_dx2 = pi**2*(t - 1)*sin(pi*x)*sin(pi*y)*sin(pi*z)
    d2A_dy2 = -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    d2A_dz2 = -pi**2*(a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)*sin(pi*z)
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2]


# def Yaf(xyzt):
#     """Analytical solution"""
#     (x, y, z, t) = xyzt
#     Ya = exp(-3*pi**2*D*t)*sin(pi*x)*sin(pi*y)*sin(pi*z)
#     return Ya

# def dYa_dxf(xyzt):
#     """Analytical x-gradient"""
#     (x, y, z, t) = xyzt
#     return exp(-3*pi**2*D*t)*pi*cos(pi*x)*sin(pi*y)*sin(pi*z)

# def dYa_dyf(xyzt):
#     """Analytical y-gradient"""
#     (x, y, z, t) = xyzt
#     return exp(-3*pi**2*D*t)*pi*sin(pi*x)*cos(pi*y)*sin(pi*z)

# def dYa_dzf(xyzt):
#     """Analytical z-gradient"""
#     (x, y, z, t) = xyzt
#     return exp(-3*pi**2*D*t)*pi*sin(pi*x)*sin(pi*y)*cos(pi*z)

# def dYa_dtf(xyzt):
#     """Analytical t-gradient"""
#     (x, y, z, t) = xyzt
#     return -3*exp(-3*pi**2*D*t)*pi**2*D*sin(pi*x)*sin(pi*y)*sin(pi*z)

# delYaf = [dYa_dxf, dYa_dyf, dYa_dzf, dYa_dtf]


# def d2Ya_dx2f(xyzt):
#     """Analytical x-Laplacian"""
#     (x, y, z, t) = xyzt
#     return -exp(-3*pi**2*D*t)*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

# def d2Ya_dy2f(xyzt):
#     """Analytical y-Laplacian"""
#     (x, y, z, t) = xyzt
#     return -exp(-3*pi**2*D*t)*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

# def d2Ya_dz2f(xyzt):
#     """Analytical z-Laplacian"""
#     (x, y, z, t) = xyzt
#     return -exp(-3*pi**2*D*t)*pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

# def d2Ya_dt2f(xyzt):
#     """Analytical t-Laplacian"""
#     (x, y, z, t) = xyzt
#     return 9*exp(-3*pi**2*D*t)*pi**4*D**2*sin(pi*x)*sin(pi*y)*sin(pi*z)

# del2Yaf = [d2Ya_dx2f, d2Ya_dy2f, d2Ya_dz2f, d2Ya_dt2f]


import numpy as np


if __name__ == '__main__':

    # Test values
    xyzt = [0.4, 0.5, 0.6, 0.7]

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    dG_ddelY_ref = [0, 0, 0, 1]
    dG_ddel2Y_ref = [-D, -D, -D, 0]
    bc_ref = [[0, 0.6657395614066075], [0, 0], [0, 0], [0.904508497187474, None]]
    delbc_ref = [[[0, 0, 0, 0], [0, 0, -0.6795638635539131, 0.9510565162951536]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0.9232909152452285, 0, -0.923290915245228, 0], [None, None, None, None]]]
    del2bc_ref = [[[0, 0, 0, 0], [0, -6.570586105237952, -6.570586105237952, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[-8.927141044664213, -8.927141044664213, -8.927141044664213, 0], [None, None, None, None]]]
    A_ref = 0.5376483737188851
    delA_ref = [0.9427268359801761, 0, -0.5488128199951336, -0.5240858906694122]
    del2A_ref = [-2.6781423133992637, -5.306376755494444, -5.306376755494444, 0]
    # Ya_ref = 0.1138378166982095
    # delYa_ref = [0.11620169660719469, 0, -0.11620169660719464, -0.3370602650085157]
    # del2Ya_ref = [-1.1235342166950524, -1.1235342166950524, -1.1235342166950524, 0.9979954424881178]

    # print("Testing differential equation.")
    # assert np.isclose(Gf(xyzt, Ya_ref, delYa_ref, del2Ya_ref), G_ref)

    # print("Testing differential equation Y-derivative.")
    # assert np.isclose(dG_dYf(xyzt, Ya_ref, delYa_ref, del2Ya_ref), dG_dY_ref)

    # print("Testing differential equation gradient-derivatives.")
    # for (i, f) in enumerate(dG_ddelYf):
    #     assert np.isclose(f(xyzt, Ya_ref, delYa_ref, del2Ya_ref),
    #                       dG_ddelY_ref[i])

    # print("Testing differential equation Laplacian-derivatives.")
    # for (i, f) in enumerate(dG_ddel2Yf):
    #     assert np.isclose(f(xyzt, Ya_ref, delYa_ref, del2Ya_ref),
    #                       dG_ddel2Y_ref[i])

    print("Testing boundary conditions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            if bc_ref[i][j] is None:
                assert f(xyzt) is None
            else:
                assert np.isclose(f(xyzt), bc_ref[i][j])

    print("Testing boundary condition gradients.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                if delbc_ref[i][j][k] is None:
                    assert f(xyzt) is None
                else:
                    assert np.isclose(f(xyzt), delbc_ref[i][j][k])

    print("Testing boundary condition Laplacians.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                if del2bc_ref[i][j][k] is None:
                    assert f(xyzt) is None
                else:
                    assert np.isclose(f(xyzt), del2bc_ref[i][j][k])

    print("Verifying BC continuity constraints.")
    assert np.isclose(f0f([0, 0, 0, 0]), g0f([0, 0, 0, 0]))
    assert np.isclose(f0f([0, 0, 0, 0]), h0f([0, 0, 0, 0]))
    assert np.isclose(f0f([0, 0, 0, 0]), Y0f([0, 0, 0, 0]))
    assert np.isclose(f1f([1, 0, 0, 0]), g0f([1, 0, 0, 0]))
    assert np.isclose(f1f([1, 0, 0, 0]), h0f([1, 0, 0, 0]))
    assert np.isclose(f1f([1, 0, 0, 0]), Y0f([1, 0, 0, 0]))
    assert np.isclose(f1f([1, 1, 0, 0]), g1f([1, 1, 0, 0]))
    assert np.isclose(f1f([1, 1, 0, 0]), h0f([1, 1, 0, 0]))
    assert np.isclose(f1f([1, 1, 0, 0]), Y0f([1, 1, 0, 0]))
    assert np.isclose(f0f([0, 1, 0, 0]), g1f([0, 1, 0, 0]))
    assert np.isclose(f0f([0, 1, 0, 0]), h0f([0, 1, 0, 0]))
    assert np.isclose(f0f([0, 1, 0, 0]), Y0f([0, 1, 0, 0]))
    assert np.isclose(f0f([0, 0, 1, 0]), g0f([0, 0, 1, 0]))
    assert np.isclose(f0f([0, 0, 1, 0]), h1f([0, 0, 1, 0]))
    assert np.isclose(f0f([0, 0, 1, 0]), Y0f([0, 0, 1, 0]))
    assert np.isclose(f1f([1, 0, 1, 0]), g0f([1, 0, 1, 0]))
    assert np.isclose(f1f([1, 0, 1, 0]), h1f([1, 0, 1, 0]))
    assert np.isclose(f1f([1, 0, 1, 0]), Y0f([1, 0, 1, 0]))
    assert np.isclose(f1f([1, 1, 1, 0]), g1f([1, 1, 1, 0]))
    assert np.isclose(f1f([1, 1, 1, 0]), h1f([1, 1, 1, 0]))
    assert np.isclose(f1f([1, 1, 1, 0]), Y0f([1, 1, 1, 0]))
    assert np.isclose(f0f([0, 1, 1, 0]), g1f([0, 1, 1, 0]))
    assert np.isclose(f0f([0, 1, 1, 0]), h1f([0, 1, 1, 0]))
    assert np.isclose(f0f([0, 1, 1, 0]), Y0f([0, 1, 1, 0]))
    # t=1 not used

    print("Testing optimized BC function.")
    assert np.isclose(Af(xyzt), A_ref)

    print("Testing optimized BC function gradient.")
    delA = delAf(xyzt)
    for i in range(len(delA_ref)):
        assert np.isclose(delA[i], delA_ref[i])

    print("Testing optimized BC function Laplacian.")
    del2A = del2Af(xyzt)
    for i in range(len(del2A_ref)):
        assert np.isclose(del2A[i], del2A_ref[i])

    # print("Testing analytical solution.")
    # assert np.isclose(Yaf(xyzt), Ya_ref)

    # print("Testing analytical solution gradient.")
    # for (i, f) in enumerate(delYaf):
    #     if delYa_ref[i] is None:
    #         assert f(xyzt) is None
    #     else:
    #         assert np.isclose(f(xyzt), delYa_ref[i])

    # print("Testing analytical solution Laplacian.")
    # for (i, f) in enumerate(del2Yaf):
    #     if del2Ya_ref[i] is None:
    #         assert f(xyzt) is None
    #     else:
    #         assert np.isclose(f(xyzt), del2Ya_ref[i])

################################################################################
"""
3-D diffusion PDE

The analytical form of the equation is:
  G(x,y,z,t,Y,delY,del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2) = 0

The equation is defined on the domain (x,y,z,t)=([0,1],[0,1],[0,1],[0,]). The boundary
conditions are:

Y(0,y,z,t) = 0
Y(1,y,z,t) = 0
Y(x,0,z,t) = 0
Y(x,1,z,t) = 0
Y(x,y,0,t) = 0
Y(x,y,1,t) = 0
Y(x,y,z,0) = 0
"""


# Diffusion coefficient
D = 0.1


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
    return 0

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
    return 0

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
    return 0

def df1_dzf(xyzt):
    """1st derivative of BC wrt z at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def df1_dtf(xyzt):
    """1st derivative of BC wrt t at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

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
    return 0

def dY0_dyf(xyzt):
    """1st derivative of BC wrt y at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return 0

def dY0_dzf(xyzt):
    """1st derivative of BC wrt zx at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return 0

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
    return 0

def d2f1_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (1,y,z,t)"""
    (x, y, z, t) = xyzt
    return 0

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
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,1,z,t)"""
    (x, y, z, t) = xyzt
    return 0

def d2g1_dt2f(xyzt):
    """2nd derivative of BC wrt t at (x,y,z,t) = (x,1,z,t)"""
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
    return 0

def d2Y0_dy2f(xyzt):
    """2nd derivative of BC wrt y at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return 0

def d2Y0_dz2f(xyzt):
    """2nd derivative of BC wrt z at (x,y,z,t) = (x,y,z,0)"""
    (x, y, z, t) = xyzt
    return 0

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


def Af(xt):
    """Optimized version of boundary condition function"""
    return 0

def delAf(xt):
    """Optimized version of boundary condition function gradient"""
    return [0, 0, 0, 0]

def del2Af(xt):
    """Optimized version of boundary condition function Laplacian"""
    return [0, 0, 0, 0]


def Yaf(xyzt):
    """Analytical solution"""
    (x, y, z, t) = xyzt
    Ya = 0
    return Ya

def dYa_dxf(xyzt):
    """Analytical x-gradient"""
    (x, y, z, t) = xyzt
    return 0

def dYa_dyf(xyzt):
    """Analytical y-gradient"""
    (x, y, z, t) = xyzt
    return 0

def dYa_dzf(xyzt):
    """Analytical z-gradient"""
    (x, y, z, t) = xyzt
    return 0

def dYa_dtf(xyzt):
    """Analytical t-gradient"""
    (x, y, z, t) = xyzt
    return 0

delYaf = [dYa_dxf, dYa_dyf, dYa_dzf, dYa_dtf]


def d2Ya_dx2f(xyzt):
    """Analytical x-Laplacian"""
    (x, y, z, t) = xyzt
    return 0

def d2Ya_dy2f(xyzt):
    """Analytical y-Laplacian"""
    (x, y, z, t) = xyzt
    return 0

def d2Ya_dz2f(xyzt):
    """Analytical z-Laplacian"""
    (x, y, z, t) = xyzt
    return 0

def d2Ya_dt2f(xyzt):
    """Analytical t-Laplacian"""
    (x, y, z, t) = xyzt
    return 0

del2Yaf = [d2Ya_dx2f, d2Ya_dy2f, d2Ya_dz2f, d2Ya_dt2f]


import numpy as np


if __name__ == '__main__':

    # Test values
    xyzt = [0.4, 0.5, 0.6, 0.7]

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    (dG_dY_dx_ref, dG_dY_dy_ref, dG_dY_dz_ref, dG_dY_dt_ref) = [0, 0, 0, 1]
    (dG_d2Y_dx2_ref, dG_d2Y_dy2_ref, dG_d2Y_dz2_ref, dG_d2Y_dt2_ref) = [-D, -D, -D, 0]
    bc_ref = [[0, 0], [0, 0], [0, 0], [0, None]]
    delbc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [None, None, None, None]]]
    del2bc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [0, 0, 0, 0]],
                  [[0, 0, 0, 0], [None, None, None, None]]]
    Ya_ref = 0
    delYa_ref = [0, 0, 0, 0]
    del2Ya_ref = [0, 0, 0, 0]

    print("Testing differential equation.")
    G = Gf(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(G, G_ref):
        print("ERROR: G = %s, vs ref %s" % (G, G_ref))

    print("Testing differential equation Y-derivative.")
    dG_dY = dG_dYf(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY, dG_dY_ref):
        print("ERROR: dG_dY = %s, vs ref %s" % (dG_dY, dG_dY_ref))

    print("Testing differential equation dY/dx-derivative.")
    dG_dY_dx = dG_dY_dxf(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dx, dG_dY_dx_ref):
        print("ERROR: dG_dY_dx = %s, vs ref %s" % (dG_dY_dx, dG_dY_dx_ref))

    print("Testing differential equation dY/dy-derivative.")
    dG_dY_dy = dG_dY_dyf(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dy, dG_dY_dy_ref):
        print("ERROR: dG_dY_dy = %s, vs ref %s" % (dG_dY_dy, dG_dY_dy_ref))

    print("Testing differential equation dY/dz-derivative.")
    dG_dY_dz = dG_dY_dzf(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dz, dG_dY_dz_ref):
        print("ERROR: dG_dY_dz = %s, vs ref %s" % (dG_dY_dz, dG_dY_dz_ref))

    print("Testing differential equation dY/dt-derivative.")
    dG_dY_dt = dG_dY_dtf(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_dY_dt, dG_dY_dt_ref):
        print("ERROR: dG_dY_dt = %s, vs ref %s" % (dG_dY_dt, dG_dY_dt_ref))

    print("Testing differential equation d2Y/dx2-derivative.")
    dG_d2Y_dx2 = dG_d2Y_dx2f(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dx2, dG_d2Y_dx2_ref):
        print("ERROR: dG_d2Y_dx2 = %s, vs ref %s" % (dG_d2Y_dx2, dG_d2Y_dx2_ref))

    print("Testing differential equation d2Y/dy2-derivative.")
    dG_d2Y_dy2 = dG_d2Y_dy2f(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dy2, dG_d2Y_dy2_ref):
        print("ERROR: dG_d2Y_dy2 = %s, vs ref %s" % (dG_d2Y_dy2, dG_d2Y_dy2_ref))

    print("Testing differential equation d2Y/dz2-derivative.")
    dG_d2Y_dz2 = dG_d2Y_dz2f(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dz2, dG_d2Y_dz2_ref):
        print("ERROR: dG_d2Y_dz2 = %s, vs ref %s" % (dG_d2Y_dz2, dG_d2Y_dz2_ref))

    print("Testing differential equation d2Y/dt2-derivative.")
    dG_d2Y_dt2 = dG_d2Y_dt2f(xyzt, Ya_ref, delYa_ref, del2Ya_ref)
    if not np.isclose(dG_d2Y_dt2, dG_d2Y_dt2_ref):
        print("ERROR: dG_d2Y_dt2 = %s, vs ref %s" % (dG_d2Y_dt2, dG_d2Y_dt2_ref))

    print("Testing boundary conditions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            bc = f(xyzt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                delbc = f(xyzt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                del2bc = f(xyzt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing analytical solution.")
    Ya = Yaf(xyzt)
    if not np.isclose(Ya, Ya_ref):
        print("ERROR: Ya = %s, vs ref %s" % (Ya, Ya_ref))

    print("Testing analytical solution gradient.")
    for (i, f) in enumerate(delYaf):
        delYa = f(xyzt)
        if ((delYa_ref[i] is not None and not np.isclose(delYa, delYa_ref[i]))
                or (delYa_ref[i] is None and delYa is not None)):
                print("ERROR: delYa[%d] = %s, vs ref %s" % (i, delYa, delYa_ref[i]))

    print("Testing analytical solution Laplacian.")
    for (i, f) in enumerate(del2Yaf):
        del2Ya = f(xyzt)
        if ((del2Ya_ref[i] is not None and not np.isclose(del2Ya, del2Ya_ref[i]))
                or (del2Ya_ref[i] is None and del2Ya is not None)):
                print("ERROR: del2Ya[%d] = %s, vs ref %s" % (i, del2Ya, del2Ya_ref[i]))

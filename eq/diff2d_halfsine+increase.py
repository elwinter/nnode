"""
2-D diffusion PDE

The analytical form of the equation is:
  G(x,y,t,Y,delY,del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2) = 0

The equation is defined on the domain (x,y,t)=([0,1],[0,1],[0,]). The
boundary conditions are:

Y(0,y,t) = C = 0
Y(1,y,t) = C = a*t*sin(pi*y)
Y(x,0,t) = C = 0
Y(x,1,t) = C = 0
Y(x,y,0) = C = sin(pi*x)*sin(pi*y)/2
"""


from math import cos, pi, sin


# Diffusion coefficient
D = 0.1

# Boundary increase rate at x=1
a = 0.1


def Gf(xyt, Y, delY, del2Y):
    """The differential equation in standard form"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2)

def dG_dYf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt Y"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dxf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dx"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dyf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dy"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

def dG_dY_dtf(xyt, Y, delY, del2Y):
    """Partial of PDE wrt dY/dt"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 1

dG_ddelYf = [dG_dY_dxf, dG_dY_dyf, dG_dY_dtf]


def dG_d2Y_dx2f(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dx2"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return -D

def dG_d2Y_dy2f(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dy2"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return -D

def dG_d2Y_dt2f(xyt, Y, delY, del2Y):
    """Partial of PDE wrt d2Y/dt2"""
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return 0

dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dy2f, dG_d2Y_dt2f]


def f0f(xyt):
    """Boundary condition at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def f1f(xyt):
    """Boundary condition at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return a*t*sin(pi*y)

def g0f(xyt):
    """Boundary condition at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def g1f(xyt):
    """Boundary condition at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def Y0f(xyt):
    """Boundary condition at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return sin(pi*x)*sin(pi*y)/2

def Y1f(xyt):
    """Boundary condition at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

bcf = [[f0f, f1f], [g0f, g1f], [Y0f, Y1f]]


def df0_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def df0_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def df0_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def df1_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def df1_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return a*pi*t*cos(pi*y)

def df1_dtf(xyt):
    """1st derivative of BC wrt z at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return a*sin(pi*y)

def dg0_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def dg0_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def dg0_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def dg1_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def dg1_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def dg1_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def dY0_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return pi/2*cos(pi*x)*sin(pi*y)

def dY0_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return pi/2*sin(pi*x)*cos(pi*y)

def dY0_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return 0

def dY1_dxf(xyt):
    """1st derivative of BC wrt x at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def dY1_dyf(xyt):
    """1st derivative of BC wrt y at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def dY1_dtf(xyt):
    """1st derivative of BC wrt t at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

delbcf = [[[df0_dxf, df0_dyf, df0_dtf], [df1_dxf, df1_dyf, df1_dtf]],
          [[dg0_dxf, dg0_dyf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dtf]],
          [[dY0_dxf, dY0_dyf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dtf]]]


def d2f0_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f0_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f0_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (0,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f1_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def d2f1_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return -a*pi**2*t*sin(pi*y)

def d2f1_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (1,y,t)"""
    (x, y, t) = xyt
    return 0

def d2g0_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def d2g0_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def d2g0_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,0,t)"""
    (x, y, t) = xyt
    return 0

def d2g1_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def d2g1_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def d2g1_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,1,t)"""
    (x, y, t) = xyt
    return 0

def d2Y0_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return -pi**2/2*sin(pi*x)*sin(pi*y)

def d2Y0_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return -pi**2/2*sin(pi*x)*sin(pi*y)

def d2Y0_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,y,0)"""
    (x, y, t) = xyt
    return 0

def d2Y1_dx2f(xyt):
    """2nd derivative of BC wrt x at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def d2Y1_dy2f(xyt):
    """2nd derivative of BC wrt y at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

def d2Y1_dt2f(xyt):
    """2nd derivative of BC wrt t at (x,y,t) = (x,y,1) NOT USED"""
    (x, y, t) = xyt
    return None

del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dt2f]],
           [[d2g0_dx2f, d2g0_dy2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dt2f]],
           [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dt2f]]]


def Af(xyt):
    """Optimized version of boundary condition function"""
    (x, y, t) = xyt
    A = 1/2*(2*a*t*x + (1 - t)*sin(pi*x))*sin(pi*y)
    return A

def delAf(xyt):
    """Optimized version of boundary condition function gradient"""
    (x, y, t) = xyt
    dA_dx = (2*a*t + pi*(1 - t)*cos(pi*x))*sin(pi*y)/2
    dA_dy = pi*cos(pi*y)*(2*a*t*x + (1 - t)*sin(pi*x))/2
    dA_dt = (2*a*x - sin(pi*x))*sin(pi*y)/2
    return [dA_dx, dA_dy, dA_dt]

def del2Af(xyt):
    """Optimized version of boundary condition function Laplacian"""
    (x, y, t) = xyt
    d2A_dx2 = pi**2*(t - 1)*sin(pi*x)*sin(pi*y)/2
    d2A_dy2 = pi**2*(-2*a*t*x + (t - 1)*sin(pi*x))*sin(pi*y)/2
    d2A_dt2 = 0
    return [d2A_dx2, d2A_dy2, d2A_dt2]


import numpy as np

if __name__ == '__main__':

    # Test values
    xyt = [0.4, 0.5, 0.6]

    # Reference values for tests.
    G_ref = 0.561
    dG_dY_ref = 0
    dG_ddelY_ref = [0, 0, 1]
    dG_ddel2Y_ref = [-D, -D, 0]
    bc_ref = [[0, 0.06],
              [0, 0],
              [0.4755282581475767, None]]
    delbc_ref = [[[0, 0, 0], [0, 0, 0.1]],
                 [[0, 0, 0], [0, 0, 0]],
                 [[0.4854027596813666, 0, 0], [None, None, None]]]
    del2bc_ref = [[[0, 0, 0], [0, -0.5921762640653615, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[-4.69327578945568, -4.69327578945568, 0], [None, None, None]]]
    A_ref = 0.2142113032590307
    delA_ref = [0.2541611038725466, 0, -0.4355282581475768]
    del2A_ref = [-1.8773103157822721, -2.1141808214084166, 0]
    Ya_ref = 0.11  # Random
    delYa_ref = [0.22, 0.33, 0.44]
    del2Ya_ref = [-0.55, -0.66, -0.77]
 
    print("Testing differential equation.")
    assert np.isclose(Gf(xyt, Ya_ref, delYa_ref, del2Ya_ref), G_ref)

    print("Testing differential equation Y-derivative.")
    assert np.isclose(dG_dYf(xyt, Ya_ref, delYa_ref, del2Ya_ref), dG_dY_ref)

    print("Testing differential equation gradient-derivatives.")
    for (i, f) in enumerate(dG_ddelYf):
        assert np.isclose(f(xyt, Ya_ref, delYa_ref, del2Ya_ref),
                          dG_ddelY_ref[i])

    print("Testing differential equation Laplacian-derivatives.")
    for (i, f) in enumerate(dG_ddel2Yf):
        assert np.isclose(f(xyt, Ya_ref, delYa_ref, del2Ya_ref),
                          dG_ddel2Y_ref[i])

    print("Testing boundary conditions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            if bc_ref[i][j] is None:
                assert f(xyt) is None
            else:
                assert np.isclose(f(xyt), bc_ref[i][j])

    print("Testing boundary condition gradients.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                if delbc_ref[i][j][k] is None:
                    assert f(xyt) is None
                else:
                    assert np.isclose(f(xyt), delbc_ref[i][j][k])

    print("Testing boundary condition Laplacians.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                if del2bc_ref[i][j][k] is None:
                    assert f(xyt) is None
                else:
                    assert np.isclose(f(xyt), del2bc_ref[i][j][k])

    print("Verifying BC continuity constraints.")
    assert np.isclose(f0f([0, 0, 0]), g0f([0, 0, 0]))
    assert np.isclose(f0f([0, 0, 0]), Y0f([0, 0, 0]))
    assert np.isclose(f1f([1, 0, 0]), g0f([1, 0, 0]))
    assert np.isclose(f1f([1, 0, 0]), Y0f([1, 0, 0]))
    assert np.isclose(f1f([1, 1, 0]), g1f([1, 1, 0]))
    assert np.isclose(f1f([1, 1, 0]), Y0f([1, 1, 0]))
    assert np.isclose(f0f([0, 1, 0]), g1f([0, 1, 0]))
    assert np.isclose(f0f([0, 1, 0]), Y0f([0, 1, 0]))
    # t=1 not used

    print("Testing optimized BC function.")
    assert np.isclose(Af(xyt), A_ref)

    print("Testing optimized BC function gradient.")
    delA = delAf(xyt)
    for i in range(len(delA_ref)):
        assert np.isclose(delA[i], delA_ref[i])

    print("Testing optimized BC function Laplacian.")
    del2A = del2Af(xyt)
    for i in range(len(del2A_ref)):
        assert np.isclose(del2A[i], del2A_ref[i])

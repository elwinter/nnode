"""
2-D diffusion PDE

The analytical form of the equation is:
  G(x,y,t,Y,delY,del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2) = 0

The equation is defined on the domain (x,y,t)=([0,1],[0,1],[0,1]). The
initial profile is:

Y(x,y,0) = sin(pi*x)*sin(pi*y)
"""


from math import cos, pi, sin
import numpy as np


# Diffusion coefficient
D = 1


def Gf(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2)

def dG_dYf(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return 0

def dG_dY_dxf(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return 0

def dG_dY_dyf(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return 0

def dG_dY_dtf(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return 1

dG_ddelYf = [dG_dY_dxf, dG_dY_dyf, dG_dY_dtf]


def dG_d2Y_dx2f(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return -D

def dG_d2Y_dy2f(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return -D

def dG_d2Y_dt2f(xyt, Y, delY, del2Y):
    (x, y, t) = xyt
    (dY_dx, dY_dy, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2t_dt2) = del2Y
    return 0

dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dy2f, dG_d2Y_dt2f]


# Boundary condition functions

def f0f(xyt):
    (x, y, t) = xyt
    return 0

def f1f(xyt):
    (x, y, t) = xyt
    return 0

def g0f(xyt):
    (x, y, t) = xyt
    return 0

def g1f(xyt):
    (x, y, t) = xyt
    return 0

def h0f(xyt):
    (x, y, t) = xyt
    return sin(pi*x)*sin(pi*y)

def h1f(xyt):
    (x, y, t) = xyt
    return None

bcf = [[f0f, f1f], [g0f, g1f], [h0f, h1f]]


# Gradients of boundary condition functions

def df0_dxf(xyt):
    (x, y, t) = xyt
    return 0

def df0_dyf(xyt):
    (x, y, t) = xyt
    return 0

def df0_dtf(xyt):
    (x, y, t) = xyt
    return 0

def df1_dxf(xyt):
    (x, y, t) = xyt
    return 0

def df1_dyf(xyt):
    (x, y, t) = xyt
    return 0

def df1_dtf(xyt):
    (x, y, t) = xyt
    return 0

def dg0_dxf(xyt):
    (x, y, t) = xyt
    return 0

def dg0_dyf(xyt):
    (x, y, t) = xyt
    return 0

def dg0_dtf(xyt):
    (x, y, t) = xyt
    return 0

def dg1_dxf(xyt):
    (x, y, t) = xyt
    return 0

def dg1_dyf(xyt):
    (x, y, t) = xyt
    return 0

def dg1_dtf(xyt):
    (x, y, t) = xyt
    return 0

def dh0_dxf(xyt):
    (x, y, t) = xyt
    return pi*cos(pi*x)*sin(pi*y)

def dh0_dyf(xyt):
    (x, y, t) = xyt
    return pi*sin(pi*x)*cos(pi*y)

def dh0_dtf(xyt):
    (x, y, t) = xyt
    return 0

def dh1_dxf(xyt):
    (x, y, t) = xyt
    return None

def dh1_dyf(xyt):
    (x, y, t) = xyt
    return None

def dh1_dtf(xyt):
    (x, y, t) = xyt
    return None

delbcf = [[[df0_dxf, df0_dyf, df0_dtf], [df1_dxf, df1_dyf, df1_dtf]],
          [[dg0_dxf, dg0_dyf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dtf]],
          [[dh0_dxf, dh0_dyf, dh0_dtf], [dh1_dxf, dh1_dyf, dh1_dtf]]]


# Laplacians of boundary condition functions

def d2f0_dx2f(xyt):
    (x, y, t) = xyt
    return 0

def d2f0_dy2f(xyt):
    (x, y, t) = xyt
    return 0

def d2f0_dt2f(xyt):
    (x, y, t) = xyt
    return 0

def d2f1_dx2f(xyt):
    (x, y, t) = xyt
    return 0

def d2f1_dy2f(xyt):
    (x, y, t) = xyt
    return 0

def d2f1_dt2f(xyt):
    (x, y, t) = xyt
    return 0

def d2g0_dx2f(xyt):
    (x, y, t) = xyt
    return 0

def d2g0_dy2f(xyt):
    (x, y, t) = xyt
    return 0

def d2g0_dt2f(xyt):
    (x, y, t) = xyt
    return 0

def d2g1_dx2f(xyt):
    (x, y, t) = xyt
    return 0

def d2g1_dy2f(xyt):
    (x, y, t) = xyt
    return 0

def d2g1_dt2f(xyt):
    (x, y, t) = xyt
    return 0

def d2h0_dx2f(xyt):
    (x, y, t) = xyt
    return 0

def d2h0_dy2f(xyt):
    (x, y, t) = xyt
    return 0

def d2h0_dt2f(xyt):
    (x, y, t) = xyt
    return 0

def d2h1_dx2f(xyt):
    (x, y, t) = xyt
    return None

def d2h1_dy2f(xyt):
    (x, y, t) = xyt
    return None

def d2h1_dt2f(xyt):
    (x, y, t) = xyt
    return None

del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dt2f]],
           [[d2g0_dx2f, d2g0_dy2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dt2f]],
           [[d2h0_dx2f, d2h0_dy2f, d2h0_dt2f], [d2h1_dx2f, d2h1_dy2f, d2h1_dt2f]]]


# Analytical solution

# def Yaf(xyt):
#     (x, y, t) = xyt
#     return 0


# Analytical gradient

# def dYa_dxf(xyt):
#     (x, y, t) = xyt
#     return 0

# def dYa_dyf(xyt):
#     (x, y, t) = xyt
#     return 0

# def dYa_dtf(xyt):
#     (x, y, t) = xyt
#     return 0

# delYaf = [dYa_dxf, dYa_dyf, dYa_dtf]


# Analytical Laplacian

# def d2Ya_dx2f(xyt):
#     (x, y, t) = xyt
#     return 0

# def d2Ya_dy2f(xyt):
#     (x, y, t) = xyt
#     return 0

# def d2Ya_dt2f(xyt):
#     (x, y, t) = xyt
#     return 0

# del2Yaf = [d2Ya_dx2f, d2Ya_dy2f, d2Ya_dt2f]


if __name__ == '__main__':

    # Test values
    xyt_ref = [0, 0, 0]
    Y_ref = 0
    delY_ref = [0, 0, 0]
    del2Y_ref = [0, 0, 0]

    # Reference values for tests.
    G_ref = 0
    dG_dY_ref = 0
    dG_ddelY_ref = [0, 0, 1]
    dG_ddel2Y_ref = [-D, -D, 0]
    bc_ref = [[0, 0], [0, 0], [0, None]]
    delbc_ref = [[[0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [None, None, None]]]
    del2bc_ref = [[[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [None, None, None]]]

    print("Testing differential equation.")
    G = Gf(xyt_ref, Y_ref, delY_ref, del2Y_ref)
    if not np.isclose(G, G_ref):
        print("ERROR: G = %s, vs ref %s" % (G, G_ref))

    print("Testing differential equation Y-derivative.")
    dG_dY = dG_dYf(xyt_ref, Y_ref, delY_ref, del2Y_ref)
    if not np.isclose(dG_dY, dG_dY_ref):
        print("ERROR: dG_dY = %s, vs ref %s" % (dG_dY, dG_dY_ref))

    print("Testing differential equation delY-derivatives.")
    for (i, f) in enumerate(dG_ddelYf):
        dG_dY_dxi = f(xyt_ref, Y_ref, delY_ref, del2Y_ref)
        if not np.isclose(dG_dY_dxi, dG_ddelY_ref[i]):
            print("ERROR: dG_ddelY[%d] = %s, vs ref %s" % (i, dG_dY_dxi, dG_ddelY_ref[i]))

    print("Testing differential equation del2Y-derivatives.")
    for (i, f) in enumerate(dG_ddel2Yf):
        dG_d2Y_dx2i = f(xyt_ref, Y_ref, delY_ref, del2Y_ref)
        if not np.isclose(dG_d2Y_dx2i, dG_ddel2Y_ref[i]):
            print("ERROR: dG_ddel2Y[%d] = %s, vs ref %s" % (i, dG_dY_dxi, dG_ddel2Y_ref[i]))

    print("Testing BC functions.")
    for i in range(len(bcf)):
        for (j, f) in enumerate(bcf[i]):
            bc = f(xyt_ref)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]) )

    print("Testing BC gradient functions.")
    for i in range(len(delbcf)):
        for j in range(len(delbcf[i])):
            for (k, f) in enumerate(delbcf[i][j]):
                delbck = f(xyt_ref)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbck, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbck is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbck, delbc_ref[i][j][k]) )

    print("Testing BC Laplacian functions.")
    for i in range(len(del2bcf)):
        for j in range(len(del2bcf[i])):
            for (k, f) in enumerate(del2bcf[i][j]):
                del2bck = f(xyt_ref)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bck, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bck is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bck, del2bc_ref[i][j][k]) )

    print('Verifying BC continuity.')
    assert np.isclose(f0f([0,0,0]), g0f([0,0,0]))
    assert np.isclose(f0f([0,0,0]), h0f([0,0,0]))
    assert np.isclose(g0f([1,0,0]), f1f([1,0,0]))
    assert np.isclose(g0f([1,0,0]), h0f([1,0,0]))
    assert np.isclose(f1f([1,1,0]), g1f([1,1,0]))
    assert np.isclose(f1f([1,1,0]), h0f([1,1,0]))
    assert np.isclose(g1f([0,1,0]), f0f([0,1,0]))
    assert np.isclose(g1f([0,1,0]), h0f([0,1,0]))
    assert np.isclose(f0f([0,0,1]), g0f([0,0,1]))
    assert np.isclose(g0f([1,0,1]), f1f([1,0,1]))
    assert np.isclose(f1f([1,1,1]), g1f([1,1,1]))
    assert np.isclose(g1f([0,1,1]), f0f([0,1,1]))
    assert np.isclose(h0f([0,0,0]), f0f([0,0,0]))
    assert np.isclose(h0f([1,0,0]), f1f([1,0,0]))
    assert np.isclose(h0f([1,1,0]), f1f([1,1,0]))
    assert np.isclose(h0f([0,1,0]), f0f([0,1,0]))
    # assert np.isclose(h1f([0,0,1]), f0f([0,0,1]))
    # assert np.isclose(h1f([1,0,1]), f1f([1,0,1]))
    # assert np.isclose(h1f([1,1,1]), f1f([1,1,1]))
    # assert np.isclose(h1f([0,1,1]), f0f([0,1,1]))

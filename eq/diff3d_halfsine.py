################################################################################
"""
3-D diffusion PDE

The analytical form of the equation is:
  G(x,y,t,Y,delY,del2Y) = dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2) = 0

The equation is defined on the domain (x,y,z,t)=([0,1],[0,1],[0,1],[0,]). The
initial profile is:

Y(x,y,z,0) = sin(pi*x)*sin(pi*y)*sin(pi*z)/3
"""


from math import cos, exp, pi, sin
import numpy as np


# Diffusion coefficient
D = 0.1


def Gf(xyzt, Y, delY, del2Y):
    (x, y, z, t) = xyzt
    (dY_dx, dY_dy, dY_dz, dY_dt) = delY
    (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2Y_dt2) = del2Y
    return dY_dt - D*(d2Y_dx2 + d2Y_dy2 + d2Y_dz2)

# def dG_dYf(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return 0

# def dG_dY_dxf(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return 0

# def dG_dY_dyf(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return 0

# def dG_dY_dzf(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return 0

# def dG_dY_dtf(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return 1

# dG_ddelYf = [dG_dY_dxf, dG_dY_dyf, dG_dY_dzf, dG_dY_dtf]


# def dG_d2Y_dx2f(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return -D

# def dG_d2Y_dy2f(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return -D

# def dG_d2Y_dz2f(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return -D

# def dG_d2Y_dt2f(xyzt, Y, delY, del2Y):
#     (x, y, z, t) = xyzt
#     (dY_dx, dY_dy, dY_dz, dY_dt) = delY
#     (d2Y_dx2, d2Y_dy2, d2Y_dz2, d2t_dt2) = del2Y
#     return 0

# dG_ddel2Yf = [dG_d2Y_dx2f, dG_d2Y_dy2f, dG_d2Y_dz2f, dG_d2Y_dt2f]


# # Boundary condition functions

# def f0f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def f1f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def g0f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def g1f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def h0f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def h1f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def Y0f(xyzt):
#     (x, y, z, t) = xyzt
#     return sin(pi*x)*sin(pi*y)*sin(pi*z)/3

# def Y1f(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# bcf = [[f0f, f1f], [g0f, g1f], [h0f, h1f], [Y0f, Y1f]]


# # Gradients of boundary condition functions

# def df0_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df0_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df0_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df0_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df1_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df1_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df1_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def df1_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0


# def dg0_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg0_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg0_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg0_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg1_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg1_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg1_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dg1_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0


# def dh0_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh0_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh0_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh0_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh1_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh1_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh1_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dh1_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dY0_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return pi*cos(pi*x)*sin(pi*y)*sin(pi*z)/3

# def dY0_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return pi*sin(pi*x)*cos(pi*y)*sin(pi*z)/3

# def dY0_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return pi*sin(pi*x)*sin(pi*y)*cos(pi*z)/3

# def dY0_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def dY1_dxf(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# def dY1_dyf(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# def dY1_dzf(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# def dY1_dtf(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# delbcf = [[[df0_dxf, df0_dyf, df0_dzf, df0_dtf], [df1_dxf, df1_dyf, df1_dzf, df1_dtf]],
#           [[dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf]],
#           [[dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf], [dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf]],
#           [[dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf]]]


# # Laplacians of boundary condition functions

# def d2f0_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f0_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f0_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f0_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f1_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f1_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f1_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2f1_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g0_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g0_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g0_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g0_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g1_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g1_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g1_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2g1_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h0_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h0_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h0_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h0_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h1_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h1_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h1_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2h1_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2Y0_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)/3

# def d2Y0_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)/3

# def d2Y0_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return -pi**2*sin(pi*x)*sin(pi*y)*sin(pi*z)/3

# def d2Y0_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return 0

# def d2Y1_dx2f(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# def d2Y1_dy2f(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# def d2Y1_dz2f(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# def d2Y1_dt2f(xyzt):
#     (x, y, z, t) = xyzt
#     return None

# del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dz2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dz2f, d2f1_dt2f]],
#            [[d2g0_dx2f, d2g0_dy2f, d2g0_dz2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dz2f, d2g1_dt2f]],
#            [[d2h0_dx2f, d2h0_dy2f, d2h0_dz2f, d2h0_dt2f], [d2h1_dx2f, d2h1_dy2f, d2h1_dz2f, d2h1_dt2f]],
#            [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dz2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dz2f, d2Y1_dt2f]]]


# Analytical solution

# def Yaf(xyzt):
#     (x, y, z, t) = xyzt
#     Ya = exp(-3*pi**2*D*t)*sin(pi*x)*sin(pi*y)*sin(pi*z)/3
#     return Ya

# Analytical gradient

# # def dYa_dxf(xyt):
# #     (x, y, t) = xyt
# #     return 0

# # def dYa_dyf(xyt):
# #     (x, y, t) = xyt
# #     return 0

# # def dYa_dtf(xyt):
# #     (x, y, t) = xyt
# #     return 0

# # delYaf = [dYa_dxf, dYa_dyf, dYa_dtf]


# # Analytical Laplacian

# # def d2Ya_dx2f(xyt):
# #     (x, y, t) = xyt
# #     return 0

# # def d2Ya_dy2f(xyt):
# #     (x, y, t) = xyt
# #     return 0

# # def d2Ya_dt2f(xyt):
# #     (x, y, t) = xyt
# #     return 0

# # del2Yaf = [d2Ya_dx2f, d2Ya_dy2f, d2Ya_dt2f]


if __name__ == '__main__':

    # Test values
    xyzt_ref = [0, 0, 0, 0]
    Y_ref = 0
    delY_ref = [0, 0, 0, 0]
    del2Y_ref = [0, 0, 0, 0]

    # Reference values for tests.
    G_ref = 0
    # dG_dY_ref = 0
    # dG_ddelY_ref = [0, 0, 0, 1]
    # dG_ddel2Y_ref = [-D, -D, -D, 0]
    # bc_ref = [[0, 0], [0, 0], [0, 0], [0, None]]
    # delbc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
    #              [[0, 0, 0, 0], [0, 0, 0, 0]],
    #              [[0, 0, 0, 0], [0, 0, 0, 0]],
    #              [[0, 0, 0, 0], [None, None, None, None]]]
    # del2bc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
    #               [[0, 0, 0, 0], [0, 0, 0, 0]],
    #               [[0, 0, 0, 0], [0, 0, 0, 0]],
    #               [[0, 0, 0, 0], [None, None, None, None]]]
    # Ya_ref = 0.0379459

    print("Testing differential equation.")
    G = Gf(xyzt_ref, Y_ref, delY_ref, del2Y_ref)
    if not np.isclose(G, G_ref):
        print("ERROR: G = %s, vs ref %s" % (G, G_ref))

    # print("Testing differential equation Y-derivative.")
    # dG_dY = dG_dYf(xyzt_ref, Y_ref, delY_ref, del2Y_ref)
    # if not np.isclose(dG_dY, dG_dY_ref):
    #     print("ERROR: dG_dY = %s, vs ref %s" % (dG_dY, dG_dY_ref))

    # print("Testing differential equation delY-derivatives.")
    # for (i, f) in enumerate(dG_ddelYf):
    #     dG_dY_dxi = f(xyzt_ref, Y_ref, delY_ref, del2Y_ref)
    #     if not np.isclose(dG_dY_dxi, dG_ddelY_ref[i]):
    #         print("ERROR: dG_ddelY[%d] = %s, vs ref %s" % (i, dG_dY_dxi, dG_ddelY_ref[i]))

    # print("Testing differential equation del2Y-derivatives.")
    # for (i, f) in enumerate(dG_ddel2Yf):
    #     dG_d2Y_dx2i = f(xyzt_ref, Y_ref, delY_ref, del2Y_ref)
    #     if not np.isclose(dG_d2Y_dx2i, dG_ddel2Y_ref[i]):
    #         print("ERROR: dG_ddel2Y[%d] = %s, vs ref %s" % (i, dG_dY_dxi, dG_ddel2Y_ref[i]))

    # print("Testing BC functions.")
    # for i in range(len(bcf)):
    #     for (j, f) in enumerate(bcf[i]):
    #         bc = f(xyzt_ref)
    #         if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
    #             or (bc_ref[i][j] is None and bc is not None)):
    #             print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]) )

    # print("Testing BC gradient functions.")
    # for i in range(len(delbcf)):
    #     for j in range(len(delbcf[i])):
    #         for (k, f) in enumerate(delbcf[i][j]):
    #             delbck = f(xyzt_ref)
    #             if ((delbc_ref[i][j][k] is not None and not np.isclose(delbck, delbc_ref[i][j][k]))
    #                 or (delbc_ref[i][j][k] is None and delbck is not None)):
    #                 print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbck, delbc_ref[i][j][k]) )

    # print("Testing BC Laplacian functions.")
    # for i in range(len(del2bcf)):
    #     for j in range(len(del2bcf[i])):
    #         for (k, f) in enumerate(del2bcf[i][j]):
    #             del2bck = f(xyzt_ref)
    #             if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bck, del2bc_ref[i][j][k]))
    #                 or (del2bc_ref[i][j][k] is None and del2bck is not None)):
    #                 print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bck, del2bc_ref[i][j][k]) )

    # print('Verifying BC continuity.')
    # assert np.isclose(f0f([0,0,0,0]), g0f([0,0,0,0]))
    # assert np.isclose(f0f([0,0,0,0]), h0f([0,0,0,0]))
    # assert np.isclose(f0f([0,0,0,0]), Y0f([0,0,0,0]))

    # assert np.isclose(f1f([1,0,0,0]), g0f([1,0,0,0]))
    # assert np.isclose(f1f([1,0,0,0]), h0f([1,0,0,0]))
    # assert np.isclose(f1f([1,0,0,0]), Y0f([1,0,0,0]))

    # assert np.isclose(f1f([1,1,0,0]), g1f([1,1,0,0]))
    # assert np.isclose(f1f([1,1,0,0]), h0f([1,1,0,0]))
    # assert np.isclose(f1f([1,1,0,0]), Y0f([1,1,0,0]))

    # assert np.isclose(f0f([0,1,0,0]), g1f([0,1,0,0]))
    # assert np.isclose(f0f([0,1,0,0]), h0f([0,1,0,0]))
    # assert np.isclose(f0f([0,1,0,0]), Y0f([0,1,0,0]))

    # assert np.isclose(f0f([0,0,1,0]), g0f([0,0,1,0]))
    # assert np.isclose(f0f([0,0,1,0]), h1f([0,0,1,0]))
    # assert np.isclose(f0f([0,0,1,0]), Y0f([0,0,1,0]))

    # assert np.isclose(f1f([1,0,1,0]), g0f([1,0,1,0]))
    # assert np.isclose(f1f([1,0,1,0]), h1f([1,0,1,0]))
    # assert np.isclose(f1f([1,0,1,0]), Y0f([1,0,1,0]))

    # assert np.isclose(f1f([1,1,1,0]), g1f([1,1,1,0]))
    # assert np.isclose(f1f([1,1,1,0]), h1f([1,1,1,0]))
    # assert np.isclose(f1f([1,1,1,0]), Y0f([1,1,1,0]))

    # assert np.isclose(f0f([0,1,1,0]), g1f([0,1,1,0]))
    # assert np.isclose(f0f([0,1,1,0]), h1f([0,1,1,0]))
    # assert np.isclose(f0f([0,1,1,0]), Y0f([0,1,1,0]))

    # print("Testing analytical solution.")
    # Ya = Yaf([0.4, 0.5, 0.6, 0.7])
    # if not np.isclose(Ya, Ya_ref):
    #     print("ERROR: Ya = %s, vs ref %s" % (Ya, Ya_ref))

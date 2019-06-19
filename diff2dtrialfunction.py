################################################################################
"""
Diff2DTrialFunction - Class implementing the trial function for 2-D diffusion
problems

The trial function takes the form:

Yt(x, y, t) = A(x, y, t) + P(x, y, t)N(x, y, t, p)

where:

A(x, y, t) = boundary condition function that reduces to BC at boundaries
P(x, y, t) = network coefficient function that vanishes at boundaries
N(x, y, t, p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff2DTrialFunction object
        Yt_obj = Diff2DTrialFunction(bcf, delbcf, del2bcf)

    Compute the value of the trial function at a given point
        Yt = Yt_obj.Ytf([x, y. t], N)

    Compute the value of the boundary condition function at a given point
        A = Yt_obj.Af([x, y, t])

Notes:
    Variables that end in 'f' are usually functions or arrays of functions.

Attributes:
    bcf - 3x2 array of BC functions at (x,y,t)=0|1
    delbcf - 3x2x3 array of BC gradient functions at (x,y,t)=0|1
    del2bcf - 3x2x3 array of BC Laplacian component functions at (x,y,t)=0|1

Methods:
    Af([x, y, t]) - Compute boundary condition function at [x, y, t]

    delAf([x, y, t]) - Compute boundary condition function gradient at
        [x, y, t]

    del2Af([x, y, t]) - Compute boundary condition function Laplacian
        components at [x, y, t]

    Pf([x, y, t]) - Compute network coefficient function at [x, y, t]

    delPf([x, y, t]) - Compute network coefficient function gradient at
        [x, y, t]

    del2Pf([x, y, t]) - Compute network coefficient function Laplacian
        components at [x, y, t]

    Ytf([x, y, t], N) - Compute trial function at [x, y, t] with network
        output N

    delYtf([x, y, t], N, delN) - Compute trial function gradient at [x, y, t]
        with network output N and network output gradient delN.

    del2Ytf([x, y, t], N, delN, del2N) - Compute trial function Laplacian
        components at [x, y, t] with network output N, network output gradient
        delN, and network output Laplacian components del2N

Todo:

"""


import numpy as np


class Diff2DTrialFunction():
    """Trial function for 2D diffusion problems."""


    # Public methods

    def __init__(self, bcf, delbcf, del2bcf):
        """Constructor"""
        self.bcf = bcf
        self.delbcf = delbcf
        self.del2bcf = del2bcf

    def Af(self, xyt):
        """Boundary condition function"""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (Y0f, Y1f)) = self.bcf
        A = (1 - x)*f0f([0, y, t]) \
          +      x *f1f([1, y, t]) \
          + (1 - y)*(g0f([x, 0, t]) - ((1 - x)*g0f([0, 0, t]) +
                     x*g0f([1, 0, t]))) \
          +      y *(g1f([x, 1, t]) - ((1 - x)*g1f([0, 1, t]) +
                     x*g1f([1, 1, t]))) \
          + (1 - t)*(Y0f([x, y, 0]) - ((1 - x)*Y0f([0, y, 0]) + x*Y0f([1, y, 0])
                                      + (1 - y)*(Y0f([x, 0, 0]) - ((1 - x)*Y0f([0, 0, 0]) + x*Y0f([1, 0, 0])))
                                      +       y*(Y0f([x, 1, 0]) - ((1 - x)*Y0f([0, 1, 0]) + x*Y0f([1, 1, 0])))))
        return A

    def delAf(self, xyt):
        """Gradient of boundary condition function"""
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dtf), (df1_dxf, df1_dyf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dtf))
         ) = self.delbcf

        dA_dx = -f0f([0, y, t]) + f1f([1, y, t]) \
            - (-1 + y)*(f0f([0, 0, t]) - f1f([1, 0, t]) + dg0_dxf([x, 0, t])) \
            + y*(f0f([0, 1, t]) - f1f([1, 1, t]) + dg1_dxf([x, 1, t])) \
            + (1 - t)*(f0f([0, y, 0]) - f1f([1, y, 0]) + (-1 + y)
                       *(f0f([0, 0, 0]) - f1f([1, 0, 0]) + dg0_dxf([x, 0, 0]))
                       - y*(f0f([0, 1, 0]) - f1f([1, 1, 0]) + dg1_dxf([x, 1, 0]))
                       + dY0_dxf([x, y, 0]))

        dA_dy = -(-1 + x)*f0f([0, 0, t]) + (-1 + x)*f0f([0, 1, t]) + x*f1f([1, 0, t]) - x*f1f([1, 1, t]) \
                - g0f([x, 0, t]) + g1f([x, 1, t]) - (-1 + x)*df0_dyf([0, y, t]) + x*df1_dyf([1, y, t]) \
                + (1 - t)*((-1 + x)*f0f([0, 0, 0]) + f0f([0, 1, 0])
                           - x*(f0f([0, 1, 0]) + f1f([1, 0, 0]) - f1f([1, 1, 0]))
                           + g0f([x, 0, 0]) - g1f([x, 1, 0])
                           + (-1 + x)*df0_dyf([0, y, 0]) - x*df1_dyf([1, y, 0])
                           + dY0_dyf([x, y, 0]))

        dA_dt = -(-1 + x)*f0f([0, y, 0]) + x*f1f([1, y, 0]) \
                - (-1 + y)*((-1 + x)*f0f([0, 0, 0]) - x*f1f([1, 0, 0]) + g0f([x, 0, 0])) \
                + y*((-1 + x)*f0f([0, 1, 0]) - x*f1f([1, 1, 0]) + g1f([x, 1, 0])) \
                - Y0f([x, y, 0]) - (-1 + x)*df0_dtf([0, y, t]) + x*df1_dtf([1, y, t]) \
                + (1 - y)*((-1 + x)*df0_dtf([0, 0, t]) - x*df1_dtf([1, 0, t]) + dg0_dtf([x, 0, t])) \
                + y*((-1 + x)*df0_dtf([0, 1, t]) - x*df1_dtf([1, 1, t]) + dg1_dtf([x, 1, t]))

        delA = [dA_dx, dA_dy, dA_dt]
        return delA

    def del2Af(self, xyt):
        (x, y, t) = xyt
        ((f0f, f1f), (g0f, g1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dtf), (df1_dxf, df1_dyf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dtf))
         ) = self.delbcf
        (((d2f0_dx2f, d2f0_dy2f, d2f0_dt2f), (d2f1_dx2f, d2f1_dy2f, d2f1_dt2f)),
         ((d2g0_dx2f, d2g0_dy2f, d2g0_dt2f), (d2g1_dx2f, d2g1_dy2f, d2g1_dt2f)),
         ((d2Y0_dx2f, d2Y0_dy2f, d2Y0_dt2f), (d2Y1_dx2f, d2Y1_dy2f, d2Y1_dt2f))
         ) = self.del2bcf

        d2A_dx2 = -(-1 + y)*d2g0_dx2f([x, 0, t]) + y*d2g1_dx2f([x, 1, t]) \
            - (-1 + t)*((-1 + y)*d2Y0_dx2f([x, 0, 0]) - y*d2Y0_dx2f([x, 1, 0])
                        + d2Y0_dx2f([x, y, 0]))

        d2A_dy2 = -(-1 + x)*d2f0_dy2f([0, y, t]) + x*d2f1_dy2f([1, y, t]) \
            - (-1 + t)*((-1 + x)*d2Y0_dy2f([0, y, 0]) - x*d2Y0_dy2f([1, y, 0])
                        + d2Y0_dy2f([x, y, 0]))
        d2A_dt2 = -(-1 + x)*d2f0_dt2f([0, y, t]) + x*d2f1_dt2f([1, y, t]) \
            + (1 - y)*((-1 + x)*d2g0_dt2f([0, 0, t]) - x*d2g0_dt2f([1, 0, t])
                       + d2g0_dt2f([x, 0, t])) \
            + y*((-1 + x)*d2g1_dt2f([0, 1, t]) - x*d2g1_dt2f([1, 1, t])
                 + d2g1_dt2f([x, 1, t]))

        del2A = [d2A_dx2, d2A_dy2, d2A_dt2]
        return del2A

    def Pf(self, xyt):
        """Network coefficient function for 2D diffusion problems"""
        (x, y, t) = xyt
        P = x*(1 - x)*y*(1 - y)*t
        return P

    def delPf(self, xyt):
        """Network coefficient function gradient"""
        (x, y, t) = xyt
        dP_dx = (1 - 2*x)*y*(1 - y)*t
        dP_dy = x*(1 - x)*(1 - 2*y)*t
        dP_dt = x*(1 - x)*y*(1 - y)
        delP = [dP_dx, dP_dy, dP_dt]
        return delP

    def del2Pf(self, xyt):
        """Network coefficient function Laplacian"""
        (x, y, t) = xyt
        d2P_dx2 = -2*y*(1 - y)*t
        d2P_dy2 = -2*x*(1 - x)*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dy2, d2P_dt2]
        return del2P

    def Ytf(self, xyt, N):
        """Trial function"""
        A = self.Af(xyt)
        P = self.Pf(xyt)
        Yt = A + P*N
        return Yt

    def delYtf(self, xyt, N, delN):
        """Trial function gradient"""
        (x, y, t) = xyt
        (dN_dx, dN_dy, dN_dt) = delN
        (dA_dx, dA_dy, dA_dt) = self.delAf(xyt)
        P = self.Pf(xyt)
        (dP_dx, dP_dy, dP_dt) = self.delPf(xyt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dy = dA_dy + P*dN_dy + dP_dy*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dy, dYt_dt]
        return delYt

    def del2Ytf(self, xyt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, y, t) = xyt
        (dN_dx, dN_dy, dN_dt) = delN
        (d2N_dx2, d2N_dy2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dy2, d2A_dt2) = self.del2Af(xyt)
        P = self.Pf(xyt)
        (dP_dx, dP_dy, dP_dt) = self.delPf(xyt)
        (d2P_dx2, d2P_dy2, d2P_dt2) = self.del2Pf(xyt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dy2 = d2A_dy2 + P*d2N_dy2 + 2*dP_dy*dN_dy + d2P_dy2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dy2, d2Yt_dt2]
        return del2Yt

#################


# Self-test code

# The code is tested using the diff2d-halfsine problem.

from math import pi, sin, cos

if __name__ == '__main__':

    # Test boundary conditions
    f0f = lambda xyt: 0
    f1f = lambda xyt: 0
    g0f = lambda xyt: 0
    g1f = lambda xyt: 0
    Y0f = lambda xyt: sin(pi*xyt[0])*sin(pi*xyt[1])/2
    Y1f = lambda xyt: None
    bcf = [[f0f, f1f],
           [g0f, g1f],
           [Y0f, Y1f]]

    # Test BC gradient
    df0_dxf = lambda xyt: 0
    df0_dyf = lambda xyt: 0
    df0_dtf = lambda xyt: 0
    df1_dxf = lambda xyt: 0
    df1_dyf = lambda xyt: 0
    df1_dtf = lambda xyt: 0
    dg0_dxf = lambda xyt: 0
    dg0_dyf = lambda xyt: 0
    dg0_dtf = lambda xyt: 0
    dg1_dxf = lambda xyt: 0
    dg1_dyf = lambda xyt: 0
    dg1_dtf = lambda xyt: 0
    dY0_dxf = lambda xyt: pi*cos(pi*xyt[0])*sin(pi*xyt[1])/2
    dY0_dyf = lambda xyt: pi*sin(pi*xyt[0])*cos(pi*xyt[1])/2
    dY0_dtf = lambda xyt: 0
    dY1_dxf = lambda xyt: None
    dY1_dyf = lambda xyt: None
    dY1_dtf = lambda xyt: None
    delbcf = [[[df0_dxf, df0_dyf, df0_dtf], [df1_dxf, df1_dyf, df1_dtf]],
              [[dg0_dxf, dg0_dyf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dtf]],
              [[dY0_dxf, dY0_dyf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dtf]]]

    # Test BC Laplacian
    d2f0_dx2f = lambda xyt: 0
    d2f0_dy2f = lambda xyt: 0
    d2f0_dt2f = lambda xyt: 0
    d2f1_dx2f = lambda xyt: 0
    d2f1_dy2f = lambda xyt: 0
    d2f1_dt2f = lambda xyt: 0
    d2g0_dx2f = lambda xyt: 0
    d2g0_dy2f = lambda xyt: 0
    d2g0_dt2f = lambda xyt: 0
    d2g1_dx2f = lambda xyt: 0
    d2g1_dy2f = lambda xyt: 0
    d2g1_dt2f = lambda xyt: 0
    d2Y0_dx2f = lambda xyt: -pi**2*sin(pi*xyt[0])*sin(pi*xyt[1])/2
    d2Y0_dy2f = lambda xyt: -pi**2*sin(pi*xyt[0])*sin(pi*xyt[1])/2
    d2Y0_dt2f = lambda xyt: 0
    d2Y1_dx2f = lambda xyt: None
    d2Y1_dy2f = lambda xyt: None
    d2Y1_dt2f = lambda xyt: None
    del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dt2f]],
               [[d2g0_dx2f, d2g0_dy2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dt2f]],
               [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dt2f]]]

    # Reference values for tests.
    bc_ref = [[0, 0],
              [0, 0],
              [0.456647, None]]
    delbc_ref = [[[0, 0, 0], [0, 0, 0]],
                 [[0, 0, 0], [0, 0, 0]],
                 [[0.466129, 0.416789, 0], [None, None, None]]]
    del2bc_ref = [[[0, 0, 0], [0, 0, 0]],
                  [[0, 0, 0], [0, 0, 0]],
                  [[-4.50692, -4.50692, 0], [None, None, None]]]
    A_ref = 0.264855
    delA_ref = [0.270355, 0.241738, -0.456647]
    del2A_ref = [-2.61402, -2.61402, 0]
    P_ref = 0.243835
    delP_ref = [0.0203196, 0.018144, 0.058056]
    del2P_ref = [-0.203196, -0.2016, 0]
    Yt_ref = 0.277047
    delYt_ref = [0.295389, 0.265928, -0.412257]
    del2Yt_ref = [-2.67351, -2.67476, 0.0909505]

    # Additional test variables.
    N_test = 0.5
    delN_test = [0.61, 0.62, 0.63]
    del2N_test = [0.71, 0.72, 0.73]

    # Test all functions near the center of the domain.
    xyt_test = [0.4, 0.41, 0.42]

    # Create a new trial function object.
    tf = Diff2DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing boundary conditions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc_test = f(xyt_test)
            if ((bc_ref[i][j] is not None and
                 not np.isclose(bc_test, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc_test is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" %
                      (i, j, bc_test, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc_test = f(xyt_test)
                if ((delbc_ref[i][j][k] is not None and
                     not np.isclose(delbc_test, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and
                        delbc_test is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" %
                          (i, j, k, delbc_test, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(tf.del2bcf)):
        for j in range(len(tf.del2bcf[i])):
            for (k, f) in enumerate(tf.del2bcf[i][j]):
                del2bc_test = f(xyt_test)
                if ((del2bc_ref[i][j][k] is not None and
                     not np.isclose(del2bc_test, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and
                        del2bc_test is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" %
                          (i, j, k, del2bc_test, del2bc_ref[i][j][k]))

    print("Testing boundary condition function.")
    A_test = tf.Af(xyt_test)
    if not np.isclose(A_test, A_ref):
        print("ERROR: A = %s, vs ref %s" % (A_test, A_ref))

    print("Testing boundary condition function gradient.")
    delA_test = tf.delAf(xyt_test)
    for (i, delA_t) in enumerate(delA_test):
        if not np.isclose(delA_t, delA_ref[i]):
            print("ERROR: delA[%d] = %s, vs ref %s" % (i, delA_t, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A_test = tf.del2Af(xyt_test)
    for (i, del2A_t) in enumerate(del2A_test):
        if not np.isclose(del2A_t, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" % (i, del2A_t, del2A_ref[i]))

    print("Testing network coefficient function.")
    P_test = tf.Pf(xyt_test)
    if not np.isclose(P_test, P_ref):
        print("ERROR: P = %s, vs ref %s" % (P_test, P_ref))

    print("Testing network coefficient function gradient.")
    delP_test = tf.delPf(xyt_test)
    for (i, delP_t) in enumerate(delP_test):
        if not np.isclose(delP_t, delP_ref[i]):
            print("ERROR: delP[%d] = %s, vs ref %s" % (delP_t, delP_ref[i]))

    print("Testing network coefficient function Laplacian.")
    del2P_test = tf.del2Pf(xyt_test)
    for (i, del2P_t) in enumerate(del2P_test):
        if not np.isclose(del2P_t, del2P_ref[i]):
            print("ERROR: del2P[%d] = %s, vs ref %s" % (del2P_t, del2P_ref[i]))

    print("Testing trial function.")
    Yt_test = tf.Ytf(xyt_test, N_test)
    if not np.isclose(Yt_test, Yt_ref):
        print("ERROR: Yt = %s, vs ref %s" % (Yt_test, Yt_ref))

    print("Testing trial function gradient.")
    delYt_test = tf.delYtf(xyt_test, N_test, delN_test)
    for (i, delYt_t) in enumerate(delYt_test):
        if not np.isclose(delYt_t, delYt_ref[i]):
            print("ERROR: delYt[%d] = %s, vs ref %s" % (i, delYt_t, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt_test = tf.del2Ytf(xyt_test, N_test, delN_test, del2N_test)
    for (i, del2Yt_t) in enumerate(del2Yt_test):
        if not np.isclose(del2Yt_t, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" % (i, del2Yt_t, del2Yt_ref[i]))

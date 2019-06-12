###############################################################################
"""
Diff1DTrialFunction - Class implementing the trial function for 1-D diffusion
problems

The trial function takes the form:

Yt(x,t) = A(x, t) + P(x, t)N(x, t, p)

where:

A(x, t) = boundary condition function that reduces to BC at boundaries
P(x, t) = network coefficient function that vanishes at boundaries
N(x, t, p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff1DTrialFunction object from boundary conditions
        Yt_obj = Diff1DTrialFunction(bcf, delbcf, del2bcf)

    Compute the value of the trial function at a given point
        Yt = Yt_obj.Ytf([x, t], N)

    Compute the value of the boundary condition function at a given point
        A = Yt_obj.Af([x, t])

Notes:
    Variables that end in 'f' are usually functions or arrays of functions.

Attributes:
    bcf - 2x2 array of BC functions at (x,t)=0|1
    delbcf - 2x2x2 array of BC gradient functions at (x,t)=0|1
    del2bcf - 2x2x2 array of BC Laplacian component functions at (x,t)=0|1

Methods:
    Af([x, t]) - Compute boundary condition function at [x, t]

    delAf([x, t]) - Compute boundary condition function gradient at [x, t]

    del2Af([x, t]) - Compute boundary condition function Laplacian components
        at [x, t]

    Pf([x, t]) - Compute network coefficient function at [x, t]

    delPf([x, t]) - Compute network coefficient function gradient at [x, t]

    del2Pf([x, t]) - Compute network coefficient function Laplacian components
        at [x, t]

    Ytf([x, t], N) - Compute trial function at [x, t] with network output N

    delYtf([x, t], N, delN) - Compute trial function gradient at [x, t] with
        network output N and network output gradient delN.

    del2Ytf([x, t], N, delN, del2N) - Compute trial function Laplacian
        components at [x, t] with network output N, network output gradient
        delN, and network output Laplacian components del2N

Todo:

"""


import numpy as np


class Diff1DTrialFunction():
    """Trial function for 1D diffusion problems."""


    # Public methods

    def __init__(self, bcf, delbcf, del2bcf):
        """Constructor"""
        self.bcf = bcf
        self.delbcf = delbcf
        self.del2bcf = del2bcf

    def Af(self, xt):
        """Boundary condition function"""
        (x, t) = xt
        ((f0f, f1f), (Y0f, Y1f)) = self.bcf
        A = (1 - x)*f0f([0, t]) + x*f1f([1, t]) + \
            (1 - t)*(Y0f([x, 0]) - ((1 - x)*Y0f([0, 0]) + x*Y0f([1, 0])))
        return A

    def delAf(self, xt):
        """Boundary condition function gradient"""
        (x, t) = xt
        ((f0f, f1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dtf), (df1_dxf, df1_dtf)),
         ((dY0_dxf, dY0_dtf), (dY1_dxf, dY1_dtf))) = self.delbcf
        dA_dx = -f0f([0, t]) + f1f([1, t]) - (-1 + t)* \
                (Y0f([0, 0]) - Y0f([1, 0]) + dY0_dxf([x, 0]))
        dA_dt =  Y0f([0, 0]) - x*Y0f([0, 0]) + x*Y0f([1, 0]) - Y0f([x, 0]) - \
                 (-1 + x)*df0_dtf([0, t]) + x*df1_dtf([1, t])
        delA = [dA_dx, dA_dt]
        return delA

    def del2Af(self, xt):
        """Laplacian of boundary condition function"""
        (x, t) = xt
        ((f0f, f1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dtf), (df1_dxf, df1_dtf)),
         ((dY0_dxf, dY0_dtf), (dY1_dxf, dY1_dtf))) = self.delbcf
        (((d2f0_dx2f, d2f0_dt2f), (d2f1_dx2f, d2f1_dt2f)),
         ((d2Y0_dx2f, d2Y0_dt2f), (d2Y1_dx2f, d2Y1_dt2f))) = self.del2bcf
        d2A_dx2 = -(-1 + t)*d2Y0_dx2f([x, 0])
        d2A_dt2 = -(-1 + x)*d2f0_dt2f([0, t]) + x*d2f1_dt2f([1, t])
        del2A = [d2A_dx2, d2A_dt2]
        return del2A

    def Pf(self, xt):
        """Network coefficient function"""
        (x, t) = xt
        P = x*(1 - x)*t
        return P

    def delPf(self, xt):
        """Network coefficient function gradient"""
        (x, t) = xt
        dP_dx = (1 - 2*x)*t
        dP_dt = x*(1 - x)
        delP = [dP_dx, dP_dt]
        return delP

    def del2Pf(self, xt):
        """Network coefficient function Laplacian"""
        (x, t) = xt
        d2P_dx2 = -2*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dt2]
        return del2P

    def Ytf(self, xt, N):
        """Trial function"""
        A = self.Af(xt)
        P = self.Pf(xt)
        Yt = A + P*N
        return Yt

    def delYtf(self, xt, N, delN):
        """Trial function gradient"""
        (x, t) = xt
        (dN_dx, dN_dt) = delN
        (dA_dx, dA_dt) = self.delAf(xt)
        P = self.Pf(xt)
        (dP_dx, dP_dt) = self.delPf(xt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dt]
        return delYt

    def del2Ytf(self, xt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, t) = xt
        (dN_dx, dN_dt) = delN
        (d2N_dx2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dt2) = self.del2Af(xt)
        P = self.Pf(xt)
        (dP_dx, dP_dt) = self.delPf(xt)
        (d2P_dx2, d2P_dt2) = self.del2Pf(xt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dt2]
        return del2Yt

#################


# Self-test code

# The code is tested using the diff1d-halfsine problem.

from math import pi, sin, cos

if __name__ == '__main__':

    # Test boundary conditions
    f0f = lambda xt: 0
    f1f = lambda xt: 0
    Y0f = lambda xt: sin(pi*xt[0])
    Y1f = lambda xt: None
    bcf = [[f0f, f1f],
           [Y0f, Y1f]]

    # Test BC gradient
    df0_dxf = lambda xt: 0
    df0_dtf = lambda xt: 0
    df1_dxf = lambda xt: 0
    df1_dtf = lambda xt: 0
    dY0_dxf = lambda xt: pi*cos(pi*xt[0])
    dY0_dtf = lambda xt: 0
    dY1_dxf = lambda xt: None
    dY1_dtf = lambda xt: None
    delbcf = [[[df0_dxf, df0_dtf], [df1_dxf, df1_dtf]],
              [[dY0_dxf, dY0_dtf], [dY1_dxf, dY1_dtf]]]

    # Test BC Laplacian
    d2f0_dx2f = lambda xt: 0
    d2f0_dt2f = lambda xt: 0
    d2f1_dx2f = lambda xt: 0
    d2f1_dt2f = lambda xt: 0
    d2Y0_dx2f = lambda xt: -pi**2*sin(pi*xt[0])
    d2Y0_dt2f = lambda xt: 0
    d2Y1_dx2f = lambda xt: None
    d2Y1_dt2f = lambda xt: None
    del2bcf = [[[d2f0_dx2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dt2f]],
               [[d2Y0_dx2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dt2f]]]

    # Reference values for tests.
    bc_ref = [[0, 0],
              [0.951057, None]]
    delbc_ref = [[[0, 0], [0, 0]],
                 [[0.970806, 0], [None, None]]]
    del2bc_ref = [[[0, 0], [0, 0]],
                  [[-9.38655, 0], [None, None]]]
    A_ref = 0.561123
    delA_ref = [0.572775, -0.951057]
    del2A_ref = [-5.53807, 0]
    P_ref = 0.0984
    delP_ref = [0.082, 0.24]
    del2P_ref = [-0.82, 0]
    Yt_ref = 0.610323
    delYt_ref = [0.673799, -0.770049]
    del2Yt_ref = [-5.77816, 0.368448]

    # Test all functions near the center of the domain.
    xt_test = [0.4, 0.41]

    # Additional test variables.
    N_test = 0.5
    delN_test = [0.61, 0.62]
    del2N_test = [0.71, 0.72]

    # Create a new trial function object.
    tf = Diff1DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing boundary conditions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc_test = f(xt_test)
            if ((bc_ref[i][j] is not None and
                 not np.isclose(bc_test, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc_test is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" %
                      (i, j, bc_test, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc_test = f(xt_test)
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
                del2bc_test = f(xt_test)
                if ((del2bc_ref[i][j][k] is not None and
                     not np.isclose(del2bc_test, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and
                        del2bc_test is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" %
                          (i, j, k, del2bc_test, del2bc_ref[i][j][k]))

    print("Testing boundary condition function.")
    A_test = tf.Af(xt_test)
    if not np.isclose(A_test, A_ref):
        print("ERROR: A = %s, vs ref %s" % (A_test, A_ref))

    print("Testing boundary condition function gradient.")
    delA_test = tf.delAf(xt_test)
    for (i, delA_t) in enumerate(delA_test):
        if not np.isclose(delA_t, delA_ref[i]):
            print("ERROR: delA[%d] = %s, vs ref %s" %
                  (i, delA_t, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A_test = tf.del2Af(xt_test)
    for (i, del2A_t) in enumerate(del2A_test):
        if not np.isclose(del2A_t, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" %
                  (i, del2A_t, del2A_ref[i]))

    print("Testing network coefficient function.")
    P_test = tf.Pf(xt_test)
    if not np.isclose(P_test, P_ref):
        print("ERROR: P = %s, vs ref %s" % (P_test, P_ref))

    print("Testing network coefficient function gradient.")
    delP_test = tf.delPf(xt_test)
    for (i, delP_t) in enumerate(delP_test):
        if not np.isclose(delP_t, delP_ref[i]):
            print("ERROR: delP[%d] = %s, vs ref %s" %
                  (i, delP_t, delP_ref[i]))

    print("Testing network coefficient function Laplacian.")
    del2P_test = tf.del2Pf(xt_test)
    for (i, del2P_t) in enumerate(del2P_test):
        if not np.isclose(del2P_t, del2P_ref[i]):
            print("ERROR: del2P[%d] = %s, vs ref %s" %
                  (i, del2P_t, del2P_ref[i]))

    print("Testing trial function.")
    Yt_test = tf.Ytf(xt_test, N_test)
    if not np.isclose(Yt_test, Yt_ref):
        print("ERROR: Yt = %s, vs ref %s" % (Yt_test, Yt_ref))

    print("Testing trial function gradient.")
    delYt_test = tf.delYtf(xt_test, N_test, delN_test)
    for (i, delYt_t) in enumerate(delYt_test):
        if not np.isclose(delYt_t, delYt_ref[i]):
            print("ERROR: delYt[%d] = %s, vs ref %s" %
                  (i, delYt_t, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt_test = tf.del2Ytf(xt_test, N_test, delN_test, del2N_test)
    for (i, del2Yt_t) in enumerate(del2Yt_test):
        if not np.isclose(del2Yt_t, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" %
                  (i, del2Yt_t, del2Yt_ref[i]))

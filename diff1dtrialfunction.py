###############################################################################
"""
Diff1DTrialFunction - Class implementing the trial function for 1-D diffusion
problems

The trial function takes the form:

Yt(x,t) = A(x,t) + P(x,t)N(x,t,p)

where:

A(x,t) = boundary condition function that reduces to BC at boundaries
P(x,t) = network coefficient function that vanishes at boundaries
N(x,t,p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff1DTrialFunction object
        Yt = Diff1DTrialFunction()

Notes:
    * Function names end in 'f'.

Attributes:
    None

Methods:
    None

Todo:
    * Expand basic functionality
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
    A_ref = 0.475228
    delA_ref = [0.485403, -0.951057]
    del2A_ref = [-4.69328, 0]
    P_ref = 0.12
    delP_ref = [0.1, 0.24]
    del2P_ref = [-1, 0]
    Yt_ref = 0.547528
    delYt_ref = [0.629403, -0.711057]
    del2Yt_ref = [-5.14008, 0.4104]

    # Test all functions near the center of the domain.
    xt = [0.4, 0.5]

    # Additional test variables.
    N_ref = 0.6
    delN_ref = [0.7, 0.8]
    del2N_ref = [0.11, 0.22]

    # Create a new trial function object.
    tf = Diff1DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing boundary conditions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc = f(xt)
            if ((bc_ref[i][j] is not None and not np.isclose(bc, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" % (i, j, bc, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc = f(xt)
                if ((delbc_ref[i][j][k] is not None and not np.isclose(delbc, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and delbc is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, delbc, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(tf.del2bcf)):
        for j in range(len(tf.del2bcf[i])):
            for (k, f) in enumerate(tf.del2bcf[i][j]):
                del2bc = f(xt)
                if ((del2bc_ref[i][j][k] is not None and not np.isclose(del2bc, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and del2bc is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" % (i, j, k, del2bc, del2bc_ref[i][j][k]))

    print("Testing boundary condition function.")
    A = tf.Af(xt)
    if not np.isclose(A, A_ref):
        print("ERROR: A = %s, vs ref %s" % (A, A_ref))

    print("Testing boundary condition function gradient.")
    delA = tf.delAf(xt)
    for (i, delAi) in enumerate(delA):
        if not np.isclose(delAi, delA_ref[i]):
            print("ERROR: delA[%d] = %s, vs ref %s" % (i, delAi, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A = tf.del2Af(xt)
    for (i, del2Ai) in enumerate(del2A):
        if not np.isclose(del2Ai, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" % (i, del2Ai, del2A_ref[i]))

    print("Testing network coefficient function.")
    P = tf.Pf(xt)
    if not np.isclose(P, P_ref):
        print("ERROR: P = %s, vs ref %s" % (P, P_ref))

    print("Testing network coefficient function gradient.")
    delP = tf.delPf(xt)
    for (i, delPi) in enumerate(delP):
        if not np.isclose(delPi, delP_ref[i]):
            print("ERROR: delP[%d] = %s, vs ref %s" % (i, delPi, delP_ref[i]))

    print("Testing network coefficient function Laplacian.")
    del2P = tf.del2Pf(xt)
    for (i, del2Pi) in enumerate(del2P):
        if not np.isclose(del2Pi, del2P_ref[i]):
            print("ERROR: del2P[%d] = %s, vs ref %s" % (i, del2Pi, del2P_ref[i]))

    print("Testing trial function.")
    Yt = tf.Ytf(xt, N_ref)
    if not np.isclose(Yt, Yt_ref):
        print("ERROR: Yt = %s, vs ref %s" % (Yt, Yt_ref))

    print("Testing trial function gradient.")
    delYt = tf.delYtf(xt, N_ref, delN_ref)
    for (i, delYti) in enumerate(delYt):
        if not np.isclose(delYti, delYt_ref[i]):
            print("ERROR: delYt[%d] = %s, vs ref %s" % (i, delYti, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt = tf.del2Ytf(xt, N_ref, delN_ref, del2N_ref)
    for (i, del2Yti) in enumerate(del2Yt):
        if not np.isclose(del2Yti, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" % (i, del2Yti, del2Yt_ref[i]))

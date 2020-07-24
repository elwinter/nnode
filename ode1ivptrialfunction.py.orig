###############################################################################
"""
ODE1IVPTrialFunction - Class implementing the trial function for 1st-order
ODE initial value problems

The trial function takes the form:

Yt(x) = A(x) + P(x)*N(x, p)

where:

A(x) = boundary condition function that reduces to BC at boundaries
P(x) = network coefficient function that vanishes at boundaries
N(x, p) = scalar output of neural network with parameter vector p

Example:
    Create a default ODE1IVPTrialFunction object from boundary conditions
        Yt_obj = ODE1IVPTrialFunction(bcf, delbcf)

    Compute the value of the boundary condition function at a given point
        A = Yt_obj.Af([x])

    Compute the value of the network coefficient function at a given point
        P = Yt_obj.Pf([x])

    Compute the value of the trial function at a given point
        Yt = Yt_obj.Ytf([x], N)

Notes:
    Variables that end in 'f' are usually functions or arrays of functions.

Attributes:
    bcf - Length 2 array of BC functions at (x)=0|1
    delbcf - 1x2 array of BC gradient functions at (x)=0|1

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


class ODE1IVPTrialFunction():
    """Trial function for 1st-order ODE IVP."""


    # Public methods

    def __init__(self, bcf, delbcf):
        """Constructor"""
        self.bcf = bcf
        self.delbcf = delbcf

    def Af(self, x):
        """Boundary condition function"""
        return self.bcf[0](0)

    def delAf(self, x):
        """Boundary condition function gradient"""
        return 0

    def Pf(self, x):
        """Network coefficient function"""
        return x

    def delPf(self, x):
        """Network coefficient function gradient"""
        return 1

    def Ytf(self, x, N):
        """Trial function"""
        return self.Af(x) + self.Pf(x)*N

    def delYtf(self, x, N, delN):
        """Trial function gradient"""
        return self.delAf(x) + self.Pf(x)*delN + self.delPf(x)*N

#################


# Self-test code

# The code is tested using the diff1d-halfsine problem.

# from math import pi, sin, cos

if __name__ == '__main__':


    # Define initial condition.
    A = 1

    # Test boundary conditions
    f0f = lambda x: A
    f1f = None
    bcf = [f0f, f1f]

    # Test BC gradient
    df0_dxf = lambda xt: 0
    df1_dxf = lambda xt: None
    delbcf = [df0_dxf, df1_dxf]

    # Test values for tests
    x_test = 0.4
    N_test = 1.5
    delN_test = 0.5

    # Reference values for tests
    A_ref = A
    delA_ref = 0
    P_ref = x_test
    delP_ref = 1
    Yt_ref = A + P_ref*N_test
    delYt_ref = delA_ref + P_ref*delN_test + delP_ref*N_test

    # Create a new trial function object.
    tf = ODE1IVPTrialFunction(bcf, delbcf)

    print("Testing boundary condition function.")
    assert np.isclose(tf.Af(x_test), A_ref)

    print("Testing boundary condition function gradient.")
    assert np.isclose(tf.delAf(x_test), delA_ref)

    print("Testing network coefficient function.")
    assert np.isclose(tf.Pf(x_test), P_ref)

    print("Testing network coefficient function gradient.")
    assert np.isclose(tf.delPf(x_test), delP_ref)

    print("Testing trial function.")
    assert np.isclose(tf.Ytf(x_test, N_test), Yt_ref)

    print("Testing trial function gradient.")
    assert np.isclose(tf.delYtf(x_test, N_test, delN_test), delYt_ref)

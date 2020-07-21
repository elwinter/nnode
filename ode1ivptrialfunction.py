"""
ODE1IVPTrialFunction - Lagaris-style trial funcion for 1st-order ODE IVP

The trial function takes the form:

Yt(xv, p) = A(xv) + P(xv)*N(xv, p)

where:

xv is a vector of independent variables (xv = [x])
p is a vector of network parameters
A(xv) = boundary condition function that reduces to BC at boundaries
P(xv) = network coefficient function that vanishes at boundaries
N(xv, p) = scalar output of neural network with parameter vector p

Notes:

The equation applies to a unit domain 0 <= x <= 1

Attributes:

Methods:
    Af(xv) - Compute boundary condition function at xv.

    delAf(xv) - Compute boundary condition function gradient at xv.

    deldelAf(xv) - Compute boundary condition function Hessian at xv.

    Pf(xv) - Compute network coefficient function at xv.

    delPf(xv) - Compute network coefficient function gradient at xv.

    deldelPf(xv) - Compute network coefficient function Hessian at xv.

    Ytf(xv, N) - Compute trial function at xv with network output N.

    delYtf(xv, N, delN) - Compute trial function gradient at xv with
        network output N and network output gradient delN.

    deldelYtf(xv, N, delN, deldelN) - Compute trial function Hessian
        at xv with network output N, network output gradient
        delN, and network output Hessian deldelN

Todo:
"""


import numpy as np

from APtrialfunction import APTrialFunction


class ODE1IVPTrialFunction(APTrialFunction):
    """Lagaris-style trial function for 1st-order ODE IVP"""


    # Public methods

    def __init__(self, bcf, delbcf):
        """Constructor"""
        super().__init__()
        self.bcf = bcf
        self.delbcf = delbcf

    def Af(self, xv):
        """Boundary condition function"""
        return self.bcf[0](0)

    def delAf(self, xv):
        """Boundary condition function gradient"""
        return [0]

    def Pf(self, xv):
        """Network coefficient function"""
        return xv[0]

    def delPf(self, xv):
        """Network coefficient function gradient"""
        return 1

    def Ytf(self, xv, N):
        """Trial function"""
        print(xv, N)
        A = self.Af(xv)
        P = self.Pf(xv)
        Yt = A + P*N
        return None

    def delYtf(self, xv, N, delN):
        """Trial function gradient"""
        delA = self.delAf(xv)
        P = self.Pf(xv)
        delYt = []
        delYt.append(delA[0] + P*delN[0] + delP[0]*n)
        return delYt


if __name__ == '__main__':

    # Test inputs
    ic_test = 0  # Initial condition
    xv_test = [0]
    N_test = 0
    bcf_test = [ lambda x: ic_test ]
    delbcf_test = [[ lambda x: 0 ]]

    # Expected test outputs
    A_ref = 0
    delA_ref = [0]
    P_ref = xv_test[0]
    delP_ref = [1]
    Yt_ref = 0
    
    tf = ODE1IVPTrialFunction(bcf_test, delbcf_test)
    print(tf)
    assert np.isclose(tf.Af(xv_test), A_ref)
    assert np.isclose(tf.delAf(xv_test), delA_ref)
    assert np.isclose(tf.Pf(xv_test), P_ref)
    assert np.isclose(tf.delPf(xv_test), delP_ref)
    assert np.isclose(tf.Ytf(xv_test, N_test), Yt_ref)


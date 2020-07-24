"""
APTrialFunction - Lagaris-style trial funcion

NOTE: This function currently only works for differential equations of
up to 2nd order.

The trial function takes the form:

Yt(xv, p) = A(xv) + P(xv)*N(xv, p)

where:

xv is a vector of independent variables
p is a vector of network parameters
A(xv) = boundary condition function that reduces to BC at boundaries
P(xv) = network coefficient function that vanishes at boundaries
N(xv, p) = scalar output of neural network with parameter vector p

Notes:

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


from trialfunction import TrialFunction


class APTrialFunction(TrialFunction):
    """Trial function base class"""


    # Public methods

    def __init__(self):
        """Constructor"""
        super().__init__()

    def Af(self, xv):
        """Boundary condition function"""
        return None

    def delAf(self, xv):
        """Boundary condition function gradient"""
        return None

    def deldelAf(self, xv):
        """Boundary condition function Hessian"""
        return None

    def Pf(self, xv):
        """Network coefficient function"""
        return None

    def delPf(self, xv):
        """Network coefficient function gradient"""
        return None

    def deldelPf(self, xv):
        """Network coefficient function Hessian"""
        return None

    def Ytf(self, xv, N):
        """Trial function"""
        A = self.Af(xv)
        P = self.Pf(xv)
        Yt = A + P*N
        return None

    def delYtf(self, xv, N, delN):
        """Trial function gradient"""
        delA = self.delAf(xv)
        P = self.Pf(xv)
        delYt = []
        m = len(xv)
        for j in range(m):
            delYt.append(delA[j] + P*delN[j] + delP[j]*n)
        return delYt

    def deldelYtf(self, xv, N, delN, deldelN):
        """Trial function Hessian"""
        deldelA = self.deldelAf(xv)
        P = self.Pf(xv)
        delP = self.delPf(xv)
        deldelP = self.deldelPf(xv)
        m = len(xv)
        for j in range(m):
            tmp = []
            for jj in range(m):
                tmp.append(deldelA[j][jj] + P*deldelN[j][jj] + delP[j]*delN[jj] +
                            delP[jj]*delN[j] + deldelP[j][jj]*N)
            deldelYt.append(tmp)

        return deldelYt


if __name__ == '__main__':

    # Test inputs
    xv_test = None
    N_test = None
    delN_test = None
    deldelN_test = None

    tf = APTrialFunction()
    print(tf)
    assert tf.Af(xv_test) is None
    assert tf.delAf(xv_test) is None
    assert tf.deldelAf(xv_test) is None
    assert tf.Pf(xv_test) is None
    assert tf.delPf(xv_test) is None
    assert tf.deldelPf(xv_test) is None

    try:
        Yt = tf.Ytf(xv_test, N_test)
    except TypeError:
        pass
    except Exception as e:
        print(e)

    try:
        delYt = tf.delYtf(xv_test, N_test, delN_test)
    except TypeError:
        pass
    except Exception as e:
        print(e)

    try:
        deldelYt = tf.deldelYtf(xv_test, N_test, delN_test, deldelN_test)
    except TypeError:
        pass
    except Exception as e:
        print(e)

"""
TrialFunction - Base class for trial functions

Notes:

Only for 2nd-order PDEs.

Attributes:

Methods:

    Ytf(xv, N) - Compute trial function at xv with network output N.

    delYtf(xv, N, delN) - Compute trial function gradient at xv with
        network output N and network output gradient delN.

    deldelYtf(xv, N, delN, deldelN) - Compute trial function Hessian
        at xv with network output N, network output gradient
        delN, and network output Hessian deldelN

Todo:

"""


class TrialFunction():
    """Trial function base class"""


    # Public methods

    def __init__(self):
        """Constructor"""
        pass

    def Ytf(self, xv, N):
        """Trial function"""
        return None

    def delYtf(self, xv, N, delN):
        """Trial function gradient"""
        return None

    def deldelYtf(self, xv, N, delN, deldelN):
        """Trial function Hessian"""
        return 


if __name__ == '__main__':
    tf = TrialFunction()
    print(tf)
    assert tf.Ytf(None, None) is None
    assert tf.delYtf(None, None, None) is None
    assert tf.deldelYtf(None, None, None, None) is None
    

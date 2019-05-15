"""
NNPDE2DIFF2D - Class to solve 2-D diffusion problems using a neural network

This module provides the functionality to solve 2-D diffusion problems using
a neural network.

Example:
    Create an empty NNPDE2DIFF2D object.
        net = NNPDE2DIFF2D()
    Create an NNPDE2DIFF2D object for a PDE2DIFF2D object.
        net = NNPDE2DIFF2D(pde2diff2d_obj)
    Create an NNPDE2DIFF2D object for a PDE2DIFF2D object, with 20 hidden
              nodes.
        net = NNPDE2DIFF2D(pde2diff2d_obj, nhid=20)

Attributes:
    TBD

Methods:
    train
    run
    run_gradient
    run_laplacian

Todo:
    * Expand base functionality.
"""
###############################################################################

from math import sqrt
import numpy as np
from scipy.optimize import minimize

from diff2dtrialfunction import Diff2DTrialFunction
from kdelta import kdelta
from pde2diff2d import PDE2DIFF2D
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from slffnn import SLFFNN


# Default values for method parameters
DEFAULT_DEBUG = False
DEFAULT_ETA = 0.1
DEFAULT_MAXEPOCHS = 100
DEFAULT_NHID = 10
DEFAULT_TRAINALG = 'delta'
DEFAULT_UMAX = 1
DEFAULT_UMIN = -1
DEFAULT_USE_HESSIAN = False
DEFAULT_USE_JACOBIAN = False
DEFAULT_VERBOSE = False
DEFAULT_VMAX = 1
DEFAULT_VMIN = -1
DEFAULT_WMAX = 1
DEFAULT_WMIN = -1
DEFAULT_OPTS = {
    'debug':        DEFAULT_DEBUG,
    'eta':          DEFAULT_ETA,
    'maxepochs':    DEFAULT_MAXEPOCHS,
    'nhid':         DEFAULT_NHID,
    'umax':         DEFAULT_UMAX,
    'umin':         DEFAULT_UMIN,
    'use_hessian':  DEFAULT_USE_HESSIAN,
    'use_jacobian': DEFAULT_USE_JACOBIAN,
    'verbose':      DEFAULT_VERBOSE,
    'vmax':         DEFAULT_VMAX,
    'vmin':         DEFAULT_VMIN,
    'wmax':         DEFAULT_WMAX,
    'wmin':         DEFAULT_WMIN
    }


# Vectorize sigma functions for speed.
# sigma_v = np.vectorize(sigma)
# dsigma_dz_v = np.vectorize(dsigma_dz)
# d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
# d3sigma_dz3_v = np.vectorize(d3sigma_dz3)


class NNPDE2DIFF2D(SLFFNN):
    """Solve a 2-D diffusion problem with a neural network"""


    # Public methods


    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS,
              options=None):
        """Train the network to solve a 2-D diffusion problem"""
        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        if trainalg == 'delta':
            self.__train_delta_debug(x, opts=my_opts)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG'):
            self.__train_minimize(x, trainalg, opts=my_opts, options=options)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(1)

    # def run(self, x):
    #     """Compute the trained solution."""

    #     # Fetch the number n of input points at which to calculate the
    #     # output, and the number m of components of each point.
    #     n = len(x)
    #     m = len(x[0])

    #     # Fetch the number of hidden nodes in the neural network.
    #     H = len(self.w[0])

    #     # Get references to the network parameters for convenience.
    #     w = self.w
    #     u = self.u
    #     v = self.v

    #     # Compute the activation for each input point and hidden node.
    #     z = np.dot(x, w) + u

    #     # Compute the sigma function for each input point and hidden node.
    #     s = sigma_v(z)

    #     # Compute the network output for each input point.
    #     N = np.dot(s, v)

    #     # Compute the value of the trial function for each input point.
    #     Yt = np.zeros(n)
    #     for i in range(n):
    #         Yt[i] = self.__Ytf(x[i], N[i])

    #     # Return the trial function values for each input point.
    #     return Yt

    def run_debug(self, x):
        """Compute the trained solution (debug version)."""

        # Fetch the number n of input points at which to calculate the
        # output, and the number m of components of each point.
        n = len(x)
        m = len(x[0])

        # Fetch the number of hidden nodes in the neural network.
        H = len(self.w[0])

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        # Compute the activation for each input point and hidden node.
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        # Compute the sigma function for each input point and hidden node.
        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])

        # Compute the network output for each input point.
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        # Compute the value of the trial function for each input point.
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Ytf(x[i], N[i])

        # Return the trial function values for each input point.
        return Yt

    # def run_gradient(self, x):
    #     """Compute the trained gradient."""

    #     # Fetch the number n of input points at which to calculate the
    #     # output, and the number m of components of each point.
    #     n = len(x)
    #     m = len(x[0])

    #     # Fetch the number of hidden nodes in the neural network.
    #     H = len(self.w[0])

    #     # Get references to the network parameters for convenience.
    #     w = self.w
    #     u = self.u
    #     v = self.v

    #     # Compute the activation for each input point and hidden node.
    #     z = np.dot(x, w) + u

    #     # Compute the sigma function for each input point and hidden node.
    #     s = sigma_v(z)

    #     # Compute the sigma function 1st derivative for each input point
    #     # and hidden node.
    #     s1 = dsigma_dz_v(z)

    #     # Compute the network output for each input point.
    #     N = np.dot(s, v)

    #     # Compute the network output gradient for each input point.
    #     delN = np.dot(s1, (v*w).T)

    #     # Compute the gradient of the booundary condition function for each
    #     # input point.
    #     delA = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delA[i, j] = self.delAf[j](x[i])

    #     # Compute the network coefficient function for each input point.
    #     P = np.zeros(n)
    #     for i in range(n):
    #         P[i] = self.__Pf(x[i])

    #     # Compute the gradient of the network coefficient function for each
    #     # input point.
    #     delP = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delP[i, j] = self.delPf[j](x[i])

    #     # Compute the gradient of the trial solution for each input point.
    #     delYt = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delYt[i, j] = delA[i, j] + P[i]*delN[i, j] + \
    #                 delP[i, j]*N[i]

    #     return delYt

    def run_gradient_debug(self, x):
        """Compute the trained gradient (debug version)."""

        # Fetch the number n of input points at which to calculate the
        # output, and the number m of components of each point.
        n = len(x)
        m = len(x[0])

        # Fetch the number of hidden nodes in the neural network.
        H = len(self.w[0])

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = dsigma_dz(z[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])

        return delYt

    # def run_hessian(self, x):
    #     """Compute the trained Hessian."""

    #     # Fetch the number n of input points at which to calculate the
    #     # output, and the number m of components of each point.
    #     n = len(x)
    #     m = len(x[0])

    #     # Fetch the number of hidden nodes in the neural network.
    #     H = len(self.w[0])

    #     # Get references to the network parameters for convenience.
    #     w = self.w
    #     u = self.u
    #     v = self.v

    #     # Compute the activation for each input point and hidden node.
    #     z = np.dot(x, w) + u

    #     # Compute the sigma function for each input point and hidden node.
    #     s = sigma_v(z)

    #     # Compute the sigma function 1st derivative for each input point
    #     # and hidden node.
    #     s1 = dsigma_dz_v(z)

    #     # Compute the sigma function 2nd derivative for each input point
    #     # and hidden node.
    #     s2 = d2sigma_dz2_v(z)

    #     # Compute the network output for each input point.
    #     N = np.dot(s, v)

    #     # Compute the network output gradient for each input point.
    #     delN = np.dot(s1, (v*w).T)

    #     # Compute the network output Hessian for each input point.
    #     deldelN = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelN[i, j, jj] = np.dot(v, s2[i]*w[j]*w[jj])

    #     # Compute the Hessian of the boundary condition function
    #     # for each input point.
    #     deldelA = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelA[i, j, jj] = self.deldelAf[j][jj](x[i])

    #     # Compute the network coefficient function for each input point.
    #     P = np.zeros(n)
    #     for i in range(n):
    #         P[i] = self.__Pf(x[i])

    #     # Compute the gradient of the network coefficient function for each
    #     # input point.
    #     delP = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delP[i, j] = self.delPf[j](x[i])

    #     # Compute the Hessian of the network coefficient function for each
    #     # input point.
    #     deldelP = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelP[i, j, jj] = self.deldelPf[j][jj](x[i])

    #     # Compute the Hessian of the trial solution for each
    #     # input point.
    #     deldelYt = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelYt[i, j, jj] = deldelA[i, j, jj] + \
    #                     P[i]*deldelN[i, j, jj] + delP[i, jj]*delN[i, j] + \
    #                     delP[i, j]*delN[i, jj] + deldelP[i, j, jj]*N[i]

    #     return deldelYt

    def run_laplacian_debug(self, x):
        """Compute the trained Laplacian (debug version)."""
        n = len(x)
        m = len(x[0])
        H = len(self.w[0])
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = dsigma_dz(z[i, k])

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = d2sigma_dz2(z[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        del2N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])

        return del2Yt


    # Internal methods below this point

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()
        self.eq = eq
        self.tf = Diff2DTrialFunction(eq.bcf, eq.delbcf, eq.del2bcf)
        m = len(eq.bcf)
        self.w = np.zeros((m, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Pre-vectorize functions for efficiency.

        # Gather trial function derivatives.
        # self.delAf = (self.__dA_dxf, self.__dA_dtf)
        # self.deldelAf = ((self.__d2A_dxdxf, self.__d2A_dxdtf),
        #                  (self.__d2A_dtdxf, self.__d2A_dtdtf))
        # self.delPf = (self.__dP_dxf, self.__dP_dtf)
        # self.deldelPf = ((self.__d2P_dxdxf, self.__d2P_dxdtf),
        #                  (self.__d2P_dtdxf, self.__d2P_dtdtf))

        # <HACK>
        self.nit = 0
        self.res = None
        # </HACK>

    def __str__(self):
        s = ''
        s += "NNPDEDIFF2D:\n"
        s += "%s\n" % self.eq
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    # def __train_delta(self, x, opts=DEFAULT_OPTS):
    #     """Train using the delta method, improved with numpy vector ops."""

    #     my_opts = dict(DEFAULT_OPTS)
    #     my_opts.update(opts)

    #     # Sanity-check arguments.
    #     assert x.any()
    #     assert my_opts['maxepochs'] > 0
    #     assert my_opts['eta'] > 0
    #     assert my_opts['vmin'] < my_opts['vmax']
    #     assert my_opts['wmin'] < my_opts['wmax']
    #     assert my_opts['umin'] < my_opts['umax']

    #     # ---------------------------------------------------------------------

    #     # Change notation for convenience.
    #     n = len(x)
    #     m = 3  # <HACK>
    #     H = my_opts['nhid']

    #     # Create the hidden node weights, biases, and output node weights.
    #     self.w = np.random.uniform(my_opts['wmin'], my_opts['wmax'], (m, H))
    #     self.u = np.random.uniform(my_opts['umin'], my_opts['umax'], H)
    #     self.v = np.random.uniform(my_opts['vmin'], my_opts['vmax'], H)

    #     # Initial parameter deltas are 0.
    #     dE_dw = np.zeros((m, H))
    #     dE_du = np.zeros(H)
    #     dE_dv = np.zeros(H)

    #     # Create local references to the parameter arrays for convenience.
    #     # THESE ARE REFERENCES, NOT COPIES.
    #     w = self.w
    #     u = self.u
    #     v = self.v

    #     # Train the network.
    #     for epoch in range(my_opts['maxepochs']):
    #         if my_opts['debug']:
    #             print('Starting epoch %d.' % epoch)

    #         # Compute the new values of the network parameters.
    #         w -= my_opts['eta']*dE_dw
    #         u -= my_opts['eta']*dE_du
    #         v -= my_opts['eta']*dE_dv

    #         # Compute the net input, the sigmoid function and its derivatives,
    #         # for each hidden node and each training point.
    #         z = np.dot(x, w) + u
    #         s = sigma_v(z)
    #         s1 = dsigma_dz_v(z)
    #         s2 = d2sigma_dz2_v(z)
    #         s3 = d3sigma_dz3_v(z)

    #         # Compute the network output and its derivatives, for each
    #         # training point.
    #         N = np.dot(s, v)
    #         delN = np.dot(s1, (v*w).T)

    #         deldelN = np.zeros((n, m, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     deldelN[i, j, jj] = np.dot(v, s2[i]*w[j]*w[jj])

    #         dN_dw = v[np.newaxis, np.newaxis, :] * \
    #                 s1[:, np.newaxis, :]*x[:, :, np.newaxis]
    #         dN_du = v*s1
    #         dN_dv = np.copy(s)

    #         d2N_dwdx = np.zeros((n, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     d2N_dwdx[i, j, jj] = v*(s1[i]*kdelta(j, jj) +
    #                                             s2[i]*w[jj]*x[i, j])

    #         d2N_dudx = v[np.newaxis, np.newaxis, :] * \
    #                    s2[:, np.newaxis, :]*w[np.newaxis, :, :]
    #         d2N_dvdx = s1[:, np.newaxis, :]*w[np.newaxis, :, :]

    #         d3N_dwdxdy = np.zeros((n, m, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     for jjj in range(m):
    #                         d3N_dwdxdy[i, j, jj, jjj] = \
    #                         v * (s2[i]*(w[jjj]*kdelta(j, jj) + \
    #                              w[jj]*kdelta(j, jjj)) + \
    #                              s3[i]*w[jj]*w[jjj]*x[i, j])

    #         d3N_dudxdy = np.zeros((n, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     d3N_dudxdy[i, j, jj] = v*s3[i]*w[j]*w[jj]

    #         d3N_dvdxdy = np.zeros((n, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     d3N_dvdxdy[i, j, jj] = s2[i]*w[j]*w[jj]

    #         # Compute the value of the trial solution and its derivatives,
    #         # for each training point.
    #         A = np.zeros(n)
    #         for i in range(n):
    #             A[i] = self.__Af(x[i])

    #         delA = np.zeros((n, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 delA[i, j] = self.delAf[j](x[i])

    #         deldelA = np.zeros((n, m, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     deldelA[i, j, jj] = self.deldelAf[j][jj](x[i])

    #         P = np.zeros(n)
    #         for i in range(n):
    #             P[i] = self.__Pf(x[i])

    #         delP = np.zeros((n, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 delP[i, j] = self.delPf[j](x[i])

    #         deldelP = np.zeros((n, m, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     deldelP[i, j, jj] = self.deldelPf[j][jj](x[i])

    #         Yt = np.zeros(n)
    #         for i in range(n):
    #             Yt[i] = self.__Ytf(x[i], N[i])

    #         delYt = delA + P[:, np.newaxis]*delN + N[:, np.newaxis]*delP

    #         deldelYt = np.zeros((n, m, m))
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelYt[:, j, jj] = deldelA[:, j, jj] + \
    #                     P*deldelN[:, j, jj] + delP[:, jj]*delN[:, j] + \
    #                     delP[:, j]*delN[:, jj] + deldelP[:, j, jj]*N

    #         dYt_dw = P[:, np.newaxis, np.newaxis]*dN_dw
    #         dYt_du = P[:, np.newaxis]*dN_du
    #         dYt_dv = P[:, np.newaxis]*dN_dv

    #         d2Yt_dwdx = np.zeros((n, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     d2Yt_dwdx[i, j, jj] = P[i] * d2N_dwdx[i, j, jj] + \
    #                         delP[i, jj]*dN_dw[i, j]

    #         d2Yt_dudx = P[:, np.newaxis, np.newaxis]*d2N_dudx + \
    #                     delP[:, :, np.newaxis]*dN_du[:, np.newaxis, :]
    #         d2Yt_dvdx = P[:, np.newaxis, np.newaxis]*d2N_dvdx + \
    #                     delP[:, :, np.newaxis]*dN_dv[:, np.newaxis, :]

    #         d3Yt_dwdxdy = np.zeros((n, m, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     for jjj in range(m):
    #                         for k in range(H):
    #                             d3Yt_dwdxdy[i, j, jj, jjj, k] = \
    #                                 P[i]*d3N_dwdxdy[i, j, jj, jjj, k] + \
    #                                 delP[i, jjj]*d2N_dwdx[i, j, jj, k] + \
    #                                 delP[i, jj]*d2N_dwdx[i, j, jjj, k] + \
    #                                 deldelP[i, jj, jjj]*dN_dw[i, j, k]

    #         d3Yt_dudxdy = np.zeros((n, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     for k in range(H):
    #                         d3Yt_dudxdy[i, j, jj, k] = \
    #                             P[i]*d3N_dudxdy[i, j, jj, k] + \
    #                             delP[i, j]*d2N_dudx[i, jj, k] + \
    #                             delP[i, jj]*d2N_dudx[i, j, k] + \
    #                             deldelP[i, j, jj]*dN_du[i, k]

    #         d3Yt_dvdxdy = np.zeros((n, m, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     for k in range(H):
    #                         d3Yt_dvdxdy[i, j, jj, k] = \
    #                             P[i]*d3N_dvdxdy[i, j, jj, k] + \
    #                             delP[i, j]*d2N_dvdx[i, jj, k] + \
    #                             delP[i, jj]*d2N_dvdx[i, j, k] + \
    #                             deldelP[i, j, jj]*dN_dv[i, k]

    #         # Compute the value of the original differential equation
    #         # for each training point, and its derivatives.
    #         G = np.zeros(n)
    #         for i in range(n):
    #             G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], deldelYt[i])

    #         dG_dYt = np.zeros(n)
    #         for i in range(n):
    #             dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], delYt[i], deldelYt[i])

    #         dG_ddelYt = np.zeros((n, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 dG_ddelYt[i, j] = \
    #                     self.eq.dG_ddelYf[j](x[i], Yt[i], delYt[i],
    #                                          deldelYt[i])

    #         dG_ddeldelYt = np.zeros((n, m, m))
    #         for i in range(n):
    #             for j in range(m):
    #                 for jj in range(m):
    #                     dG_ddeldelYt[i, j, jj] = \
    #                         self.eq.dG_ddeldelYf[j][jj](x[i], Yt[i], delYt[i],
    #                                                     deldelYt[i])

    #         dG_dw = np.zeros((n, m, H))
    #         for i in range(n):
    #             for j in range(m):
    #                 for k in range(H):
    #                     dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
    #                     for jj in range(m):
    #                         dG_dw[i, j, k] += \
    #                             dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k]
    #                         for jjj in range(m):
    #                             dG_dw[i, j, k] += dG_ddeldelYt[i, jj, jjj] * \
    #                                 d3Yt_dwdxdy[i, j, jj, jjj, k]

    #         dG_du = np.zeros((n, H))
    #         for i in range(n):
    #             for k in range(H):
    #                 dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
    #                 for j in range(m):
    #                     dG_du[i, k] += dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
    #                     for jj in range(m):
    #                         dG_du[i, k] += dG_ddeldelYt[i, j, jj] * \
    #                             d3Yt_dudxdy[i, j, jj, k]

    #         dG_dv = np.zeros((n, H))
    #         for i in range(n):
    #             for k in range(H):
    #                 dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
    #                 for j in range(m):
    #                     dG_dv[i, k] += dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
    #                     for jj in range(m):
    #                         dG_dv[i, k] += dG_ddeldelYt[i, j, jj] * \
    #                             d3Yt_dvdxdy[i, j, jj, k]

    #         # Compute the partial derivatives of the error with respect to the
    #         # network parameters.
    #         dE_dw = np.zeros((m, H))
    #         for j in range(m):
    #             for k in range(H):
    #                 for i in range(n):
    #                     dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

    #         dE_du = np.zeros(H)
    #         for k in range(H):
    #             for i in range(n):
    #                 dE_du[k] += 2*G[i]*dG_du[i, k]

    #         dE_dv = np.zeros(H)
    #         for k in range(H):
    #             for i in range(n):
    #                 dE_dv[k] += 2*G[i]*dG_dv[i, k]

    #         # Compute the error function for this epoch.
    #         E = np.sum(G**2)
    #         if my_opts['verbose']:
    #             rmse = sqrt(E/n)
    #             print(epoch, rmse)

    def __train_delta_debug(self, x, opts=DEFAULT_OPTS):
        """Train using the delta method."""

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert x.any()
        assert my_opts['maxepochs'] > 0
        assert my_opts['eta'] > 0
        assert my_opts['vmin'] < my_opts['vmax']
        assert my_opts['wmin'] < my_opts['wmax']
        assert my_opts['umin'] < my_opts['umax']

        # Determine the number of training points, and change notation for
        # convenience.
        n = len(x)  # Number of training points
        m = len(self.eq.bcf)   # Number of dimensions in a training point
        H = my_opts['nhid']   # Number of hidden nodes
        debug = my_opts['debug']
        verbose = my_opts['verbose']
        eta = my_opts['eta']  # Learning rate
        maxepochs = my_opts['maxepochs']  # Number of training epochs
        wmin = my_opts['wmin']  # Network parameter limits
        wmax = my_opts['wmax']
        umin = my_opts['umin']
        umax = my_opts['umax']
        vmin = my_opts['vmin']
        vmax = my_opts['vmax']

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, (m, H))
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if verbose:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            for j in range(m):
                for k in range(H):
                    w[j, k] -= eta*dE_dw[j, k]

            for k in range(H):
                u[k] -= eta*dE_du[k]

            for k in range(H):
                v[k] -= eta*dE_dv[k]

            # Compute the net input, the sigmoid function and its
            # derivatives, for each hidden node and each training point.
            z = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    z[i, k] = u[k]
                    for j in range(m):
                        z[i, k] += w[j, k]*x[i, j]

            s = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s[i, k] = sigma(z[i, k])

            s1 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s1[i, k] = dsigma_dz(z[i, k])

            s2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s2[i, k] = d2sigma_dz2(z[i, k])

            s3 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s3[i, k] = d3sigma_dz3(z[i, k])

            # Compute the network output and its derivatives, for each
            # training point.
            N = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    N[i] += s[i, k]*v[k]

            delN = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        delN[i, j] += v[k]*s1[i, k]*w[j, k]

            del2N = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

            dN_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dN_dw[i, j, k] = v[k]*s1[i, k]*x[i, j]

            dN_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_du[i, k] = v[k]*s1[i, k]

            dN_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_dv[i, k] = s[i, k]

            d2N_dwdx = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d2N_dwdx[i, j, jj, k] = \
                                v[k]*(s1[i, k]*kdelta(j, jj) + \
                                    s2[i, k]*w[jj, k]*x[i, j])

            d2N_dudx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2N_dudx[i, j, k] = v[k]*s2[i, k]*w[j, k]

            d2N_dvdx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2N_dvdx[i, j, k] = s1[i, k]*w[j, k]

            d3N_dwdx2 = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d3N_dwdx2[i, j, jj, k] = \
                                v[k]*(2*s2[i, k]*w[jj, k]*kdelta(j, jj) + \
                                    s3[i, k]*w[j, k]**2*x[i, j])

            d3N_dudx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3N_dudx2[i, j, k] = v[k]*s3[i, k]*w[j, k]**2

            d3N_dvdx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3N_dvdx2[i, j, k] = s2[i, k]*w[j, k]**2

            # Compute the value of the trial solution, its coefficients,
            # and derivatives, for each training point.

            P = np.zeros(n)
            for i in range(n):
                P[i] = self.tf.Pf(x[i])

            delP = np.zeros((n, m))
            for i in range(n):
                delP[i] = self.tf.delPf(x[i])

            del2P = np.zeros((n, m))
            for i in range(n):
                del2P[i] = self.tf.del2Pf(x[i])

            Yt = np.zeros(n)
            for i in range(n):
                Yt[i] = self.tf.Ytf(x[i], N[i])

            delYt = np.zeros((n, m))
            for i in range(n):
                delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])

            del2Yt = np.zeros((n, m))
            for i in range(n):
                del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])

            dYt_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dYt_dw[i, j, k] = P[i]*dN_dw[i, j, k]

            dYt_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_du[i, k] = P[i]*dN_du[i, k]

            dYt_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_dv[i, k] = P[i]*dN_dv[i, k]

            d2Yt_dwdx = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d2Yt_dwdx[i, j, jj, k] = \
                                P[i]*d2N_dwdx[i, j, jj, k] + \
                                    delP[i, jj]*dN_dw[i, j, k]

            d2Yt_dudx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2Yt_dudx[i, j, k] = \
                            P[i]*d2N_dudx[i, j, k] + delP[i, j]*dN_du[i, k]

            d2Yt_dvdx = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d2Yt_dvdx[i, j, k] = \
                            P[i]*d2N_dvdx[i, j, k] + delP[i, j]*dN_dv[i, k]

            d3Yt_dwdx2 = np.zeros((n, m, m, H))
            for i in range(n):
                for j in range(m):
                    for jj in range(m):
                        for k in range(H):
                            d3Yt_dwdx2[i, j, jj, k] = \
                                P[i]*d3N_dwdx2[i, j, jj, k] + \
                                2*delP[i, jj]*d2N_dwdx[i, j, jj, k] + \
                                del2P[i, jj]*dN_dw[i, j, k]

            d3Yt_dudx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3Yt_dudx2[i, j, k] = \
                            P[i]*d3N_dudx2[i, j, k] + \
                            2*delP[i, j]*d2N_dudx[i, j, k] + \
                            del2P[i, j]*dN_du[i, k]

            d3Yt_dvdx2 = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        d3Yt_dvdx2[i, j, k] = \
                            P[i]*d3N_dvdx2[i, j, k] + \
                            2*delP[i, j]*d2N_dvdx[i, j, k] + \
                            del2P[i, j]*dN_dv[i, k]

            # Compute the value of the original differential equation
            # for each training point, and its derivatives.
            G = np.zeros(n)
            for i in range(n):
                G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], del2Yt[i])

            dG_dYt = np.zeros(n)
            for i in range(n):
                dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], delYt[i],
                                           del2Yt[i])

            dG_ddelYt = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dG_ddelYt[i, j] = \
                        self.eq.dG_ddelYf[j](x[i], Yt[i], delYt[i],
                                             del2Yt[i])

            dG_ddel2Yt = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    dG_ddel2Yt[i, j] = \
                        self.eq.dG_ddel2Yf[j](x[i], Yt[i], delYt[i],
                                              del2Yt[i])

            dG_dw = np.zeros((n, m, H))
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
                        for jj in range(m):
                            dG_dw[i, j, k] += \
                                dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k] + \
                                    dG_ddel2Yt[i, jj]*d3Yt_dwdx2[i, j, jj, k]

            dG_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                    for j in range(m):
                        dG_du[i, k] += \
                            dG_ddelYt[i, j]*d2Yt_dudx[i, j, k] \
                            + dG_ddel2Yt[i, j] * d3Yt_dudx2[i, j, k]

            dG_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                    for j in range(m):
                        dG_dv[i, k] += \
                            dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k] \
                            + dG_ddel2Yt[i, j] * d3Yt_dvdx2[i, j, k]

            # Compute the error function for this epoch.
            E2 = 0
            for i in range(n):
                E2 += G[i]**2
            if verbose:
                rmse = sqrt(E2/n)
                print(epoch, rmse)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            dE_dw = np.zeros((m, H))
            for j in range(m):
                for k in range(H):
                    for i in range(n):
                        dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

            dE_du = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_du[k] += 2*G[i]*dG_du[i, k]

            dE_dv = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_dv[k] += 2*G[i]*dG_dv[i, k]

        # Save the optimized parameters.
        self.w = w
        self.u = u
        self.v = v

    def __train_minimize(self, x, trainalg, opts=DEFAULT_OPTS, options=None):
        """Train using the scipy minimize() function"""

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert x.any()
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        callback = None
        if my_opts['verbose']:
            callback = self.__print_progress

        # ---------------------------------------------------------------------

        # Create the hidden node weights, biases, and output node weights.
        m = len(self.eq.bcf)
        H = my_opts['nhid']
        self.w = np.random.uniform(my_opts['wmin'], my_opts['wmax'], (m, H))
        self.u = np.random.uniform(my_opts['umin'], my_opts['umax'], H)
        self.v = np.random.uniform(my_opts['vmin'], my_opts['vmax'], H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((self.w.flatten(), self.u, self.v))

        res = None
        if my_opts['use_hessian']:
            pass
            # hess = self.__compute_error_hessian_debug
            # jac = self.__compute_error_gradient_debug
        elif my_opts['use_jacobian']:
            pass
            jac = self.__compute_error_gradient_debug
            res = minimize(self.__compute_error_debug, p, method=trainalg, args=(x),
                           jac=jac, options=options, callback=callback)
        else:
            res = minimize(self.__compute_error_debug, p, method=trainalg,
                           args=(x), jac=None, hess=None,
                           options=options, callback=callback)

        if my_opts['verbose']:
            print('res =', res)
        self.res = res

        # Unpack the optimized network parameters.
        for j in range(m):
            self.w[j] = res.x[j*H:(j + 1)*H]
        self.u = res.x[(m - 1)*H:m*H]
        self.v = res.x[m*H:(m + 1)*H]

    # def __compute_error(self, p, x):
    #     """Compute the current error in the trained solution."""

    #     # Unpack the network parameters.
    #     n = len(x)
    #     m = len(x[0])
    #     H = int(len(p)/(m + 2))  # HACK!
    #     w = np.zeros((m, H))
    #     (w[0], w[1], u, v) = np.hsplit(p, 4)

    #     # Compute the forward pass through the network.

    #     # Weighted inputs and transfer functions and derivatives.
    #     z = np.dot(x, w) + u
    #     s = sigma_v(z)
    #     s1 = dsigma_dz_v(z)
    #     s2 = d2sigma_dz2_v(z)

    #     # Network output and derivatives.
    #     N = np.dot(s, v)
    #     delN = np.dot(s1, (v*w).T)

    #     deldelN = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelN[i, j, jj] = np.dot(v, s2[i]*w[j]*w[jj])

    #     # Trial function BC term and derivatives
    #     A = np.zeros(n)
    #     for i in range(n):
    #         A[i] = self.__Af(x[i])

    #     delA = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delA[i, j] = self.delAf[j](x[i])

    #     deldelA = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelA[i, j, jj] = self.deldelAf[j][jj](x[i])

    #     # Trial function coefficient term and derivatives
    #     P = np.zeros(n)
    #     for i in range(n):
    #         P[i] = self.__Pf(x[i])

    #     delP = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delP[i, j] = self.delPf[j](x[i])

    #     deldelP = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelP[i, j, jj] = self.deldelPf[j][jj](x[i])

    #     # Trial function and derivatives
    #     Yt = np.zeros(n)
    #     for i in range(n):
    #         Yt[i] = self.__Ytf(x[i], N[i])

    #     delYt = delA + P[:, np.newaxis]*delN + N[:, np.newaxis]*delP

    #     deldelYt = np.zeros((n, m, m))
    #     for j in range(m):
    #         for jj in range(m):
    #             deldelYt[:, j, jj] = deldelA[:, j, jj] + \
    #                 P*deldelN[:, j, jj] + delP[:, jj]*delN[:, j] + \
    #                 delP[:, j]*delN[:, jj] + deldelP[:, j, jj]*N

    #     # Differential equation
    #     G = np.zeros(n)
    #     for i in range(n):
    #         G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], deldelYt[i])

    #     # Sum of squared error
    #     E = np.sum(G**2)
    #     return E

    def __compute_error_debug(self, p, x):
        """Compute the current error in the trained solution."""

        # Unpack the network parameters.
        n = len(x)
        m = len(x[0])
        H = int(len(p)/(m + 2))
        w = np.zeros((m, H))
        for j in range(m):
            w[j] = p[j*H:(j + 1)*H]
        u = p[(m- 1)*H:m*H]
        v = p[m*H:(m + 1)*H]


        # Weighted inputs and transfer functions and derivatives.
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = dsigma_dz(z[i, k])

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = d2sigma_dz2(z[i, k])

        # Network output and derivatives.
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        del2N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

        # Trial function and derivatives
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Ytf(x[i], N[i])

        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])

        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])

        # Differential equation
        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], del2Yt[i])

        E2 = 0
        for i in range(n):
            E2 += G[i]**2

        return E2

    # def __compute_error_gradient(self, p, x):
    #     """Compute the error gradient in the trained solution."""

    #     # Unpack the network parameters.
    #     n = len(x)
    #     m = len(x[0])
    #     H = int(len(p)/(m + 2))  # HACK!
    #     w = np.zeros((m, H))
    #     (w[0], w[1], u, v) = np.hsplit(p, 4)

    #     # Compute the forward pass through the network.

    #     # Weighted inputs and transfer functions and derivatives.
    #     z = np.dot(x, w) + u
    #     s = sigma_v(z)
    #     s1 = dsigma_dz_v(z)
    #     s2 = d2sigma_dz2_v(z)
    #     s3 = d3sigma_dz3_v(z)

    #     # Network output and derivatives.
    #     N = np.dot(s, v)
    #     delN = np.dot(s1, (v*w).T)
    #     dN_dw = v[np.newaxis, np.newaxis, :] * \
    #             s1[:, np.newaxis, :]*x[:, :, np.newaxis]

    #     deldelN = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelN[i, j, jj] = np.dot(v, s2[i]*w[j]*w[jj])

    #     dN_dw = v[np.newaxis, np.newaxis, :] * \
    #             s1[:, np.newaxis, :]*x[:, :, np.newaxis]
    #     dN_du = v*s1
    #     dN_dv = np.copy(s)

    #     d2N_dwdx = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d2N_dwdx[i, j, jj] = v*(s1[i]*kdelta(j, jj) +
    #                                         s2[i]*w[jj]*x[i, j])

    #     d2N_dudx = v[np.newaxis, np.newaxis, :] * \
    #                 s2[:, np.newaxis, :]*w[np.newaxis, :, :]
    #     d2N_dvdx = s1[:, np.newaxis, :]*w[np.newaxis, :, :]

    #     d3N_dwdxdy = np.zeros((n, m, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for jjj in range(m):
    #                     d3N_dwdxdy[i, j, jj, jjj] = \
    #                     v * (s2[i]*(w[jjj]*kdelta(j, jj) + \
    #                             w[jj]*kdelta(j, jjj)) + \
    #                             s3[i]*w[jj]*w[jjj]*x[i, j])

    #     d3N_dudxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d3N_dudxdy[i, j, jj] = v*s3[i]*w[j]*w[jj]

    #     d3N_dvdxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d3N_dvdxdy[i, j, jj] = s2[i]*w[j]*w[jj]


    #     delA = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delA[i, j] = self.delAf[j](x[i])

    #     deldelA = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelA[i, j, jj] = self.deldelAf[j][jj](x[i])

    #     P = np.zeros(n)
    #     for i in range(n):
    #         P[i] = self.__Pf(x[i])

    #     delP = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delP[i, j] = self.delPf[j](x[i])

    #     deldelP = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelP[i, j, jj] = self.deldelPf[j][jj](x[i])

    #     Yt = np.zeros(n)
    #     for i in range(n):
    #         Yt[i] = self.__Ytf(x[i], N[i])

    #     delYt = delA + P[:, np.newaxis]*delN + N[:, np.newaxis]*delP

    #     deldelYt = np.zeros((n, m, m))
    #     for j in range(m):
    #         for jj in range(m):
    #             deldelYt[:, j, jj] = deldelA[:, j, jj] + \
    #                 P*deldelN[:, j, jj] + delP[:, jj]*delN[:, j] + \
    #                 delP[:, j]*delN[:, jj] + deldelP[:, j, jj]*N

    #     dYt_dw = P[:, np.newaxis, np.newaxis]*dN_dw
    #     dYt_du = P[:, np.newaxis]*dN_du
    #     dYt_dv = P[:, np.newaxis]*dN_dv

    #     d2Yt_dwdx = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d2Yt_dwdx[i, j, jj] = P[i] * d2N_dwdx[i, j, jj] + \
    #                     delP[i, jj]*dN_dw[i, j]

    #     d2Yt_dudx = P[:, np.newaxis, np.newaxis]*d2N_dudx + \
    #                 delP[:, :, np.newaxis]*dN_du[:, np.newaxis, :]
    #     d2Yt_dvdx = P[:, np.newaxis, np.newaxis]*d2N_dvdx + \
    #                 delP[:, :, np.newaxis]*dN_dv[:, np.newaxis, :]

    #     d3Yt_dwdxdy = np.zeros((n, m, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for jjj in range(m):
    #                     for k in range(H):
    #                         d3Yt_dwdxdy[i, j, jj, jjj, k] = \
    #                             P[i]*d3N_dwdxdy[i, j, jj, jjj, k] + \
    #                             delP[i, jjj]*d2N_dwdx[i, j, jj, k] + \
    #                             delP[i, jj]*d2N_dwdx[i, j, jjj, k] + \
    #                             deldelP[i, jj, jjj]*dN_dw[i, j, k]

    #     d3Yt_dudxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for k in range(H):
    #                     d3Yt_dudxdy[i, j, jj, k] = \
    #                         P[i]*d3N_dudxdy[i, j, jj, k] + \
    #                         delP[i, j]*d2N_dudx[i, jj, k] + \
    #                         delP[i, jj]*d2N_dudx[i, j, k] + \
    #                         deldelP[i, j, jj]*dN_du[i, k]

    #     d3Yt_dvdxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for k in range(H):
    #                     d3Yt_dvdxdy[i, j, jj, k] = \
    #                         P[i]*d3N_dvdxdy[i, j, jj, k] + \
    #                         delP[i, j]*d2N_dvdx[i, jj, k] + \
    #                         delP[i, jj]*d2N_dvdx[i, j, k] + \
    #                         deldelP[i, j, jj]*dN_dv[i, k]

    #     G = np.zeros(n)
    #     for i in range(n):
    #         G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], deldelYt[i])

    #     dG_dYt = np.zeros(n)
    #     for i in range(n):
    #         dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], delYt[i], deldelYt[i])

    #     dG_ddelYt = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             dG_ddelYt[i, j] = \
    #                 self.eq.dG_ddelYf[j](x[i], Yt[i], delYt[i],
    #                                      deldelYt[i])

    #     dG_ddeldelYt = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 dG_ddeldelYt[i, j, jj] = \
    #                     self.eq.dG_ddeldelYf[j][jj](x[i], Yt[i], delYt[i],
    #                                                 deldelYt[i])

    #     dG_dw = np.zeros((n, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(H):
    #                 dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
    #                 for jj in range(m):
    #                     dG_dw[i, j, k] += \
    #                         dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k]
    #                     for jjj in range(m):
    #                         dG_dw[i, j, k] += dG_ddeldelYt[i, jj, jjj] * \
    #                             d3Yt_dwdxdy[i, j, jj, jjj, k]

    #     dG_du = np.zeros((n, H))
    #     for i in range(n):
    #         for k in range(H):
    #             dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
    #             for j in range(m):
    #                 dG_du[i, k] += dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
    #                 for jj in range(m):
    #                     dG_du[i, k] += dG_ddeldelYt[i, j, jj] * \
    #                         d3Yt_dudxdy[i, j, jj, k]

    #     dG_dv = np.zeros((n, H))
    #     for i in range(n):
    #         for k in range(H):
    #             dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
    #             for j in range(m):
    #                 dG_dv[i, k] += dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
    #                 for jj in range(m):
    #                     dG_dv[i, k] += dG_ddeldelYt[i, j, jj] * \
    #                         d3Yt_dvdxdy[i, j, jj, k]

    #     dE_dw = np.zeros((m, H))
    #     for j in range(m):
    #         for k in range(H):
    #             for i in range(n):
    #                 dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

    #     dE_du = np.zeros(H)
    #     for k in range(H):
    #         for i in range(n):
    #             dE_du[k] += 2*G[i]*dG_du[i, k]

    #     dE_dv = np.zeros(H)
    #     for k in range(H):
    #         for i in range(n):
    #             dE_dv[k] += 2*G[i]*dG_dv[i, k]

    #     jac = np.hstack((dE_dw[0], dE_dw[1], dE_du, dE_dv))
    #     return jac

    def __compute_error_gradient_debug(self, p, x):
        """Compute the error gradient in the trained solution."""

        # Unpack the network parameters.
        n = len(x)
        m = len(x[0])
        H = int(len(p)/(m + 2))
        w = np.zeros((m, H))
        for j in range(m):
            w[j] = p[j*H:(j + 1)*H]
        u = p[(m- 1)*H:m*H]
        v = p[m*H:(m + 1)*H]

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = u[k]
                for j in range(m):
                    z[i, k] += w[j, k]*x[i, j]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = dsigma_dz(z[i, k])

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = d2sigma_dz2(z[i, k])

        s3 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s3[i, k] = d3sigma_dz3(z[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        delN = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    delN[i, j] += v[k]*s1[i, k]*w[j, k]

        del2N = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    del2N[i, j] += v[k]*s2[i, k]*w[j, k]**2

        dN_dw = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    dN_dw[i, j, k] = v[k]*s1[i, k]*x[i, j]

        dN_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_du[i, k] = v[k]*s1[i, k]

        dN_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_dv[i, k] = s[i, k]

        d2N_dwdx = np.zeros((n, m, m, H))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    for k in range(H):
                        d2N_dwdx[i, j, jj, k] = \
                            v[k]*(s1[i, k]*kdelta(j, jj) + \
                                s2[i, k]*w[jj, k]*x[i, j])

        d2N_dudx = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d2N_dudx[i, j, k] = v[k]*s2[i, k]*w[j, k]

        d2N_dvdx = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d2N_dvdx[i, j, k] = s1[i, k]*w[j, k]

        d3N_dwdx2 = np.zeros((n, m, m, H))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    for k in range(H):
                        d3N_dwdx2[i, j, jj, k] = \
                            v[k]*(2*s2[i, k]*w[jj, k]*kdelta(j, jj) + \
                                s3[i, k]*w[j, k]**2*x[i, j])

        d3N_dudx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3N_dudx2[i, j, k] = v[k]*s3[i, k]*w[j, k]**2

        d3N_dvdx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3N_dvdx2[i, j, k] = s2[i, k]*w[j, k]**2

        P = np.zeros(n)
        for i in range(n):
            P[i] = self.tf.Pf(x[i])

        delP = np.zeros((n, m))
        for i in range(n):
            delP[i] = self.tf.delPf(x[i])

        del2P = np.zeros((n, m))
        for i in range(n):
            del2P[i] = self.tf.del2Pf(x[i])

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Ytf(x[i], N[i])

        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])

        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])

        dYt_dw = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    dYt_dw[i, j, k] = P[i]*dN_dw[i, j, k]

        dYt_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dYt_du[i, k] = P[i]*dN_du[i, k]

        dYt_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dYt_dv[i, k] = P[i]*dN_dv[i, k]

        d2Yt_dwdx = np.zeros((n, m, m, H))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    for k in range(H):
                        d2Yt_dwdx[i, j, jj, k] = \
                            P[i]*d2N_dwdx[i, j, jj, k] + \
                                delP[i, jj]*dN_dw[i, j, k]

        d2Yt_dudx = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d2Yt_dudx[i, j, k] = \
                        P[i]*d2N_dudx[i, j, k] + delP[i, j]*dN_du[i, k]

        d2Yt_dvdx = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d2Yt_dvdx[i, j, k] = \
                        P[i]*d2N_dvdx[i, j, k] + delP[i, j]*dN_dv[i, k]

        d3Yt_dwdx2 = np.zeros((n, m, m, H))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    for k in range(H):
                        d3Yt_dwdx2[i, j, jj, k] = \
                            P[i]*d3N_dwdx2[i, j, jj, k] + \
                            2*delP[i, jj]*d2N_dwdx[i, j, jj, k] + \
                            del2P[i, jj]*dN_dw[i, j, k]

        d3Yt_dudx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3Yt_dudx2[i, j, k] = \
                        P[i]*d3N_dudx2[i, j, k] + \
                        2*delP[i, j]*d2N_dudx[i, j, k] + \
                        del2P[i, j]*dN_du[i, k]

        d3Yt_dvdx2 = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    d3Yt_dvdx2[i, j, k] = \
                        P[i]*d3N_dvdx2[i, j, k] + \
                        2*delP[i, j]*d2N_dvdx[i, j, k] + \
                        del2P[i, j]*dN_dv[i, k]

        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], del2Yt[i])

        dG_dYt = np.zeros(n)
        for i in range(n):
            dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], delYt[i],
                                        del2Yt[i])

        dG_ddelYt = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dG_ddelYt[i, j] = \
                    self.eq.dG_ddelYf[j](x[i], Yt[i], delYt[i],
                                            del2Yt[i])

        dG_ddel2Yt = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dG_ddel2Yt[i, j] = \
                    self.eq.dG_ddel2Yf[j](x[i], Yt[i], delYt[i],
                                            del2Yt[i])

        dG_dw = np.zeros((n, m, H))
        for i in range(n):
            for j in range(m):
                for k in range(H):
                    dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
                    for jj in range(m):
                        dG_dw[i, j, k] += \
                            dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k] + \
                                dG_ddel2Yt[i, jj]*d3Yt_dwdx2[i, j, jj, k]

        dG_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                for j in range(m):
                    dG_du[i, k] += \
                        dG_ddelYt[i, j]*d2Yt_dudx[i, j, k] \
                        + dG_ddel2Yt[i, j] * d3Yt_dudx2[i, j, k]

        dG_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                for j in range(m):
                    dG_dv[i, k] += \
                        dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k] \
                        + dG_ddel2Yt[i, j] * d3Yt_dvdx2[i, j, k]

        dE_dw = np.zeros((m, H))
        for j in range(m):
            for k in range(H):
                for i in range(n):
                    dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

        dE_du = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_du[k] += 2*G[i]*dG_du[i, k]

        dE_dv = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2*G[i]*dG_dv[i, k]

        jac = np.hstack((dE_dw.flatten(), dE_du, dE_dv))
        return jac

    # def __compute_error_and_jacobian(self, p, x):
    #     """Compute the squared error and its gradient (Jacobian)."""

    #     # Unpack the network parameters.
    #     n = len(x)
    #     m = len(x[0])
    #     H = int(len(p)/(m + 2))  # HACK!
    #     w = np.zeros((m, H))
    #     (w[0], w[1], u, v) = np.hsplit(p, 4)

    #     # Compute the forward pass through the network.

    #     # Weighted inputs and transfer functions and derivatives.
    #     z = np.dot(x, w) + u
    #     s = sigma_v(z)
    #     s1 = dsigma_dz_v(z)
    #     s2 = d2sigma_dz2_v(z)
    #     s3 = d3sigma_dz3_v(z)

    #     # Network output and derivatives.
    #     N = np.dot(s, v)
    #     delN = np.dot(s1, (v*w).T)
    #     dN_dw = v[np.newaxis, np.newaxis, :] * \
    #             s1[:, np.newaxis, :]*x[:, :, np.newaxis]

    #     deldelN = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelN[i, j, jj] = np.dot(v, s2[i]*w[j]*w[jj])

    #     dN_dw = v[np.newaxis, np.newaxis, :] * \
    #             s1[:, np.newaxis, :]*x[:, :, np.newaxis]
    #     dN_du = v*s1
    #     dN_dv = np.copy(s)

    #     d2N_dwdx = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d2N_dwdx[i, j, jj] = v*(s1[i]*kdelta(j, jj) +
    #                                         s2[i]*w[jj]*x[i, j])

    #     d2N_dudx = v[np.newaxis, np.newaxis, :] * \
    #                 s2[:, np.newaxis, :]*w[np.newaxis, :, :]
    #     d2N_dvdx = s1[:, np.newaxis, :]*w[np.newaxis, :, :]

    #     d3N_dwdxdy = np.zeros((n, m, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for jjj in range(m):
    #                     d3N_dwdxdy[i, j, jj, jjj] = \
    #                     v * (s2[i]*(w[jjj]*kdelta(j, jj) + \
    #                             w[jj]*kdelta(j, jjj)) + \
    #                             s3[i]*w[jj]*w[jjj]*x[i, j])

    #     d3N_dudxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d3N_dudxdy[i, j, jj] = v*s3[i]*w[j]*w[jj]

    #     d3N_dvdxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d3N_dvdxdy[i, j, jj] = s2[i]*w[j]*w[jj]


    #     delA = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delA[i, j] = self.delAf[j](x[i])

    #     deldelA = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelA[i, j, jj] = self.deldelAf[j][jj](x[i])

    #     P = np.zeros(n)
    #     for i in range(n):
    #         P[i] = self.__Pf(x[i])

    #     delP = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             delP[i, j] = self.delPf[j](x[i])

    #     deldelP = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 deldelP[i, j, jj] = self.deldelPf[j][jj](x[i])

    #     Yt = np.zeros(n)
    #     for i in range(n):
    #         Yt[i] = self.__Ytf(x[i], N[i])

    #     delYt = delA + P[:, np.newaxis]*delN + N[:, np.newaxis]*delP

    #     deldelYt = np.zeros((n, m, m))
    #     for j in range(m):
    #         for jj in range(m):
    #             deldelYt[:, j, jj] = deldelA[:, j, jj] + \
    #                 P*deldelN[:, j, jj] + delP[:, jj]*delN[:, j] + \
    #                 delP[:, j]*delN[:, jj] + deldelP[:, j, jj]*N

    #     dYt_dw = P[:, np.newaxis, np.newaxis]*dN_dw
    #     dYt_du = P[:, np.newaxis]*dN_du
    #     dYt_dv = P[:, np.newaxis]*dN_dv

    #     d2Yt_dwdx = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 d2Yt_dwdx[i, j, jj] = P[i] * d2N_dwdx[i, j, jj] + \
    #                     delP[i, jj]*dN_dw[i, j]

    #     d2Yt_dudx = P[:, np.newaxis, np.newaxis]*d2N_dudx + \
    #                 delP[:, :, np.newaxis]*dN_du[:, np.newaxis, :]
    #     d2Yt_dvdx = P[:, np.newaxis, np.newaxis]*d2N_dvdx + \
    #                 delP[:, :, np.newaxis]*dN_dv[:, np.newaxis, :]

    #     d3Yt_dwdxdy = np.zeros((n, m, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for jjj in range(m):
    #                     for k in range(H):
    #                         d3Yt_dwdxdy[i, j, jj, jjj, k] = \
    #                             P[i]*d3N_dwdxdy[i, j, jj, jjj, k] + \
    #                             delP[i, jjj]*d2N_dwdx[i, j, jj, k] + \
    #                             delP[i, jj]*d2N_dwdx[i, j, jjj, k] + \
    #                             deldelP[i, jj, jjj]*dN_dw[i, j, k]

    #     d3Yt_dudxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for k in range(H):
    #                     d3Yt_dudxdy[i, j, jj, k] = \
    #                         P[i]*d3N_dudxdy[i, j, jj, k] + \
    #                         delP[i, j]*d2N_dudx[i, jj, k] + \
    #                         delP[i, jj]*d2N_dudx[i, j, k] + \
    #                         deldelP[i, j, jj]*dN_du[i, k]

    #     d3Yt_dvdxdy = np.zeros((n, m, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 for k in range(H):
    #                     d3Yt_dvdxdy[i, j, jj, k] = \
    #                         P[i]*d3N_dvdxdy[i, j, jj, k] + \
    #                         delP[i, j]*d2N_dvdx[i, jj, k] + \
    #                         delP[i, jj]*d2N_dvdx[i, j, k] + \
    #                         deldelP[i, j, jj]*dN_dv[i, k]

    #     G = np.zeros(n)
    #     for i in range(n):
    #         G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], deldelYt[i])

    #     E2 = np.sum(G**2)

    #     dG_dYt = np.zeros(n)
    #     for i in range(n):
    #         dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], delYt[i], deldelYt[i])

    #     dG_ddelYt = np.zeros((n, m))
    #     for i in range(n):
    #         for j in range(m):
    #             dG_ddelYt[i, j] = \
    #                 self.eq.dG_ddelYf[j](x[i], Yt[i], delYt[i],
    #                                      deldelYt[i])

    #     dG_ddeldelYt = np.zeros((n, m, m))
    #     for i in range(n):
    #         for j in range(m):
    #             for jj in range(m):
    #                 dG_ddeldelYt[i, j, jj] = \
    #                     self.eq.dG_ddeldelYf[j][jj](x[i], Yt[i], delYt[i],
    #                                                 deldelYt[i])

    #     dG_dw = np.zeros((n, m, H))
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(H):
    #                 dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
    #                 for jj in range(m):
    #                     dG_dw[i, j, k] += \
    #                         dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k]
    #                     for jjj in range(m):
    #                         dG_dw[i, j, k] += dG_ddeldelYt[i, jj, jjj] * \
    #                             d3Yt_dwdxdy[i, j, jj, jjj, k]

    #     dG_du = np.zeros((n, H))
    #     for i in range(n):
    #         for k in range(H):
    #             dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
    #             for j in range(m):
    #                 dG_du[i, k] += dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
    #                 for jj in range(m):
    #                     dG_du[i, k] += dG_ddeldelYt[i, j, jj] * \
    #                         d3Yt_dudxdy[i, j, jj, k]

    #     dG_dv = np.zeros((n, H))
    #     for i in range(n):
    #         for k in range(H):
    #             dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
    #             for j in range(m):
    #                 dG_dv[i, k] += dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
    #                 for jj in range(m):
    #                     dG_dv[i, k] += dG_ddeldelYt[i, j, jj] * \
    #                         d3Yt_dvdxdy[i, j, jj, k]

    #     dE_dw = np.zeros((m, H))
    #     for j in range(m):
    #         for k in range(H):
    #             for i in range(n):
    #                 dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

    #     dE_du = np.zeros(H)
    #     for k in range(H):
    #         for i in range(n):
    #             dE_du[k] += 2*G[i]*dG_du[i, k]

    #     dE_dv = np.zeros(H)
    #     for k in range(H):
    #         for i in range(n):
    #             dE_dv[k] += 2*G[i]*dG_dv[i, k]

    #     jac = np.hstack((dE_dw[0], dE_dw[1], dE_du, dE_dv))
    #     return E2, jac

    def __print_progress(self, xk):
        """Callback to print progress message from optimizer"""
        print('nit =', self.nit)
        self.nit += 1
        # print('xk =', xk)


if __name__ == '__main__':

    # Create training data.

    # Training point counts in each dimension (skip boundaries).
    nx = 3
    ny = 4
    nt = 5

    # Training point intervals
    dx = 1/nx
    dy = 1/ny
    dt = 1/nt

    # Training points (drop boundaries)
    xt = np.linspace(0, 1, nx + 2)[1:-1]
    yt = np.linspace(0, 1, ny + 2)[1:-1]
    tt = np.linspace(0, 1, nt + 2)[1:-1]
    x_train = np.zeros((nx*ny*nt, 3)) # HACK
    l = 0
    for k in range(nt):
        for j in range(ny):
            for i in range(nx):
                x_train[l,0] = xt[i]
                x_train[l,1] = yt[j]
                x_train[l,2] = tt[k]
                l += 1                
    n = len(x_train)

    # Options for np.minimize()
    minimize_options = {}
    # Set for convergence report.
    minimize_options['disp'] = True

    # Options for training
    training_opts = {}
    training_opts['debug'] = False
    training_opts['verbose'] = False

    # Test each training algorithm on each equation.
    for pde in ('diff2d_0', 'diff2d_1'):
        print('Examining %s.' % pde)
        pde2diff2d = PDE2DIFF2D(pde)
        m = len(pde2diff2d.bcf)
        net = NNPDE2DIFF2D(pde2diff2d)
        Ya = None
        if net.eq.Yaf is not None:
            Ya = np.zeros(n)
            for i in range(n):
                Ya[i] = net.eq.Yaf(x_train[i])
            # print('The analytical solution is:')
            # print('Ya =', Ya.reshape(nt, ny, nx))
            # print()
        delYa = None
        if net.eq.delYaf is not None:
            delYa = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    delYa[i, j] = net.eq.delYaf[j](x_train[i])
            # print('The analytical gradient is:')
            # print('delYa =', delYa)
            # print()
        del2Ya = None
        if net.eq.del2Yaf is not None:
            del2Ya = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    del2Ya[i, j] = net.eq.del2Yaf[j](x_train[i])
            # print('The analytical Laplacian is:')
            # print('del2Ya =', del2Ya)
            # print()

        # Test each training algorithm.

        print("The following methods do not use a jacobian or a hessian.")
        training_opts['use_jacobian'] = False
        training_opts['use_hessian'] = False
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',):
        # for trainalg in ('BFGS',):
            print('Training using %s algorithm.' % trainalg)
            np.random.seed(0)
            net.nit = 0
            try:
                net.train(x_train, trainalg=trainalg,
                          opts = training_opts, options=minimize_options)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, pde))
                print(e)
                print()
                continue
            Yt = net.run_debug(x_train)
            # print('The trained solution is:')
            # print('Yt =', Yt.reshape(nt, ny, nx))
            # print()
            # if Ya is not None:
                # print('The error in the trained solution is:')
                # print('Yt - Ya =', (Yt - Ya).reshape(nt, ny, nx))
                # print()
            delYt = net.run_gradient_debug(x_train)
            # print('The trained gradient is:')
            # print('delYt =', delYt.reshape(m, nt, ny, nx))
            # print()
            # if delYa is not None:
            #     print('The error in the trained gradient is:')
            #     print('delYt - delYa =', (delYt - delYa).reshape(m, nt, ny, nx))
            #     print()
            del2Yt = net.run_laplacian_debug(x_train)
            # print('The trained Laplacian is:')
            # print('del2Yt =', del2Yt.reshape(m, nt, ny, nx))
            # print()
            # if del2Ya is not None:
            #     print('The error in the trained Laplacian is:')
            #     print('del2Yt - del2Ya =',
            #           (del2Yt - del2Ya).reshape(m, nt, ny, nx))

        print("The following methods use a jacobian but not a hessian.")
        training_opts['use_jacobian'] = True
        training_opts['use_hessian'] = False
        for trainalg in ('CG', 'BFGS', 'Newton-CG'):
            print('Training using %s algorithm.' % trainalg)
            np.random.seed(0)
            net.nit = 0
            try:
                net.train(x_train, trainalg=trainalg,
                          opts = training_opts, options=minimize_options)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, pde))
                print(e)
                print()
                continue
            Yt = net.run_debug(x_train)
            # print('The trained solution is:')
            # print('Yt =', Yt.reshape(nt, ny, nx))
            # print()
            # if Ya is not None:
            #     print('The error in the trained solution is:')
            #     print('Yt - Ya =', (Yt - Ya).reshape(nt, ny, nx))
            #     print()
            delYt = net.run_gradient_debug(x_train)
            # print('The trained gradient is:')
            # print('delYt =', delYt.reshape(m, nt, ny, nx))
            # print()
            # if delYa is not None:
            #     print('The error in the trained gradient is:')
            #     print('delYt - delYa =', (delYt - delYa).reshape(m, nt, ny, nx))
            #     print()
            del2Yt = net.run_laplacian_debug(x_train)
            # print('The trained Laplacian is:')
            # print('del2Yt =', del2Yt.reshape(m, nt, ny, nx))
            # print()
            # if del2Ya is not None:
            #     print('The error in the trained Laplacian is:')
            #     print('del2Yt - del2Ya =',
            #           (del2Yt - del2Ya).reshape(m, nt, ny, nx))

        # THIS CODE IS NOT READY YET.
        # print("The following methods use a jacobian and a hessian.")
        # training_opts['use_hessian'] = True
        # training_opts['use_jacobian'] = True
        # for trainalg in ('Newton-CG',):
        #     print('Training using %s algorithm.' % trainalg)
        #     np.random.seed(0)
        #     net.nit = 0
        #     try:
        #         net.train(x_train, trainalg=trainalg,
        #                   opts = training_opts, options=minimize_options)
        #     except (OverflowError, ValueError) as e:
        #         print('Error using %s algorithm on %s!' % (trainalg, pde))
        #         print(e)
        #         print()
        #         continue
        #     Yt = net.run_debug(x_train)
        #     print('The trained solution is:')
        #     print('Yt =', Yt.reshape(nt, ny, nx))
        #     print()
        #     if Ya is not None:
        #         print('The error in the trained solution is:')
        #         print('Yt - Ya =', (Yt - Ya).reshape(nt, ny, nx))
        #         print()
        #     delYt = net.run_gradient_debug(x_train)
        #     print('The trained gradient is:')
        #     print('delYt =', delYt.reshape(m, nt, ny, nx))
        #     print()
        #     if delYa is not None:
        #         print('The error in the trained gradient is:')
        #         print('delYt - delYa =', (delYt - delYa).reshape(m, nt, ny, nx))
        #         print()
        #     del2Yt = net.run_laplacian_debug(x_train)
        #     print('The trained Laplacian is:')
        #     print('del2Yt =', del2Yt.reshape(m, nt, ny, nx))
        #     print()
        #     if del2Ya is not None:
        #         print('The error in the trained Laplacian is:')
        #         print('del2Yt - del2Ya =',
        #               (del2Yt - del2Ya).reshape(m, nt, ny, nx))

###############################################################################
"""
NNPDE2DIFF2D - Class to solve 1-, 2-, and 3D diffusion problems using a 
neural network

This module provides the functionality to solve 1-, 2-, and 3-D diffusion
problems using a neural network.

Example:
    Create an empty NNPDE2DIFF object.
        net = NNPDE2DIFF()
    Create an NNPDE2DIFF object for a PDE2DIFF1D object.
        net = NNPDE2DIFF(pde2diff1d_obj)
    Create an NNPDE2DIFF object for a PDE2DIFF2D object.
        net = NNPDE2DIFF(pde2diff2d_obj)
    Create an NNPDE2DIFF object for a PDE2DIFF3D object.
        net = NNPDE2DIFF(pde2diff3d_obj)
    Create an NNPDE2DIFF object for a PDE2DIFF2D object, with 20 hidden
              nodes.
        net = NNPDE2DIFF(pde2diff2d_obj, nhid=20)

Attributes:
    TBD

Methods:
    TBD

Todo:
    * Expand base functionality.
"""

from importlib import import_module
from math import sqrt
import numpy as np
from scipy.optimize import minimize
import sys
import types

from diff1dtrialfunction import Diff1DTrialFunction
from diff2dtrialfunction import Diff2DTrialFunction
from diff3dtrialfunction import Diff3DTrialFunction
from kdelta import kdelta
from pde2diff import PDE2DIFF
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from slffnn import SLFFNN


# Default values for method parameters
DEFAULT_DEBUG = False
DEFAULT_ETA = 0.1
DEFAULT_MAXEPOCHS = 1000
DEFAULT_NHID = 10
DEFAULT_TRAINALG = 'delta'
DEFAULT_UMAX = 1
DEFAULT_UMIN = -1
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
    'verbose':      DEFAULT_VERBOSE,
    'vmax':         DEFAULT_VMAX,
    'vmin':         DEFAULT_VMIN,
    'wmax':         DEFAULT_WMAX,
    'wmin':         DEFAULT_WMIN
    }


# Vectorize sigma functions for speed.
sigma_v = np.vectorize(sigma)
dsigma_dz_v = np.vectorize(dsigma_dz)
d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
d3sigma_dz3_v = np.vectorize(d3sigma_dz3)


class NNPDE2DIFF(SLFFNN):
    """Solve a diffusion problem with a neural network"""


    # Public methods


    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS,
              options=None):
        """Train the network to solve a 2-D diffusion problem"""
        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        if trainalg == 'delta':
            self.__train_delta(x, opts=my_opts)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            self.__train_minimize(x, trainalg, opts=my_opts, options=options)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(1)

    def run(self, x):
        """Compute the trained solution."""

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        # Compute the activation for each input point and hidden node.
        z = np.dot(x, w) + u

        # Compute the sigma function for each input point and hidden node.
        s = sigma_v(z)

        # Compute the network output for each input point.
        N = np.dot(s, v)

        # Compute the value of the trial function for each input point.
        n = len(x)
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.tf.Ytf(x[i], N[i])

        # Return the trial function values for each input point.
        return Yt

    def run_gradient(self, x):
        """Compute the trained gradient."""

        # Fetch the number n of input points at which to calculate the
        # output, and the number m of components of each point.
        n = len(x)
        m = len(x[0])

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        # Compute the activation for each input point and hidden node.
        z = x.dot(w) + u

        # Compute the sigma function for each input point and hidden node.
        s = sigma_v(z)

        # Compute the sigma function 1st derivative for each input point
        # and hidden node.
        s1 = dsigma_dz_v(z)

        # Compute the network output for each input point.
        N = s.dot(v)

        # Compute the network output gradient for each input point.
        delN = np.dot(s1, (w*v).T)

        # Compute the gradient of the trial solution for each input point.
        delYt = np.zeros((n, m))
        for i in range(n):
            delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])

        return delYt

    def run_laplacian(self, x):
        """Compute the trained Laplacian."""

        # Fetch the number n of input points at which to calculate the
        # output, and the number m of components of each point.
        n = len(x)
        m = len(x[0])

        # Get references to the network parameters for convenience.
        w = self.w
        u = self.u
        v = self.v

        # Compute the net input, the sigmoid function and its
        # derivatives, for each hidden node and each training point.
        z = x.dot(w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)

        # Compute the network output and its derivatives, for each
        # training point.
        N = s.dot(v)
        delN = s1.dot((w*v).T)
        del2N = s2.dot((w**2*v).T)

        # Compute the Laplacian components for the trial function.
        del2Yt = np.zeros((n, m))
        for i in range(n):
            del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])

        return del2Yt


    # Internal methods below this point

    def __init__(self, eq, nhid=DEFAULT_NHID):
        self.eq = eq
        m = len(eq.bcf)
        if m == 2:
            self.tf = Diff1DTrialFunction(eq.bcf, eq.delbcf, eq.del2bcf)
        elif m == 3:
            self.tf = Diff2DTrialFunction(eq.bcf, eq.delbcf, eq.del2bcf)
        elif m == 4:
            self.tf = Diff3DTrialFunction(eq.bcf, eq.delbcf, eq.del2bcf)
        else:
            print("Unexpected problem dimensionality: %s!", m)
            exit(1)
        
        # If the supplied equation object has optimized versions of the
        # boundary condition function and derivatives, use them.
        pdemod = import_module(eq.name)
        if hasattr(pdemod, 'Af'):
            print("Using optimized Af().")
            self.tf.Af = pdemod.Af
        if hasattr(pdemod, 'delAf'):
            print("Using optimized delAf().")
            self.tf.delAf = pdemod.delAf
        if hasattr(pdemod, 'del2Af'):
            print("Using optimized del2Af().")
            self.tf.del2Af = pdemod.del2Af

        # Create the weight and bias arrays.
        self.w = np.zeros((m, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Initialize results from minimize().
        self.nit = 0
        self.res = None

    def __str__(self):
        s = ''
        s += "NNPDEDIFF:\n"
        s += "%s\n" % self.eq
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def __train_delta(self, x, opts=DEFAULT_OPTS):
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

        # This small identity matrix is used during the computation of
        # some of the derivatives below.
        kd = np.identity(m)
        kd = kd[np.newaxis, :, :, np.newaxis]

        # Train the network for the specified number of epochs.
        for epoch in range(maxepochs):
            if verbose:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            w -= eta*dE_dw
            u -= eta*dE_du
            v -= eta*dE_dv

            # Compute the node activation, the sigmoid function and its
            # derivatives, for each hidden node and each training point.
            z = x.dot(w) + u
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)
            s3 = d3sigma_dz3_v(z)

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(v)
            delN = s1.dot((w*v).T)
            del2N = s2.dot((w**2*v).T)
            dN_dw = v*s1[:, np.newaxis, :]*x[:, :, np.newaxis]
            dN_du = v*s1
            dN_dv = s
            d2N_dwdx = v[np.newaxis, np.newaxis, np.newaxis, :]* \
                       (s1[:, np.newaxis, np.newaxis, :]*kd +
                        s2[:, np.newaxis, np.newaxis, :]*w[np.newaxis, np.newaxis, :, :]*
                        x[:, :, np.newaxis, np.newaxis])
            d2N_dudx = v*s2[:, np.newaxis, :]*w[np.newaxis, :, :]
            d2N_dvdx = s1[:, np.newaxis, :]*w[np.newaxis, :, :]
            d3N_dwdx2 = v[np.newaxis, np.newaxis, np.newaxis, :]* \
                       (2*s2[:, np.newaxis, np.newaxis, :]*w[np.newaxis, np.newaxis, :, :]*kd +
                        s3[:, np.newaxis, np.newaxis, :]*w[np.newaxis, :, np.newaxis, :]**2*
                        x[:, :, np.newaxis, np.newaxis])
            d3N_dudx2 = v*s3[:, np.newaxis, :]*w[np.newaxis, :, :]**2
            d3N_dvdx2 = s2[:, np.newaxis, :]*w[np.newaxis, :, :]**2

            # Compute the value of the trial solution, its coefficients,
            # and derivatives, for each training point.
            P = np.zeros(n)
            delP = np.zeros((n, m))
            del2P = np.zeros((n, m))
            Yt = np.zeros(n)
            delYt = np.zeros((n, m))
            del2Yt = np.zeros((n, m))
            for i in range(n):
                P[i] = self.tf.Pf(x[i])
                delP[i] = self.tf.delPf(x[i])
                del2P[i] = self.tf.del2Pf(x[i])
                Yt[i] = self.tf.Ytf(x[i], N[i])
                delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])
                del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])
            dYt_dw = P[:, np.newaxis, np.newaxis]*dN_dw
            dYt_du = P[:, np.newaxis]*dN_du
            dYt_dv = P[:, np.newaxis]*dN_dv
            d2Yt_dwdx = P[:, np.newaxis, np.newaxis, np.newaxis]*d2N_dwdx + \
                        delP[:, np.newaxis, :, np.newaxis]* \
                        dN_dw[:, :, np.newaxis, :]
            d2Yt_dudx = P[:, np.newaxis, np.newaxis]*d2N_dudx + \
                        delP[:, :, np.newaxis]*dN_du[:, np.newaxis, :]
            d2Yt_dvdx = P[:, np.newaxis, np.newaxis]*d2N_dvdx + \
                        delP[:, :, np.newaxis]*dN_dv[:, np.newaxis, :]
            d3Yt_dwdx2 = P[:, np.newaxis, np.newaxis, np.newaxis]*d3N_dwdx2 + \
                         2*delP[:, np.newaxis, :, np.newaxis]*d2N_dwdx + \
                         del2P[:, np.newaxis, :, np.newaxis]* \
                         dN_dw[:, :, np.newaxis, :]
            d3Yt_dudx2 = P[:, np.newaxis, np.newaxis]*d3N_dudx2 + \
                         2*delP[:, :, np.newaxis]*d2N_dudx + \
                         del2P[:, :, np.newaxis]*dN_du[:, np.newaxis, :]
            d3Yt_dvdx2 = P[:, np.newaxis, np.newaxis]*d3N_dvdx2 + \
                         2*delP[:, :, np.newaxis]*d2N_dvdx + \
                         del2P[:, :, np.newaxis]*dN_dv[:, np.newaxis, :]

            # Compute the value of the original differential equation
            # for each training point, and its derivatives.
            G = np.zeros(n)
            dG_dYt = np.zeros(n)
            dG_ddelYt = np.zeros((n, m))
            dG_ddel2Yt = np.zeros((n, m))
            for i in range(n):
                G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], del2Yt[i])
                dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], delYt[i], del2Yt[i])
                for j in range(m):
                    dG_ddelYt[i, j] = \
                        self.eq.dG_ddelYf[j](x[i], Yt[i], delYt[i], del2Yt[i])
                    dG_ddel2Yt[i, j] = \
                        self.eq.dG_ddel2Yf[j](x[i], Yt[i], delYt[i], del2Yt[i])

            dG_dw = dG_dYt[:, np.newaxis, np.newaxis]*dYt_dw
            for i in range(n):
                for j in range(m):
                    for k in range(H):
                        for jj in range(m):
                            dG_dw[i, j, k] += \
                                dG_ddelYt[i, jj]*d2Yt_dwdx[i, j, jj, k] + \
                                dG_ddel2Yt[i, jj]*d3Yt_dwdx2[i, j, jj, k]

            dG_du = dG_dYt[:, np.newaxis]*dYt_du
            for i in range(n):
                for k in range(H):
                    for j in range(m):
                        dG_du[i, k] += \
                            dG_ddelYt[i, j]*d2Yt_dudx[i, j, k] + \
                            dG_ddel2Yt[i, j]*d3Yt_dudx2[i, j, k]

            dG_dv = dG_dYt[:, np.newaxis]*dYt_dv
            for i in range(n):
                for k in range(H):
                    for j in range(m):
                        dG_dv[i, k] += \
                            dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k] + \
                            dG_ddel2Yt[i, j]*d3Yt_dvdx2[i, j, k]

            # Compute the error function for this epoch.
            E2 = np.sum(G**2)
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

    def __train_minimize(self, x, trainalg, opts=DEFAULT_OPTS,
                         options=None):
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

        #----------------------------------------------------------------------

        # Create the hidden node weights, biases, and output node weights.
        m = len(self.eq.bcf)
        H = my_opts['nhid']
        self.w = np.random.uniform(my_opts['wmin'], my_opts['wmax'], (m, H))
        self.u = np.random.uniform(my_opts['umin'], my_opts['umax'], H)
        self.v = np.random.uniform(my_opts['vmin'], my_opts['vmax'], H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((self.w.flatten(), self.u, self.v))

        res = minimize(self.__compute_error, p, method=trainalg,
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

    def __compute_error(self, p, x):
        """Compute the current error in the trained solution."""

        # Unpack the network parameters.
        n = len(x)
        m = len(x[0])
        H = int(len(p)/(m + 2))
        w = np.zeros((m, H))
        for j in range(m):
            w[j] = p[j*H:(j + 1)*H]
        u = p[(m - 1)*H:m*H]
        v = p[m*H:(m + 1)*H]

        # Weighted inputs and transfer functions and derivatives.
        z = x.dot(w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)

        # Network output and derivatives.
        N = s.dot(v)
        delN = s1.dot((w*v).T)
        del2N = s2.dot((w**2*v).T)

        # Trial function and derivatives
        Yt = np.zeros(n)
        delYt = np.zeros((n, m))
        del2Yt = np.zeros((n, m))
        for i in range(n):
            Yt[i] = self.tf.Ytf(x[i], N[i])
            delYt[i] = self.tf.delYtf(x[i], N[i], delN[i])
            del2Yt[i] = self.tf.del2Ytf(x[i], N[i], delN[i], del2N[i])

        # Differential equation
        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], Yt[i], delYt[i], del2Yt[i])

        E2 = np.sum(G**2)

        return E2

    def __print_progress(self, xk):
        """Callback to print progress message from optimizer"""
        print('nit =', self.nit)
        self.nit += 1
        # print('xk =', xk)

#########

# Self-test code

if __name__ == '__main__':

    # Create training data.

    # Training point counts in each dimension (skip boundaries).
    nx = 11
    ny = 11
    nz = 11
    nt = 11

    # Training grid points (drop endpoints)
    xt = np.linspace(0, 1, nx)
    yt = np.linspace(0, 1, ny)
    zt = np.linspace(0, 1, nz)
    tt = np.linspace(0, 1, nt)

    # 1-D training points
    x_train1 = np.zeros((nx*nt, 2))
    ii = 0
    for l in range(nt):
        for i in range(nx):
            x_train1[ii, 0] = xt[i]
            x_train1[ii, 1] = tt[l]
            ii += 1
    n1 = len(x_train1)

    # 2-D training points
    x_train2 = np.zeros((nx*ny*nt, 3))
    ii = 0
    for l in range(nt):
        for j in range(ny):
            for i in range(nx):
                x_train2[ii, 0] = xt[i]
                x_train2[ii, 1] = yt[j]
                x_train2[ii, 2] = tt[l]
                ii += 1
    n2 = len(x_train2)

    # 3-D training points
    x_train3 = np.zeros((nx*ny*nz*nt, 4))
    ii = 0
    for l in range(nt):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x_train3[ii, 0] = xt[i]
                    x_train3[ii, 1] = yt[j]
                    x_train3[ii, 2] = zt[k]
                    x_train3[ii, 3] = tt[l]
                    ii += 1
    n3 = len(x_train3)

    # Options for scipy.optimize.minimize()
    minimize_options = {}
    minimize_options['disp'] = True  # Set for convergence report.

    # Options for training
    training_opts = DEFAULT_OPTS
    training_opts['debug'] = True
    training_opts['verbose'] = True
    
    # Test each training algorithm on each equation.
    for pde in ('diff1d_zero',):
        print('Examining %s.' % pde)

        # Read the equation definition.
        eq = PDE2DIFF(pde)

        # Fetch the dimensionality of the problem.
        m = len(eq.bcf)

        # Select the appropriate training set.
        if m == 2:
            n = n1
            x_train = x_train1
        elif m == 3:
            n = n2
            x_train = x_train2
        elif m == 4:
            n = n3
            x_train = x_train3
        else:
            print("INVALID PROBLEM DIMENSION: %s" % m)
            sys.exit(1)

        Ya = None
        if eq.Yaf is not None:
            print("Computing analytical solution.")
            Ya = np.zeros(n)
            for i in range(n):
                Ya[i] = eq.Yaf(x_train[i])

        delYa = None
        if eq.delYaf is not None:
            print("Computing analytical gradient.")
            delYa = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    delYa[i, j] = eq.delYaf[j](x_train[i])

        del2Ya = None
        if eq.del2Yaf is not None:
            print("Computing analytical Laplacian.")
            del2Ya = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    del2Ya[i, j] = eq.del2Yaf[j](x_train[i])

        for trainalg in ('BFGS',):
            print('Training using %s algorithm.' % trainalg)

            # Create and train the neural network.
            net = NNPDE2DIFF(eq)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg,
                          opts=training_opts, options=minimize_options)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, pde))
                print(e)
                print()
                continue

            # Compute the trained results.
            Yt = net.run(x_train)
            print('The trained solution is:')
            print('Yt =', Yt)
            print()
            if Ya is not None:
                print('The analytical solution is:')
                print('Ya =', Ya)
                print()
                Yt_err = Yt - Ya
                print('The error in the trained solution is:')
                print('Yt_err =', Yt_err)
                print()
                Yt_rmserr = sqrt(np.sum(Yt_err**2)/n)
                print('The RMS error of the trained solution is:', Yt_rmserr)

            delYt = net.run_gradient(x_train)
            print('The trained gradient is:')
            print('delYt =', delYt)
            print()
            if delYa is not None:
                print('The analytical gradient is:')
                print('delYa =', delYa)
                print()
                delYt_err = delYt - delYa
                print('The error in the trained gradient is:')
                print('delYt_err =', delYt_err)
                print()
                delYt_rmserr = sqrt(np.sum(delYt_err**2)/n)
                print('The RMS error of the trained gradient is:',
                      delYt_rmserr)

            del2Yt = net.run_laplacian(x_train)
            print('The trained Laplacian is:')
            print('del2Yt =', del2Yt)
            print()
            if del2Ya is not None:
                print('The analytical Laplacian is:')
                print('del2Ya =', del2Ya)
                print()
                del2Yt_err = del2Yt - del2Ya
                print('The error in the trained Laplacian is:')
                print('del2Yt_err =', del2Yt_err)
                print()
                del2Yt_rmserr = sqrt(np.sum(del2Yt_err**2)/n)
                print('The RMS error of the trained Laplacian is:',
                      del2Yt_rmserr)

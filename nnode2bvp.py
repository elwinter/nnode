"""
NNODE1IVP - Class to solve 2nd-order ordinary differential equation initial
value problems using a neural network

This module provides the functionality to solve 1st-order ordinary differential
equation initial value problems using a neural network.

Example:
    Create an empty NNODE2BVP object.
        net = NNODE2BVP()
    Create an NNODE2BVP object for a ODE2BVP object.
        net = NNODE2BVP(ode2bvp_obj)

Attributes:

Methods:

Todo:
"""


from math import sqrt
import numpy as np
from scipy.optimize import minimize

from kdelta import kdelta
from ode2bvp import ODE2BVP
import sigma
from slffnn import SLFFNN


# Default values for method parameters
DEFAULT_DEBUG = False
DEFAULT_ETA = 0.01
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
    'debug':     DEFAULT_DEBUG,
    'eta':       DEFAULT_ETA,
    'maxepochs': DEFAULT_MAXEPOCHS,
    'nhid':      DEFAULT_NHID,
    'umax':      DEFAULT_UMAX,
    'umin':      DEFAULT_UMIN,
    'verbose':   DEFAULT_VERBOSE,
    'vmax':      DEFAULT_VMAX,
    'vmin':      DEFAULT_VMIN,
    'wmax':      DEFAULT_WMAX,
    'wmin':      DEFAULT_WMIN
    }


# Vectorize sigma functions for speed.
s_v = np.vectorize(sigma.s)
s1_v = np.vectorize(sigma.s1)
s2_v = np.vectorize(sigma.s2)
s3_v = np.vectorize(sigma.s3)


class NNODE2BVP(SLFFNN):
    """Solve a 2nd-order ODE BVP with a single-layer feedforward neural network."""

    # Public methods

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()

        # Save the differential equation object.
        self.eq = eq

        # Initialize all network parameters to 0.
        self.w = np.zeros(nhid)
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Clear the result structure for minimize() calls.
        self.res = None

        # Initialize iteration counter.
        self.nit = 0

        # Pre-vectorize (_v suffix) functions for efficiency.
        self.Gf_v = np.vectorize(self.eq.Gf)
        self.dG_dYf_v = np.vectorize(self.eq.dG_dYf)
        self.dG_ddYdxf_v = np.vectorize(self.eq.dG_ddYdxf)
        self.dG_dd2Ydx2f_v = np.vectorize(self.eq.dG_dd2Ydx2f)
        self.Ytf_v = np.vectorize(self.__Ytf)
        self.dYt_dxf_v = np.vectorize(self.__dYt_dxf)
        self.d2Yt_dx2f_v = np.vectorize(self.__d2Yt_dx2f)

    def __str__(self):
        s = ''
        s += "%s\n" % self.eq.name
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network to solve a 2nd-order ODE BVP. """
        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        if trainalg == 'delta':
            self.__train_delta(x, my_opts)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG'):
            self.__train_minimize(x, trainalg, my_opts)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(1)

    def run(self, x):
        """Compute the trained solution."""
        w = self.w
        u = self.u
        v = self.v

        z = np.outer(x, w) + u
        s = s_v(z)
        N = s.dot(v)
        Yt = self.Ytf_v(x, N)

        return Yt

    def run_debug(self, x):
        """Compute the trained solution (debug version)."""
        n = len(x)
        H = len(self.v)
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = w[k]*x[i] + u[k]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.__Ytf(x[i], N[i])

        return Yt

    def run_derivative(self, x):
        """Compute the trained derivative."""
        w = self.w
        u = self.u
        v = self.v

        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        dYt_dx = self.__dYt_dxf(x, N, dN_dx)

        return dYt_dx

    def run_derivative_debug(self, x):
        """Compute the trained derivative (debug version)."""
        n = len(x)
        H = len(self.v)
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = w[k]*x[i] + u[k]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = sigma.s1(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += v[k]*s1[i, k]*w[k]

        dYt_dx = np.zeros(n)
        for i in range(n):
            dYt_dx[i] = self.__dYt_dxf(x[i], N[i], dN_dx[i], d2N_dx2[i])

        return dYt_dx

    def run_derivative2(self, x):
        """Compute the trained 2nd derivative."""
        w = self.w
        u = self.u
        v = self.v

        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        s2 = s2_v(s)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        d2N_dx2 = s2.dot(v*w**2)
        d2Yt_dx2 = self.__d2Yt_dx2f(x, N, dN_dx, d2N_dx2)

        return d2Yt_dx2

    def run_derivative2_debug(self, x):
        """Compute the trained 2nd derivative (debug version)."""
        n = len(x)
        H = len(self.v)
        w = self.w
        u = self.u
        v = self.v

        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = w[k]*x[i] + u[k]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])

        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = sigma.s1(s[i, k])

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = sigma.s2(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += v[k]*s1[i, k]*w[k]

                
        d2N_dx2 = np.zeros(n)
        for i in range(n):
            for k in range(H):
                d2N_dx2[i] += v[k]*s2[i, k]*w[k]**2

        d2Yt_dx2 = np.zeros(n)
        for i in range(n):
            d2Yt_dx2[i] = self.__d2Yt_dx2f(x[i], N[i], dN_dx[i], d2N_dx2[i])

        return d2Yt_dx2

    # Internal methods below this point

    def __Ytf(self, x, N):
        """Trial function"""
        return self.eq.bc0*(1 - x) + self.eq.bc1*x + x*(1 - x)*N

    def __dYt_dxf(self, x, N, dN_dx):
        """First derivative of trial function"""
        return -self.eq.bc0 + self.eq.bc1 + x*(1 - x)*dN_dx + (1 - 2*x)*N

    def __d2Yt_dx2f(self, x, N, dN_dx, d2N_dx2):
        """2nd derivative of trial function"""
        return x*(1 - x)*d2N_dx2 + 2*(1 - 2*x)*dN_dx - 2*N

    def __train_delta(self, x, opts=DEFAULT_OPTS):
        """Train the network using the delta method. """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert len(x) > 0
        assert opts['maxepochs'] > 0
        assert opts['eta'] > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # Determine the number of training points, and change notation for
        # convenience.
        n = len(x)  # Number of training points
        H = len(self.v)
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
        w = np.random.uniform(wmin, wmax, H)
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
        for epoch in range(my_opts['maxepochs']):
            if verbose:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            w -= eta*dE_dw
            u -= eta*dE_du
            v -= eta*dE_dv

            # Compute the input, the sigmoid function, and its derivatives, for
            # each hidden node and training point.
            # x is nx1, w, u are 1xH
            # z, s, s1, s2, s3 are nxH
            z = np.outer(x, w) + u
            s = s_v(z)
            s1 = s1_v(s)
            s2 = s2_v(s)
            s3 = s3_v(s)

            # Compute the network output and its derivatives, for each
            # training point.
            # s, v are Hx1
            # N is scalar
            N = s.dot(v)
            dN_dx = s1.dot(v*w)
            d2N_dx2 = s2.dot(v*w**2)
            dN_dw = s1*np.outer(x, v)
            dN_du = s1*v
            dN_dv = np.copy(s)
            d2N_dwdx = v*(s1 + s2*np.outer(x, w))
            d2N_dudx = v*s2*w
            d2N_dvdx = s1*w
            d3N_dwdx2 = v*(2*s2*w + s3*np.outer(x, w**2))
            d3N_dudx2 = v*s3*w**2
            d3N_dvdx2 = s2*w**2

            # Compute the value of the trial solution, its coefficients,
            # and derivatives, for each training point.
            Yt = self.Ytf_v(x, N)
            dYt_dx = self.dYt_dxf_v(x, N, dN_dx)
            d2Yt_dx2 = self.__d2Yt_dx2f(x, N, dN_dx, d2N_dx2)
            # Temporary broadcast versions of P, dP_dx, d2P_dx2. 
            P_b = np.broadcast_to(x*(1 - x), (H, n)).T
            dP_dx_b = np.broadcast_to(1 - 2*x, (H, n)).T
            d2P_dx2_b = np.broadcast_to(-2, (H, n)).T
            dYt_dw = P_b*dN_dw
            dYt_du = P_b*dN_du
            dYt_dv = P_b*dN_dv
            d2Yt_dwdx = P_b*d2N_dwdx + dP_dx_b*dN_dw
            d2Yt_dudx = P_b*d2N_dudx + dP_dx_b*dN_du
            d2Yt_dvdx = P_b*d2N_dvdx + dP_dx_b*dN_dv
            d3Yt_dwdx2 = P_b*d3N_dwdx2 + 2*dP_dx_b*d2N_dwdx + d2P_dx2_b*dN_dw
            d3Yt_dudx2 = P_b*d3N_dudx2 + 2*dP_dx_b*d2N_dudx + d2P_dx2_b*dN_du
            d3Yt_dvdx2 = P_b*d3N_dvdx2 + 2*dP_dx_b*d2N_dvdx + d2P_dx2_b*dN_dv

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = self.Gf_v(x, Yt, dYt_dx, d2Yt_dx2)
            dG_dYt = self.dG_dYf_v(x, Yt, dYt_dx, d2Yt_dx2)
            dG_ddYtdx = self.dG_ddYdxf_v(x, Yt, dYt_dx, d2Yt_dx2)
            dG_dd2Ytdx2 = self.dG_dd2Ydx2f_v(x, Yt, dYt_dx, d2Yt_dx2)
            # Temporary broadcast versions of dG_dyt and dG_dytdx.
            dG_dYt_b = np.broadcast_to(dG_dYt, (H, n)).T
            dG_ddYtdx_b = np.broadcast_to(dG_ddYtdx, (H, n)).T
            dG_dd2Ytdx2_b = np.broadcast_to(dG_dd2Ytdx2, (H, n)).T
            dG_dw = dG_dYt_b*dYt_dw + dG_ddYtdx_b*d2Yt_dwdx + dG_dd2Ytdx2_b*d3Yt_dwdx2
            dG_du = dG_dYt_b*dYt_du + dG_ddYtdx_b*d2Yt_dudx + dG_dd2Ytdx2_b*d3Yt_dudx2
            dG_dv = dG_dYt_b*dYt_dv + dG_ddYtdx_b*d2Yt_dvdx + dG_dd2Ytdx2_b*d3Yt_dvdx2

            # Compute the error function for this epoch.
            E = np.sum(G**2)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            # Temporary boradcast version of G.
            G_b = np.broadcast_to(G, (H, n)).T
            dE_dw = 2*np.sum(G_b*dG_dw, axis=0)
            dE_du = 2*np.sum(G_b*dG_du, axis=0)
            dE_dv = 2*np.sum(G_b*dG_dv, axis=0)

            # Compute RMS error for this epoch.
            rmse = sqrt(E/n)
            if opts['verbose']:
                print(epoch, rmse)

        # Save the optimized parameters.
        self.w = w
        self.u = u
        self.v = v

    def __train_delta_debug(self, x, opts=DEFAULT_OPTS):
        """Train using the delta method (debug version). """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert len(x) > 0
        assert opts['maxepochs'] > 0
        assert opts['eta'] > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # Determine the number of training points, and change notation for
        # convenience.
        n = len(x)  # Number of training points
        H = len(self.v)
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
        w = np.random.uniform(wmin, wmax, H)
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if verbose:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            for k in range(H):
                w[k] -= eta*dE_dw[k]

            for k in range(H):
                u[k] -= eta*dE_du[k]

            for k in range(H):
                v[k] -= eta*dE_dv[k]

            # Compute the input, the sigmoid function, and its derivatives, for
            # each hidden node and each training point.
            z = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    z[i, k] = w[k]*x[i] + u[k]

            s = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s[i, k] = sigma.s(z[i, k])
            
            s1 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s1[i, k] = sigma.s1(s[i, k])

            s2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s2[i, k] = sigma.s2(s[i, k])

            s3 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s3[i, k] = sigma.s3(s[i, k])
            
            # Compute the network output and its derivatives, for each
            # training point.
            N = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    N[i] += v[k]*s[i, k]
            
            dN_dx = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    dN_dx[i] += v[k]*s1[i, k]*w[k]
            
            d2N_dx2 = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    d2N_dx2[i] += v[k]*s2[i, k]*w[k]**2

            dN_dw = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_dw[i, k] = v[k]*s1[i, k]*x[i]
            
            dN_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_du[i, k] = v[k]*s1[i, k]
            
            dN_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_dv[i, k] = s[i, k]
            
            d2N_dwdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2N_dwdx[i, k] = v[k]*(s1[i, k] + s2[i, k]*w[k]*x[i])
            
            d2N_dudx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2N_dudx[i, k] = v[k]*s2[i, k]*w[k]
            
            d2N_dvdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2N_dvdx[i, k] = s1[i, k]*w[k]

            d3N_dwdx2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d3N_dwdx2[i, k] = v[k]*(2*s2[i, k]*w[k] + s3[i, k]*w[k]**2*x[i])
            
            d3N_dudx2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d3N_dudx2[i, k] = v[k]*s3[i, k]*w[k]**2
            
            d3N_dvdx2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d3N_dvdx2[i, k] = s2[i, k]*w[k]**2
            
            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            Yt = np.zeros(n)
            for i in range(n):
                Yt[i] = self.__Ytf(x[i], N[i])
            
            dYt_dx = np.zeros(n)
            for i in range(n):
                dYt_dx[i] = self.__dYt_dxf(x[i], N[i], dN_dx[i])

            d2Yt_dx2 = np.zeros(n)
            for i in range(n):
                d2Yt_dx2[i] = self.__d2Yt_dx2f(x[i], N[i], dN_dx[i], d2N_dx2[i])
            
            dYt_dw = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_dw[i, k] = x[i]*(1 - x[i])*dN_dw[i, k]
            
            dYt_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_du[i, k] = x[i]*(1 - x[i])*dN_du[i, k]

            dYt_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dYt_dv[i, k] = x[i]*(1 - x[i])*dN_dv[i, k]

            d2Yt_dwdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2Yt_dwdx[i, k] = x[i]*(1 - x[i])*d2N_dwdx[i, k] + (1 - 2*x[i])*dN_dw[i, k]
            
            d2Yt_dudx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2Yt_dudx[i, k] = x[i]*(1 - x[i])*d2N_dudx[i, k] + (1 - 2*x[i])*dN_du[i, k]

            d2Yt_dvdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2Yt_dvdx[i, k] = x[i]*(1 - x[i])*d2N_dvdx[i, k] + (1 - 2*x[i])*dN_dv[i, k]

            d3Yt_dwdx2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d3Yt_dwdx2[i, k] = x[i]*(1 - x[i])*d3N_dwdx2[i, k] + 2*(1 - 2*x[i])*d2N_dwdx[i, k] - 2*dN_dw[i, k]

            d3Yt_dudx2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d3Yt_dudx2[i, k] = x[i]*(1 - x[i])*d3N_dudx2[i, k] + 2*(1 - 2*x[i])*d2N_dudx[i, k] - 2*dN_du[i, k]

            d3Yt_dvdx2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d3Yt_dvdx2[i, k] = x[i]*(1 - x[i])*d3N_dvdx2[i, k] + 2*(1 - 2*x[i])*d2N_dvdx[i, k] - 2*dN_dv[i, k]

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = np.zeros(n)
            for i in range(n):
                G[i] = self.eq.Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

            dG_dYt = np.zeros(n)
            for i in range(n):
                dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

            dG_ddYtdx = np.zeros(n)
            for i in range(n):
                dG_ddYtdx[i] = self.eq.dG_ddYdxf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

            dG_dd2Ytdx2 = np.zeros(n)
            for i in range(n):
                dG_dd2Ytdx2[i] = self.eq.dG_dd2Ydx2f(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

            dG_dw = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dw[i, k] = dG_dYt[i]*dYt_dw[i, k] + dG_ddYtdx[i]*d2Yt_dwdx[i, k] + dG_dd2Ytdx2[i]*d3Yt_dwdx2[i, k]
            
            dG_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_du[i, k] = dG_dYt[i]*dYt_du[i, k] + dG_ddYtdx[i]*d2Yt_dudx[i, k] + dG_dd2Ytdx2[i]*d3Yt_dudx2[i, k]

            dG_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k] + dG_ddYtdx[i]*d2Yt_dvdx[i, k] + dG_dd2Ytdx2[i]*d3Yt_dvdx2[i, k]

            # Compute the error function for this epoch.
            E = 0
            for i in range(n):
                E += G[i]**2

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            dE_dw = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_dw[k] += 2*G[i]*dG_dw[i, k]
            
            dE_du = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_du[k] += 2*G[i]*dG_du[i, k]

            dE_dv = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_dv[k] += 2*G[i]*dG_dv[i, k]
  
            # Compute the RMS error for this epoch.
            rmse = sqrt(E/n)
            if opts['verbose']:
                print(epoch, rmse)

        # Save the optimized parameters.
        self.w = w
        self.u = u
        self.v = v

    def __train_minimize(self, x, trainalg, opts=DEFAULT_OPTS):
        """Train the network using the SciPy minimize() function. """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert len(x) > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # Create the hidden node weights, biases, and output node weights.
        H = len(self.v)
        wmin = my_opts['wmin']  # Network parameter limits
        wmax = my_opts['wmax']
        umin = my_opts['umin']
        umax = my_opts['umax']
        vmin = my_opts['vmin']
        vmax = my_opts['vmax']

        # Create the hidden node weights, biases, and output node weights.
        w = np.random.uniform(wmin, wmax, H)
        u = np.random.uniform(umin, umax, H)
        v = np.random.uniform(vmin, vmax, H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((w, u, v))

        # Add the status callback if requested.
        callback = None
        if my_opts['verbose']:
            callback = self.__print_progress

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            jac = None
        elif trainalg in ('Newton-CG',):
            jac = self.__compute_error_gradient
        res = minimize(self.__compute_error, p, method=trainalg, jac=jac,
                       args=(x), callback=callback)
        self.res = res

        # Unpack the optimized network parameters.
        self.w = res.x[0:H]
        self.u = res.x[H:2*H]
        self.v = res.x[2*H:3*H]

    def __compute_error(self, p, x):
        """Compute the error function using the current parameter values."""

        # Unpack the network parameters (hsplit() returns views, so no copies made).
        H = len(self.v)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        s2 = s2_v(s)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        d2N_dx2 = s2.dot(v*w**2)
        Yt = self.Ytf_v(x, N)
        dYt_dx = self.dYt_dxf_v(x, N, dN_dx)
        d2Yt_dx2 = self.d2Yt_dx2f_v(x, N, dN_dx, d2N_dx2)
        G = self.Gf_v(x, Yt, dYt_dx, d2Yt_dx2)
        E = np.sum(G**2)

        return E

    def __compute_error_debug(self, p, x):
        """Compute the error function using the current parameter values (debug version)."""

        # Fetch the number of training points.
        n = len(x)

        # Unpack the network parameters.
        H = len(self.v)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = x[i]*w[k] + u[k]
        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])
        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = sigma.s1(s[i, k])
        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = sigma.s2(s[i, k])
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]
        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += s1[i, k]*v[k]*w[k]
        d2N_dx2 = np.zeros(n)
        for i in range(n):
            for k in range(H):
                d2N_dx2[i] += v[k]*s2[i, k]*w[k]**2
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.__Ytf(x[i], N[i])
        dYt_dx = np.zeros(n)
        for i in range(n):
            dYt_dx[i] = self.__dYt_dxf(x[i], N[i], dN_dx[i])
        d2Yt_dx2 = np.zeros(n)
        for i in range(n):
            d2Yt_dx2[i] = self.__d2Yt_dx2f(x[i], N[i], dN_dx[i], d2N_dx2[i])
        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])
        E = 0
        for i in range(n):
            E += G[i]**2

        return E

    def __compute_error_gradient(self, p, x):
        """Compute the gradient of the error function wrt network
        parameters."""

        # Fetch the number of training points.
        n = len(x)

        # Unpack the network parameters (hsplit() returns views, so no copies made).
        H = len(self.v)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = s_v(z)
        s1 = s1_v(s)
        s2 = s2_v(s)
        s3 = s3_v(s)

        # WARNING: Numpy and loop code below can give different results with Newton-CG after
        # a few iterations. The differences are very slight, but they result in significantly
        # different values for the weights and biases. To avoid this for now, loop code has been
        # retained for some computations below.

        # N = s.dot(v)
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += s[i, k]*v[k]

        # dN_dx = s1.dot(v*w)
        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += s1[i, k]*v[k]*w[k]

        d2N_dx2 = s2.dot(v*w**2)

        # dN_dw = s1*np.outer(x, v)
        dN_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_dw[i, k] = s1[i, k]*x[i]*v[k]

        dN_du = s1*v
        dN_dv = s

        # d2N_dwdx = v*(s1 + s2*np.outer(x, w))
        d2N_dwdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dwdx[i, k] = v[k]*(s1[i, k] + s2[i, k]*x[i]*w[k])

        d2N_dudx = v*s2*w
        d2N_dvdx = s1*w
        d3N_dwdx2 = v*(2*s2*w + s3*np.outer(x, w**2))
        d3N_dudx2 = v*s3*w**2
        d3N_dvdx2 = s2*w**2
        Yt = self.__Ytf(x, N)
        dYt_dx = self.__dYt_dxf(x, N, dN_dx)
        d2Yt_dx2 = self.__d2Yt_dx2f(x, N, dN_dx, d2N_dx2)
        P_b = np.broadcast_to(x*(1 - x), (H, n)).T
        dP_dx_b = np.broadcast_to(1 - 2*x, (H, n)).T
        d2P_dx2_b = np.broadcast_to(-2, (H, n)).T
        dYt_dw = P_b*dN_dw
        dYt_du = P_b*dN_du
        dYt_dv = P_b*dN_dv
        d2Yt_dwdx = P_b*d2N_dwdx + dP_dx_b*dN_dw
        d2Yt_dudx = P_b*d2N_dudx + dP_dx_b*dN_du
        d2Yt_dvdx = P_b*d2N_dvdx + dP_dx_b*dN_dv
        d3Yt_dwdx2 = P_b*d3N_dwdx2 + 2*dP_dx_b*d2N_dwdx + d2P_dx2_b*dN_dw
        d3Yt_dudx2 = P_b*d3N_dudx2 + 2*dP_dx_b*d2N_dudx + d2P_dx2_b*dN_du
        d3Yt_dvdx2 = P_b*d3N_dvdx2 + 2*dP_dx_b*d2N_dvdx + d2P_dx2_b*dN_dv

        G = self.Gf_v(x, Yt, dYt_dx, d2Yt_dx2)
        dG_dYt = self.dG_dYf_v(x, Yt, dYt_dx, d2Yt_dx2)
        dG_ddYtdx = self.dG_ddYdxf_v(x, Yt, dYt_dx, d2Yt_dx2)
        dG_dd2Ytdx2 = self.dG_dd2Ydx2f_v(x, Yt, dYt_dx, d2Yt_dx2)
        # Temporary broadcast versions of dG_dyt and dG_dytdx.
        dG_dYt_b = np.broadcast_to(dG_dYt, (H, n)).T
        dG_ddYtdx_b = np.broadcast_to(dG_ddYtdx, (H, n)).T
        dG_dd2Ytdx2_b = np.broadcast_to(dG_dd2Ytdx2, (H, n)).T
        dG_dw = dG_dYt_b*dYt_dw + dG_ddYtdx_b*d2Yt_dwdx + dG_dd2Ytdx2_b*d3Yt_dwdx2
        dG_du = dG_dYt_b*dYt_du + dG_ddYtdx_b*d2Yt_dudx + dG_dd2Ytdx2_b*d3Yt_dudx2
        dG_dv = dG_dYt_b*dYt_dv + dG_ddYtdx_b*d2Yt_dvdx + dG_dd2Ytdx2_b*d3Yt_dvdx2

        G_b = np.broadcast_to(G, (H, n)).T
        dE_dw = 2*np.sum(G_b*dG_dw, axis=0)
        dE_du = 2*np.sum(G_b*dG_du, axis=0)
        dE_dv = 2*np.sum(G_b*dG_dv, axis=0)

        jac = np.hstack((dE_dw, dE_du, dE_dv))

        return jac

    def __compute_error_gradient_debug(self, p, x):
        """Compute the gradient of the error function wrt network
        parameters (debug version)."""

        # Fetch the number of training points.
        n = len(x)

        # Unpack the network parameters (hsplit() returns views, so no copies made).
        H = len(self.v)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = w[k]*x[i] + u[k]

        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma.s(z[i, k])
            
        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = sigma.s1(s[i, k])

        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = sigma.s2(s[i, k])

        s3 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s3[i, k] = sigma.s3(s[i, k])

        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]
            
        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += v[k]*s1[i, k]*w[k]
            
        d2N_dx2 = np.zeros(n)
        for i in range(n):
            for k in range(H):
                d2N_dx2[i] += v[k]*s2[i, k]*w[k]**2

        dN_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_dw[i, k] = v[k]*s1[i, k]*x[i]
            
        dN_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_du[i, k] = v[k]*s1[i, k]
            
        dN_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_dv[i, k] = s[i, k]
            
        d2N_dwdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dwdx[i, k] = v[k]*(s1[i, k] + s2[i, k]*w[k]*x[i])
            
        d2N_dudx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dudx[i, k] = v[k]*s2[i, k]*w[k]
            
        d2N_dvdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dvdx[i, k] = s1[i, k]*w[k]

        d3N_dwdx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d3N_dwdx2[i, k] = v[k]*(2*s2[i, k]*w[k] + s3[i, k]*w[k]**2*x[i])
            
        d3N_dudx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d3N_dudx2[i, k] = v[k]*s3[i, k]*w[k]**2
            
        d3N_dvdx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d3N_dvdx2[i, k] = s2[i, k]*w[k]**2

        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = self.__Ytf(x[i], N[i])

        dYt_dx = np.zeros(n)
        for i in range(n):
            dYt_dx[i] = self.__dYt_dxf(x[i], N[i], dN_dx[i])

        d2Yt_dx2 = np.zeros(n)
        for i in range(n):
            d2Yt_dx2[i] = self.__d2Yt_dx2f(x[i], N[i], dN_dx[i], d2N_dx2[i])
            
        dYt_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dYt_dw[i, k] = x[i]*(1 - x[i])*dN_dw[i, k]
            
        dYt_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dYt_du[i, k] = x[i]*(1 - x[i])*dN_du[i, k]

        dYt_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dYt_dv[i, k] = x[i]*(1 - x[i])*dN_dv[i, k]

        d2Yt_dwdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2Yt_dwdx[i, k] = x[i]*(1 - x[i])*d2N_dwdx[i, k] + (1 - 2*x[i])*dN_dw[i, k]
            
        d2Yt_dudx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2Yt_dudx[i, k] = x[i]*(1 - x[i])*d2N_dudx[i, k] + (1 - 2*x[i])*dN_du[i, k]

        d2Yt_dvdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2Yt_dvdx[i, k] = x[i]*(1 - x[i])*d2N_dvdx[i, k] + (1 - 2*x[i])*dN_dv[i, k]

        d3Yt_dwdx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d3Yt_dwdx2[i, k] = x[i]*(1 - x[i])*d3N_dwdx2[i, k] + 2*(1 - 2*x[i])*d2N_dwdx[i, k] - 2*dN_dw[i, k]

        d3Yt_dudx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d3Yt_dudx2[i, k] = x[i]*(1 - x[i])*d3N_dudx2[i, k] + 2*(1 - 2*x[i])*d2N_dudx[i, k] - 2*dN_du[i, k]

        d3Yt_dvdx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d3Yt_dvdx2[i, k] = x[i]*(1 - x[i])*d3N_dvdx2[i, k] + 2*(1 - 2*x[i])*d2N_dvdx[i, k] - 2*dN_dv[i, k]

        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

        dG_dYt = np.zeros(n)
        for i in range(n):
            dG_dYt[i] = self.eq.dG_dYf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

        dG_ddYtdx = np.zeros(n)
        for i in range(n):
            dG_ddYtdx[i] = self.eq.dG_ddYdxf(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

        dG_dd2Ytdx2 = np.zeros(n)
        for i in range(n):
            dG_dd2Ytdx2[i] = self.eq.dG_dd2Ydx2f(x[i], Yt[i], dYt_dx[i], d2Yt_dx2[i])

        dG_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_dw[i, k] = dG_dYt[i]*dYt_dw[i, k] + dG_ddYtdx[i]*d2Yt_dwdx[i, k] + dG_dd2Ytdx2[i]*d3Yt_dwdx2[i, k]
            
        dG_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_du[i, k] = dG_dYt[i]*dYt_du[i, k] + dG_ddYtdx[i]*d2Yt_dudx[i, k] + dG_dd2Ytdx2[i]*d3Yt_dudx2[i, k]

        dG_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k] + dG_ddYtdx[i]*d2Yt_dvdx[i, k] + dG_dd2Ytdx2[i]*d3Yt_dvdx2[i, k]

        dE_dw = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_dw[k] += 2*G[i]*dG_dw[i, k]
            
        dE_du = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_du[k] += 2*G[i]*dG_du[i, k]

        dE_dv = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2*G[i]*dG_dv[i, k]

        jac = np.zeros(3*H)
        for j in range(H):
            jac[j] = dE_dw[j]
        for j in range(H):
            jac[H + j] = dE_du[j]
        for j in range(H):
            jac[2*H + j] = dE_dv[j]

        return jac

    def __print_progress(self, xk):
        """Callback to print progress message from optimizer"""
        print('nit =', self.nit)
        self.nit += 1
        print('xk =', xk)

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # Create training data.
    nx = 10
    x_train = np.linspace(0, 1, nx)
    print('The training points are:\n', x_train)

    # Options for training
    training_opts = {}
    training_opts['debug'] = True
    training_opts['verbose'] = True
    training_opts['maxepochs'] = 1000

    # Test each training algorithm on each equation.
    for eq in ('eq.lagaris_03_bvp',):
        print('Examining %s.' % eq)
        ode2bvp = ODE2BVP(eq)
        print(ode2bvp)

        # (Optional) analytical solution and derivatives
        if ode2bvp.Yaf:
            Ya = np.zeros(nx)
            for i in range(nx):
                Ya[i] = ode2bvp.Yaf(x_train[i])
            print('The analytical solution at the training points is:')
            print(Ya)
        if ode2bvp.dYa_dxf:
            dYa_dx = np.zeros(nx)
            for i in range(nx):
                dYa_dx[i] = ode2bvp.dYa_dxf(x_train[i])
            print('The analytical derivative at the training points is:')
            print(dYa_dx)
        if ode2bvp.d2Ya_dx2f:
            d2Ya_dx2 = np.zeros(nx)
            for i in range(nx):
                d2Ya_dx2[i] = ode2bvp.d2Ya_dx2f(x_train[i])
            print('The analytical 2nd derivative at the training points is:')
            print(d2Ya_dx2)
        print()

        # Create and train the networks.
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG'):
            print('Training using %s algorithm.' % trainalg)
            net = NNODE2BVP(ode2bvp)
            print(net)
            np.random.seed(0)  # Use same seed for reproducibility.
            try:
                net.train(x_train, trainalg=trainalg, opts=training_opts)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, eq))
                print(e)
                continue
            print(net.res)
            print('The trained network is:')
            print(net)
            Yt = net.run(x_train)
            dYt_dx = net.run_derivative(x_train)
            d2Yt_dx2 = net.run_derivative2(x_train)
            print('The trained solution is:')
            print('Yt =', Yt)
            print('The trained derivative is:')
            print('dYt_dx =', dYt_dx)
            print('The trained 2nd derivative is:')
            print('d2Yt_dx2 =', d2Yt_dx2)

            # (Optional) Error in solution and derivative
            if ode2bvp.Yaf:
                print('The error in the trained solution is:')
                print(Yt - Ya)
            if ode2bvp.dYa_dxf:
                print('The error in the trained derivative is:')
                print(dYt_dx - dYa_dx)
            if ode2bvp.d2Ya_dx2f:
                print('The error in the trained 2nd derivative is:')
                print(d2Yt_dx2 - d2Ya_dx2)

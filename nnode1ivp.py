"""
NNODE1IVP - Class to solve 1st-order ordinary differential equation initial
value problems using a neural network

This module provides the functionality to solve 1st-order ordinary differential
equation initial value problems using a neural network.

Example:
    Create an empty NNODE1IVP object.
        net = NNODE1IVP()
    Create an NNODE1IVP object for a ODE1IVP object.
        net = NNODE1IVP(ode1ivp_obj)
    Create an NNODE1IVP object for a ODE1IVP object, with 20 hidden nodes.
        net = NNODE1IVP(ode1ivp_obj, nhid=20)

Attributes:
    None

Methods:
    train
    run
    run_derivative

Todo:
    * Expand base functionality.
    * Combine error and gradient code into a single function for speed.
"""


from math import sqrt
import numpy as np
from scipy.optimize import minimize

from ode1ivp import ODE1IVP
from sigma import sigma, dsigma_dz, d2sigma_dz2
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
sigma_v = np.vectorize(sigma)
dsigma_dz_v = np.vectorize(dsigma_dz)
d2sigma_dz2_v = np.vectorize(d2sigma_dz2)


class NNODE1IVP(SLFFNN):
    """Solve a 1st-order ODE IVP with a neural network."""

    # Public methods

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()
        self.eq = eq
        self.w = np.zeros(nhid)
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

        # Pre-vectorize functions for efficiency.
        self.Gf_v = np.vectorize(self.eq.Gf)
        self.dG_dyf_v = np.vectorize(self.eq.dG_dyf)
        self.dG_dydxf_v = np.vectorize(self.eq.dG_dydxf)
        self.ytf_v = np.vectorize(self.__ytf)
        self.dyt_dxf_v = np.vectorize(self.__dyt_dxf)

        # <HACK>
        self.nit = 0
        self.res = None
        # </HACK>

    def __str__(self):
        s = ''
        s += 'NNODE1IVP:\n'
        s += "%s\n" % self.eq
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network to solve a 1st-order ODE IVP. """
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
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        N = s.dot(self.v)
        yt = self.ytf_v(x, N)
        return yt

    def run_derivative(self, x):
        """Compute the trained derivative."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        dyt_dx = self.dyt_dxf_v(x, N, dN_dx)
        return dyt_dx

    # Internal methods below this point

    def __ytf(self, x, N):
        """Trial function"""
        return self.eq.ic + x*N

    def __dyt_dxf(self, x, N, dN_dx):
        """First derivative of trial function"""
        return x*dN_dx + N

    def __train_delta(self, x, opts=DEFAULT_OPTS):
        """Train the network using the delta method. """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert x.any()
        assert opts['maxepochs'] > 0
        assert opts['eta'] > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # ---------------------------------------------------------------------

        # Determine the number of training points, and change notation for
        # convenience.
        n = len(x)
        H = my_opts['nhid']

        # Create the hidden node weights, biases, and output node weights.
        self.w = np.random.uniform(my_opts['wmin'], my_opts['wmax'], H)
        self.u = np.random.uniform(my_opts['umin'], my_opts['umax'], H)
        self.v = np.random.uniform(my_opts['vmin'], my_opts['vmax'], H)

        # Initial parameter deltas are 0.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros(H)

        # Create local references to the parameter arrays for convenience.
        # THESE ARE REFERENCES, NOT COPIES.
        w = self.w
        u = self.u
        v = self.v

        # Train the network.
        for epoch in range(my_opts['maxepochs']):
            if my_opts['debug']:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            w -= my_opts['eta']*dE_dw
            u -= my_opts['eta']*dE_du
            v -= my_opts['eta']*dE_dv

            # Compute the input, the sigmoid function, and its
            # derivatives, for each hidden node k, for each training
            # point i.
            z = np.outer(x, w) + u
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(v)
            dN_dx = s1.dot(v*w)
            dN_dw = s1*np.outer(x, v)
            dN_du = s1*v
            dN_dv = np.copy(s)
            d2N_dwdx = v*(s1 + s2*np.outer(x, w))
            d2N_dudx = v*s2*w
            d2N_dvdx = s1*w

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            yt = self.ytf_v(x, N)
            dyt_dx = self.dyt_dxf_v(x, N, dN_dx)
            # Temporary broadcast version of x.
            x_b = np.broadcast_to(x, (H, n)).T
            dyt_dw = x_b*dN_dw
            dyt_du = x_b*dN_du
            dyt_dv = x_b*dN_dv
            d2yt_dwdx = x_b*d2N_dwdx + dN_dw
            d2yt_dudx = x_b*d2N_dudx + dN_du
            d2yt_dvdx = x_b*d2N_dvdx + dN_dv

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = self.Gf_v(x, yt, dyt_dx)
            dG_dyt = self.dG_dyf_v(x, yt, dyt_dx)
            dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx)
            # Temporary broadcast versions of dG_dyt and dG_dytdx.
            dG_dyt_b = np.broadcast_to(dG_dyt, (H, n)).T
            dG_dytdx_b = np.broadcast_to(dG_dytdx, (H, n)).T
            dG_dw = dG_dyt_b*dyt_dw + dG_dytdx_b*d2yt_dwdx
            dG_du = dG_dyt_b*dyt_du + dG_dytdx_b*d2yt_dudx
            dG_dv = dG_dyt_b*dyt_dv + dG_dytdx_b*d2yt_dvdx

            # Compute the error function for this epoch.
            E = np.sum(G**2)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            # Temporary boradcast version of G.
            G_b = np.broadcast_to(G, (H, n)).T
            dE_dw = 2*np.sum(G_b*dG_dw, axis=0)
            dE_du = 2*np.sum(G_b*dG_du, axis=0)
            dE_dv = 2*np.sum(G_b*dG_dv, axis=0)

            # Compute the RMS error for this epoch.
            rmse = sqrt(E/n)
            if opts['verbose']:
                print(epoch, rmse)

    def __train_delta_debug(self, x, opts=DEFAULT_OPTS):
        """Train the network using the delta method (debug version). """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert x.any()
        assert opts['maxepochs'] > 0
        assert opts['eta'] > 0
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # ---------------------------------------------------------------------

        # Determine the number of training points, and change notation for
        # convenience.
        n = len(x)
        H = opts['nhid']

        # Create the hidden node weights, biases, and output node weights.
        self.w = np.random.uniform(opts['wmin'], opts['wmax'], H)
        self.u = np.random.uniform(opts['umin'], opts['umax'], H)
        self.v = np.random.uniform(opts['vmin'], opts['vmax'], H)

        # Initial parameter deltas are 0.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros(H)

        # Train the network.
        for epoch in range(opts['maxepochs']):
            if opts['debug']:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            # self.w -= opts['eta']*dE_dw
            for k in range(H):
                self.w[k] -= opts['eta']*dE_dw[k]
            # self.u -= opts['eta']*dE_du
            for k in range(H):
                self.u[k] -= opts['eta']*dE_du[k]
            # self.v -= opts['eta']*dE_dv
            for k in range(H):
                self.v[k] -= opts['eta']*dE_dv[k]
                                  
            # Compute the input, the sigmoid function, and its
            # derivatives, for each hidden node k, for each training
            # point i.
            # z = np.outer(x, self.w) + self.u
            z = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    z[i, k] = x[i]*self.w[k] + self.u[k]
            # s = sigma_v(z)
            s = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s[i, k] = sigma(z[i, k])
            # s1 = dsigma_dz_v(z)
            s1 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s1[i, k] = dsigma_dz(z[i, k])
            # s2 = d2sigma_dz2_v(z)
            s2 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    s2[i, k] = d2sigma_dz2(z[i, k])

            # Compute the network output and its derivatives, for each
            # training point.
            # N = s.dot(self.v)
            N = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    N[i] += self.v[k]*s[i, k]
            # dN_dx = s1.dot(self.v*self.w)
            dN_dx = np.zeros(n)
            for i in range(n):
                for k in range(H):
                    dN_dx[i] += s1[i, k]*self.v[k]*self.w[k]
            # dN_dw = s1*np.outer(x, self.v)
            dN_dw = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_dw[i, k] = s1[i, k]*x[i]*self.v[k]
            # dN_du = s1*self.v
            dN_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dN_du[i, k] = s1[i, k]*self.v[k]
            dN_dv = np.zeros((n, H))
            # dN_dv = s
            for i in range(n):
                for k in range(H):
                    dN_dv[i, k] = s[i, k]
            # d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))
            d2N_dwdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2N_dwdx[i, k] = self.v[k]*(s1[i, k] + s2[i, k]*
                                                x[i]*self.w[k])
            # d2N_dudx = self.v*s2*self.w
            d2N_dudx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2N_dudx[i, k] = self.v[k]*s2[i, k]*self.w[k]
            # d2N_dvdx = s1*self.w
            d2N_dvdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2N_dvdx[i, k] = s1[i, k]*self.w[k]

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            # yt = self.__ytf(x, N)
            yt = np.zeros(n)
            for i in range(n):
                yt[i] = self.__ytf(x[i], N[i])
            # dyt_dx = self.__dyt_dxf(x, N, dN_dx)
            dyt_dx = np.zeros(n)
            for i in range(n):
                dyt_dx[i] = self.__dyt_dxf(x[i], N[i], dN_dx[i])
            # dyt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
            dyt_dw = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dyt_dw[i, k] = x[i]*dN_dw[i, k]
            # dyt_du = np.broadcast_to(x, (H, n)).T*dN_du
            dyt_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dyt_du[i, k] = x[i]*dN_du[i, k]
            # dyt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
            dyt_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dyt_dv[i, k] = x[i]*dN_dv[i, k]
            # d2yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
            d2yt_dwdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2yt_dwdx[i, k] = x[i]*d2N_dwdx[i, k] + dN_dw[i, k]
            # d2yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
            d2yt_dudx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2yt_dudx[i, k] = x[i]*d2N_dudx[i, k] + dN_du[i, k]
            # d2yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv
            d2yt_dvdx = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    d2yt_dvdx[i, k] = x[i]*d2N_dvdx[i, k] + dN_dv[i, k]

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            # G = self.Gf_v(x, yt, dyt_dx)
            G = np.zeros(n)
            for i in range(n):
                G[i] = self.eq.Gf(x[i], yt[i], dyt_dx[i])
            # dG_dyt = self.dG_dyf_v(x, yt, dyt_dx)
            dG_dyt = np.zeros(n)
            for i in range(n):
                dG_dyt[i] = self.eq.dG_dyf(x[i], yt[i], dyt_dx[i])
            # dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx)
            dG_dytdx = np.zeros(n)
            for i in range(n):
                dG_dytdx[i] = self.eq.dG_dydxf(x[i], yt[i], dyt_dx[i])
            # dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
            #     np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx
            dG_dw = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dw[i, k] = dG_dyt[i]*dyt_dw[i, k] + \
                                  dG_dytdx[i]*d2yt_dwdx[i, k]
            # dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
            #     np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx
            dG_du = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_du[i, k] = dG_dyt[i]*dyt_du[i, k] + \
                                  dG_dytdx[i]*d2yt_dudx[i, k]
            # dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
            #     np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx
            dG_dv = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    dG_dv[i, k] = dG_dyt[i]*dyt_dv[i, k] + \
                                  dG_dytdx[i]*d2yt_dvdx[i, k]
            # Compute the error function for this epoch.
            # E = np.sum(G**2)
            E = 0
            for i in range(n):
                E += G[i]**2

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            # dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
            dE_dw = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_dw[k] += 2*G[i]*dG_dw[i, k]
            # dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
            dE_du = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_du[k] += 2*G[i]*dG_du[i, k]
            # dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)
            dE_dv = np.zeros(H)
            for k in range(H):
                for i in range(n):
                    dE_dv[k] += 2*G[i]*dG_dv[i, k]

            # Compute the RMS error for this epoch.
            rmse = sqrt(E/n)
            if opts['verbose']:
                print(epoch, rmse)

    def __train_minimize(self, x, trainalg, opts=DEFAULT_OPTS):
        """Train the network using the SciPy minimize() function. """

        my_opts = dict(DEFAULT_OPTS)
        my_opts.update(opts)

        # Sanity-check arguments.
        assert x.any()
        assert opts['vmin'] < opts['vmax']
        assert opts['wmin'] < opts['wmax']
        assert opts['umin'] < opts['umax']

        # ---------------------------------------------------------------------

        # Create the hidden node weights, biases, and output node weights.
        H = opts['nhid']
        self.w = np.random.uniform(opts['wmin'], opts['wmax'], H)
        self.u = np.random.uniform(opts['umin'], opts['umax'], H)
        self.v = np.random.uniform(opts['vmin'], opts['vmax'], H)

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        p = np.hstack((self.w, self.u, self.v))

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            jac = None
        elif trainalg in ('Newton-CG',):
            jac = self.__compute_error_gradient
        res = minimize(self.__compute_error, p, method=trainalg, jac=jac,
                       args=(x))

        # Unpack the optimized network parameters.
        self.w = res.x[0:H]
        self.u = res.x[H:2*H]
        self.v = res.x[2*H:3*H]

    def __compute_error(self, p, x):
        """Compute the error function using the current parameter values."""

        # Unpack the network parameters (hsplit() returns views, so no copies made).
        H = len(self.w)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        yt = self.ytf_v(x, N)
        dyt_dx = self.dyt_dxf_v(x, N, dN_dx)
        G = self.Gf_v(x, yt, dyt_dx)
        E2 = np.sum(G**2)
        return E2

    def __compute_error_debug(self, p, x):
        """Compute the error function using the current parameter values (debug version)."""

        # Fetch the number of training points.
        n = len(x)

        # Unpack the network parameters.
        H = len(self.w)
        w = p[0:H]
        u = p[H:2*H]
        v = p[2*H:3*H]

        # Compute the forward pass through the network.
        # z = np.outer(x, w) + u
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = x[i]*w[k] + u[k]
        # s = sigma_v(z)
        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])
        # s1 = dsigma_dz_v(z)
        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = dsigma_dz(z[i, k])
        # N = s.dot(v)
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]
        # dN_dx = s1.dot(v*w)
        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += s1[i, k]*v[k]*w[k]
        # yt = self.ytf_v(x, N)
        yt = np.zeros(n)
        for i in range(n):
            yt[i] = self.__ytf(x[i], N[i])
        # dyt_dx = self.__dyt_dxf(x, N, dN_dx)
        dyt_dx = np.zeros(n)
        for i in range(n):
            dyt_dx[i] = self.__dyt_dxf(x[i], N[i], dN_dx[i])
        # G = self.Gf_v(x, yt, dyt_dx)
        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], yt[i], dyt_dx[i])
        # E2 = np.sum(G**2)
        E2 = 0
        for i in range(n):
            E2 += G[i]**2
        return E2

    def __compute_error_gradient(self, p, x):
        """Compute the gradient of the error function wrt network
        parameters."""

        # Compute the number of training points.
        n = len(x)

        # Unpack the network parameters (hsplit() returns views, so no copies made).
        H = len(self.w)
        (w, u, v) = np.hsplit(p, 3)

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        dN_dw = s1*np.outer(x, v)
        dN_du = s1*v
        dN_dv = s
        d2N_dwdx = v*(s1 + s2*np.outer(x, w))
        d2N_dudx = v*s2*w
        d2N_dvdx = s1*w
        yt = self.__ytf(x, N)
        dyt_dx = self.__dyt_dxf(x, N, dN_dx)
        dyt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
        dyt_du = np.broadcast_to(x, (H, n)).T*dN_du
        dyt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
        d2yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
        d2yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
        d2yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv
        G = self.Gf_v(x, yt, dyt_dx)
        dG_dyt = self.dG_dyf_v(x, yt, dyt_dx)
        dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx)
        dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
            np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx
        dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
            np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx
        dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
            np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx
        dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
        dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
        dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)
        jac = np.hstack((dE_dw, dE_du, dE_dv))
        return jac

    def __compute_error_gradient_debug(self, p, x):
        """Compute the gradient of the error function wrt network
        parameters."""

        # Fetch the number of training points.
        n = len(x)

        # Unpack the network parameters (hsplit() returns views, so no copies made).
        H = len(self.w)
        w = p[0:H]
        u = p[H:2*H]
        v = p[2*H:3*H]

        # Compute the forward pass through the network.
        # z = np.outer(x, w) + u
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = x[i]*w[k] + u[k]
        # s = sigma_v(z)
        s = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s[i, k] = sigma(z[i, k])
        # s1 = dsigma_dz_v(z)
        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s1[i, k] = dsigma_dz(z[i, k])
        # s2 = d2sigma_dz2_v(z)
        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                s2[i, k] = d2sigma_dz2(z[i, k])
        # N = s.dot(v)
        N = np.zeros(n)
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i, k]
        # dN_dx = s1.dot(v*w)
        dN_dx = np.zeros(n)
        for i in range(n):
            for k in range(H):
                dN_dx[i] += s1[i, k]*v[k]*w[k]
        # dN_dw = s1*np.outer(x, self.v)
        dN_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_dw[i, k] = s1[i, k]*x[i]*self.v[k]
        # dN_du = s1*self.v
        dN_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dN_du[i, k] = s1[i, k]*self.v[k]
            dN_dv = np.zeros((n, H))
        # dN_dv = s
        for i in range(n):
            for k in range(H):
                dN_dv[i, k] = s[i, k]
        # d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))
        d2N_dwdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dwdx[i, k] = self.v[k]*(s1[i, k] + s2[i, k]*
                                            x[i]*self.w[k])
        # d2N_dudx = self.v*s2*self.w
        d2N_dudx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dudx[i, k] = self.v[k]*s2[i, k]*self.w[k]
        # d2N_dvdx = s1*self.w
        d2N_dvdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2N_dvdx[i, k] = s1[i, k]*self.w[k]
        # yt = self.__ytf(x, N)
        yt = np.zeros(n)
        for i in range(n):
            yt[i] = self.__ytf(x[i], N[i])
        # dyt_dx = self.__dyt_dxf(x, N, dN_dx)
        dyt_dx = np.zeros(n)
        for i in range(n):
            dyt_dx[i] = self.__dyt_dxf(x[i], N[i], dN_dx[i])
        # dyt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
        dyt_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dyt_dw[i, k] = x[i]*dN_dw[i, k]
        # dyt_du = np.broadcast_to(x, (H, n)).T*dN_du
        dyt_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dyt_du[i, k] = x[i]*dN_du[i, k]
        # dyt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
        dyt_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dyt_dv[i, k] = x[i]*dN_dv[i, k]
        # d2yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
        d2yt_dwdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2yt_dwdx[i, k] = x[i]*d2N_dwdx[i, k] + dN_dw[i, k]
        # d2yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
        d2yt_dudx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2yt_dudx[i, k] = x[i]*d2N_dudx[i, k] + dN_du[i, k]
        # d2yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv
        d2yt_dvdx = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                d2yt_dvdx[i, k] = x[i]*d2N_dvdx[i, k] + dN_dv[i, k]
            G = np.zeros(n)
            for i in range(n):
                G[i] = self.eq.Gf(x[i], yt[i], dyt_dx[i])
        # dG_dyt = self.dG_dyf_v(x, yt, dyt_dx)
        dG_dyt = np.zeros(n)
        for i in range(n):
            dG_dyt[i] = self.eq.dG_dyf(x[i], yt[i], dyt_dx[i])
        # dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx)
        dG_dytdx = np.zeros(n)
        for i in range(n):
            dG_dytdx[i] = self.eq.dG_dydxf(x[i], yt[i], dyt_dx[i])
        G = np.zeros(n)
        for i in range(n):
            G[i] = self.eq.Gf(x[i], yt[i], dyt_dx[i])
        # dG_dyt = self.dG_dyf_v(x, yt, dyt_dx)
        dG_dyt = np.zeros(n)
        for i in range(n):
            dG_dyt[i] = self.eq.dG_dyf(x[i], yt[i], dyt_dx[i])
        # dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx)
        dG_dytdx = np.zeros(n)
        for i in range(n):
            dG_dytdx[i] = self.eq.dG_dydxf(x[i], yt[i], dyt_dx[i])
        # dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
        #     np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx
        dG_dw = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_dw[i, k] = dG_dyt[i]*dyt_dw[i, k] + \
                                dG_dytdx[i]*d2yt_dwdx[i, k]
        # dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
        #     np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx
        dG_du = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_du[i, k] = dG_dyt[i]*dyt_du[i, k] + \
                                dG_dytdx[i]*d2yt_dudx[i, k]
        # dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
        #     np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx
        dG_dv = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                dG_dv[i, k] = dG_dyt[i]*dyt_dv[i, k] + \
                                dG_dytdx[i]*d2yt_dvdx[i, k]
        # dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
        dE_dw = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_dw[k] += 2*G[i]*dG_dw[i, k]
        # dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
        dE_du = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_du[k] += 2*G[i]*dG_du[i, k]
        # dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)
        dE_dv = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2*G[i]*dG_dv[i, k]
        # jac = np.hstack((dE_dw, dE_du, dE_dv))
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
        #    print('xk =', xk)

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # Create training data.
    nx = 11
    x_train = np.linspace(0, 1, nx)

    # Test each training algorithm on each equation.
    for eq in ('ode1_00', 'ode1_01', 'ode1_02', 'ode1_03', 'ode1_04',
               'lagaris_01', 'lagaris_02'):
        print('Examining %s.' % eq)
        ode1ivp = ODE1IVP(eq)
        print(ode1ivp)
        print()

        # (Optional) analytical solution and derivative
        if ode1ivp.yaf:
            ya = np.vectorize(ode1ivp.yaf)(x_train)
            print('The analytical solution is:')
            print(ya)
        if ode1ivp.dya_dxf:
            dya_dx = np.vectorize(ode1ivp.dya_dxf)(x_train)
            print('The analytical derivative is:')
            print(dya_dx)
        print()

        # Create and train the networks.
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG'):
            print('Training using %s algorithm.' % trainalg)
            net = NNODE1IVP(ode1ivp)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, eq))
                print(e)
                print()
                continue
            print('The optimized network is:')
            print(net)
            yt = net.run(x_train)
            dyt_dx = net.run_derivative(x_train)
            print('The trained solution is:')
            print('yt =', yt)
            print('The trained derivative is:')
            print('dyt_dx =', dyt_dx)
            print()

            # (Optional) Error in solution and derivative
            if ode1ivp.yaf:
                print('The error in the trained solution is:')
                print(yt - ya)
            if ode1ivp.dya_dxf:
                print('The error in the trained derivative is:')
                print(dyt_dx - dya_dx)
            print()

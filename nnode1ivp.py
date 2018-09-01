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

Attributes:
    None

Methods:
    __init__
    __str__
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

# Vectorize sigma functions.
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
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS',
                          'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            self.__train_minimize(x, trainalg, my_opts)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(0)

    def run(self, x):
        """Compute the trained solution."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        N = s.dot(self.v)
        yt = self.__ytf(x, N)
        return yt

    def run_derivative(self, x):
        """Compute the trained derivative."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        dyt_dx = self.__dyt_dxf(x, N, dN_dx)
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

        #------------------------------------------------------------------------

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
            self.w -= opts['eta']*dE_dw
            self.u -= opts['eta']*dE_du
            self.v -= opts['eta']*dE_dv

            # Compute the input, the sigmoid function, and its
            # derivatives, for each hidden node k, for each training
            # point i.
            z = np.outer(x, self.w) + self.u
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(self.v)
            dN_dx = s1.dot(self.v*self.w)
            dN_dw = s1*np.outer(x, self.v)
            dN_du = s1*self.v
            dN_dv = s
            d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))
            d2N_dudx = self.v*s2*self.w
            d2N_dvdx = s1*self.w

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            yt = self.__ytf(x, N)
            dyt_dx = self.__dyt_dxf(x, N, dN_dx)
            dyt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
            dyt_du = np.broadcast_to(x, (H, n)).T*dN_du
            dyt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
            d2yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
            d2yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
            d2yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = self.Gf_v(x, yt, dyt_dx)
            dG_dyt = self.dG_dyf_v(x, yt, dyt_dx)
            dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx)
            dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
                    np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx
            dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
                    np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx
            dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
                    np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx

            # Compute the error function for this epoch.
            E = np.sum(G**2)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
            dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
            dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)

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

        #------------------------------------------------------------------------

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
        elif trainalg in ('Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            jac = self.__compute_error_gradient
        res = minimize(self.__compute_error, p, method=trainalg, jac=jac,
                       args=(x))

        # Unpack the optimized network parameters.
        self.w = res.x[0:H]
        self.u = res.x[H:2*H]
        self.v = res.x[2*H:3*H]

    def __compute_error(self, p, x):
        """Compute the error function using the current parameter values."""

        # Unpack the network parameters.
        H = len(self.w)
        w = p[0:H]
        u = p[H:2*H]
        v = p[2*H:3*H]

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        N = s.dot(v)
        yt = self.__ytf(x, N)
        dN_dx = s1.dot(v*w)
        dyt_dx = self.__dyt_dxf(x, N, dN_dx)
        G = self.Gf_v(x, yt, dyt_dx)
        E2 = np.sum(G**2)
        return E2

    def __compute_error_gradient(self, p, x):
        """Compute the gradient of the error function wrt network
        parameters."""

        # Compute the number of training points.
        n = len(x)

        # Unpack the network parameters.
        H = len(self.w)
        w = p[0:H]
        u = p[H:2*H]
        v = p[2*H:3*H]

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

#--------------------------------------------------------------------------------

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

        # (Optional) analytical solution and derivative
        if ode1ivp.yaf:
            print('The analytical solution is:')
            print(np.vectorize(ode1ivp.yaf)(x_train))
        if ode1ivp.dya_dxf:
            print('The analytical derivative is:')
            print(np.vectorize(ode1ivp.dya_dxf)(x_train))
        print()

        # Create and train the networks.
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
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

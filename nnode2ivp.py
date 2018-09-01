"""
NNODE2IVP - Class to solve 2nd-order ordinary differential equation initial
value problems using a neural network

This module provides the functionality to solve 2nd-order ordinary differential
equation initial value problems using a neural network.

Example:
    Create an empty NNODE2IVP object.
        net = NNODE2IVP()
    Create an NNODE2IVP object for a ODE2IVP object.
        net = NNODE2IVP(ode2ivp_obj)

Attributes:
    None

Methods:
    __init__
    __str__
    train
    run
    run_derivative
    run_2nd_derivative

Todo:
    * Expand base functionality.
    * Combine error and gradient code into a single function for speed.
"""

from math import sqrt
import numpy as np
from scipy.optimize import minimize

from ode2ivp import ODE2IVP
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
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
d3sigma_dz3_v = np.vectorize(d3sigma_dz3)

class NNODE2IVP(SLFFNN):
    """Solve a 2nd-order ODE IVP with a neural network."""

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
        self.dG_d2ydx2f_v = np.vectorize(self.eq.dG_dydxf)

    def __str__(self):
        s = ''
        s += "NNODE2IVP:\n"
        s += "%s\n" % self.eq
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network. """
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

    def run_2nd_derivative(self, x):
        """Compute the trained 2nd derivative."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        d2N_dx2 = s2.dot(self.v*self.w**2)
        d2yt_dx2 = self.__d2yt_dx2f(x, N, dN_dx, d2N_dx2)
        return d2yt_dx2

    # Internal methods below this point

    def __ytf(self, x, N):
        """Trial function, x and N are 1xn arrays"""
        return self.eq.ic + x*self.eq.ic1 + x**2*N

    def __dyt_dxf(self, x, N, dN_dx):
        """First derivative of trial function, all are 1xn arrays"""
        return self.eq.ic1 + x**2*dN_dx + 2*x*N

    def __d2yt_dx2f(self, x, N, dN_dx, d2N_dx2):
        """2nd derivative of trial function, all are 1xn arrays"""
        return x**2*d2N_dx2 + 4*x*dN_dx + 2*N

    def __train_delta(self, x, opts=DEFAULT_OPTS):
        """Train the network with the delta method. """

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
            z = np.outer(x, self.w) + self.u  # n x H array
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)
            s3 = d3sigma_dz3_v(z)

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(self.v)
            dN_dx = s1.dot(self.v*self.w)
            d2N_dx2 = s2.dot(self.v*self.w**2)
            dN_dw = s1*np.outer(x, self.v)
            dN_du = s1*self.v
            dN_dv = s
            d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))
            d2N_dudx = self.v*s2*self.w
            d2N_dvdx = s1*self.w
            d3N_dwdx2 = self.v*(2*s2*self.w + s3*np.outer(x, self.w**2))
            d3N_dudx2 = self.v*s3*self.w**2
            d3N_dvdx2 = s2*self.w**2

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            yt = self.__ytf(x, N)
            dyt_dx = self.__dyt_dxf(x, N, dN_dx)
            d2yt_dx2 = self.__d2yt_dx2f(x, N, dN_dx, d2N_dx2)
            dyt_dw = np.broadcast_to(x**2, (H, n)).T*dN_dw
            dyt_du = np.broadcast_to(x**2, (H, n)).T*dN_du
            dyt_dv = np.broadcast_to(x**2, (H, n)).T*dN_dv
            d2yt_dwdx = np.broadcast_to(x**2, (H, n)).T*d2N_dwdx \
                        + 2*np.broadcast_to(x, (H, n)).T*dN_dw
            d2yt_dudx = np.broadcast_to(x**2, (H, n)).T*d2N_dudx \
                        + 2*np.broadcast_to(x, (H, n)).T*dN_du
            d2yt_dvdx = np.broadcast_to(x**2, (H, n)).T*d2N_dvdx \
                        + 2*np.broadcast_to(x, (H, n)).T*dN_dv
            d3yt_dwdx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dwdx2 \
                         + 4*np.broadcast_to(x, (H, n)).T*d2N_dwdx \
                         + 2*dN_dw
            d3yt_dudx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dudx2 \
                         + 4*np.broadcast_to(x, (H, n)).T*d2N_dudx \
                         + 2*dN_du
            d3yt_dvdx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dvdx2 \
                         + 4*np.broadcast_to(x, (H, n)).T*d2N_dvdx \
                         + 2*dN_dv

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = self.Gf_v(x, yt, dyt_dx, d2yt_dx2)
            dG_dyt = self.dG_dyf_v(x, yt, dyt_dx, d2yt_dx2)
            dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx, d2yt_dx2)
            dG_d2ytdx2 = self.dG_d2ydx2f_v(x, yt, dyt_dx, d2yt_dx2)
            dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw \
                    + np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx \
                    + np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dwdx2
            dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du \
                    + np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx \
                    + np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dudx2
            dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv \
                    + np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx \
                    + np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dvdx2

            # Compute the error function for this epoch.
            E = np.sum(G**2)

            # Compute the partial derivatives of the error with respect to
            # the network parameters.
            dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
            dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
            dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)

            # Record the current RMSE.
            rmse = sqrt(E/n)
            if opts['verbose']:
                print(epoch, rmse)

    def __train_minimize(self, x, trainalg, opts=DEFAULT_OPTS):
        """Train the network with minimize(). """

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
        s2 = d2sigma_dz2_v(z)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        d2N_dx2 = s2.dot(v*w**2)
        yt = self.__ytf(x, N)
        dyt_dx = self.__dyt_dxf(x, N, dN_dx)
        d2yt_dx2 = self.__d2yt_dx2f(x, N, dN_dx, d2N_dx2)
        G = self.Gf_v(x, yt, dyt_dx, d2yt_dx2)
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
        s3 = d3sigma_dz3_v(z)
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        d2N_dx2 = s2.dot(v*w**2)
        dN_dw = s1*np.outer(x, v)
        dN_du = s1*v
        dN_dv = s
        d2N_dwdx = v*(s1 + s2*np.outer(x, w))
        d2N_dudx = v*s2*w
        d2N_dvdx = s1*w
        d3N_dwdx2 = v*(2*s2*w + s3*np.outer(x, w**2))
        d3N_dudx2 = v*s3*w**2
        d3N_dvdx2 = s2*w**2
        yt = self.__ytf(x, N)
        dyt_dx = self.__dyt_dxf(x, N, dN_dx)
        d2yt_dx2 = self.__d2yt_dx2f(x, N, dN_dx, d2N_dx2)
        dyt_dw = np.broadcast_to(x**2, (H, n)).T*dN_dw
        dyt_du = np.broadcast_to(x**2, (H, n)).T*dN_du
        dyt_dv = np.broadcast_to(x**2, (H, n)).T*dN_dv
        d2yt_dwdx = np.broadcast_to(x**2, (H, n)).T*d2N_dwdx + \
                    2*np.broadcast_to(x, (H, n)).T*dN_dw
        d2yt_dudx = np.broadcast_to(x**2, (H, n)).T*d2N_dudx + \
                    2*np.broadcast_to(x, (H, n)).T*dN_du
        d2yt_dvdx = np.broadcast_to(x**2, (H, n)).T*d2N_dvdx + \
                    2*np.broadcast_to(x, (H, n)).T*dN_dv
        d3yt_dwdx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dwdx2 + \
                     4*np.broadcast_to(x, (H, n)).T*d2N_dwdx + \
                     2*dN_dw
        d3yt_dudx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dudx2 + \
                     4*np.broadcast_to(x, (H, n)).T*d2N_dudx + \
                     2*dN_du
        d3yt_dvdx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dvdx2 + \
                     4*np.broadcast_to(x, (H, n)).T*d2N_dvdx + \
                     2*dN_dv
        G = self.Gf_v(x, yt, dyt_dx, d2yt_dx2)
        dG_dyt = self.dG_dyf_v(x, yt, dyt_dx, d2yt_dx2)
        dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx, d2yt_dx2)
        dG_d2ytdx2 = self.dG_d2ydx2f_v(x, yt, dyt_dx, d2yt_dx2)
        dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
                np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx + \
                np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dwdx2
        dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
                np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx + \
                np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dudx2
        dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
                np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx + \
                np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dvdx2
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
    for ode in ('ode2_ivp_00',):
        print('Examining %s.' % ode)
        ode2ivp = ODE2IVP(ode)
        print(ode2ivp)

        # (Optional) analytical solution and derivative
        if ode2ivp.yaf:
            print('The analytical solution is:')
            print(np.vectorize(ode2ivp.yaf)(x_train))
        if ode2ivp.dya_dxf:
            print('The analytical derivative is:')
            print(np.vectorize(ode2ivp.dya_dxf)(x_train))
        if ode2ivp.d2ya_dx2f:
            print('The analytical 2nd derivative is:')
            print(np.vectorize(ode2ivp.d2ya_dx2f)(x_train))
        print()

        # Create and train the networks.
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Training using %s algorithm.' % trainalg)
            net = NNODE2IVP(ode2ivp)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg)
            except (OverflowError,) as e:
                print('Error using %s algorithm on %s!' % (trainalg, ode))
                print(e)
                print()
                continue
            print('The optimized network is:')
            print(net)
            yt = net.run(x_train)
            dyt_dx = net.run_derivative(x_train)
            d2yt_dx2 = net.run_2nd_derivative(x_train)
            print('The trained solution is:')
            print('yt =', yt)
            print('The trained derivative is:')
            print('dyt_dx =', dyt_dx)
            print('The trained 2nd derivative is:')
            print('d2yt_dx2 =', d2yt_dx2)
            print()

"""
NNODE2BVP - Class to solve 2nd-order ordinary differential equation boundary
value problems using a neural network

This module provides the functionality to solve 2nd-order ordinary differential
equation boundary value problems using a neural network.

Example:
    Create an empty NNODE2BVP object.
        net = NNODE2BVP()
    Create an NNODE2BVP object for a ODE2BVP object.
        net = NNODE2BVP(ode2bvp_obj)

Attributes:
    None

Methods:
    __init__
    __str__
    train
    train_delta
    train_minimize
    compute_error
    compute_error_gradient
    run
    run_derivative
    run_2nd_derivative

Todo:
    * Expand base functionality.
"""

from math import sqrt
import numpy as np
from scipy.optimize import minimize

from ode2bvp import ODE2BVP
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from slffnn import SLFFNN

# Default values for method parameters
DEFAULT_CLAMP = False
DEFAULT_DEBUG = False
DEFAULT_ETA = 0.01
DEFAULT_MAXEPOCHS = 1000
DEFAULT_NHID = 10
DEFAULT_NTRAIN = 10
DEFAULT_ODE = 'ode01bvp'
DEFAULT_SEED = 0
DEFAULT_TRAINALG = 'delta'
DEFAULT_UMAX = 1
DEFAULT_UMIN = -1
DEFAULT_VERBOSE = False
DEFAULT_VMAX = 1
DEFAULT_VMIN = -1
DEFAULT_WMAX = 1
DEFAULT_WMIN = -1

# Define the trial solution for a 2nd-order ODE BVP.
def ytf(A, B, x, N):
    """Trial function"""
    return A*(1 - x) + B*x + x*(1 - x)*N

# Define the 1st trial derivative.
def dyt_dxf(A, B, x, N, dN_dx):
    """First derivative of trial function"""
    return -A + B + x*(1 - x)*dN_dx + (1 - 2*x)*N

# Define the 2nd trial derivative.
def d2yt_dx2f(A, Ap, x, N, dN_dx, d2N_dx2):
    """2nd derivative of trial function"""
    return x*(1 - x)*d2N_dx2 + 2*(1 - 2*x)*dN_dx - 2*N

# Vectorize the trial solution and derivatives.
ytf_v = np.vectorize(ytf)
dyt_dxf_v = np.vectorize(dyt_dxf)
d2yt_dx2f_v = np.vectorize(d2yt_dx2f)

# Vectorize sigma functions.
sigma_v = np.vectorize(sigma)
dsigma_dz_v = np.vectorize(dsigma_dz)
d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
d3sigma_dz3_v = np.vectorize(d3sigma_dz3)

class NNODE2BVP(SLFFNN):
    """Solve a 2nd-order ODE BVP with a neural network."""

    def __init__(self, ode2bvp, nhid=DEFAULT_NHID):
        super().__init__()

        # ODE, in the form G(x,y,dy/dx,d2y/dx2)=0.
        self.Gf = ode2bvp.Gf
        self.Gf_v = np.vectorize(self.Gf)

        # dG/dy
        self.dG_dyf = ode2bvp.dG_dyf
        self.dG_dyf_v = np.vectorize(self.dG_dyf)

        # dG/d(dy/dx)
        self.dG_dydxf = ode2bvp.dG_dydxf
        self.dG_dydxf_v = np.vectorize(self.dG_dydxf)

        # dG/d(d2y/dx2)
        self.dG_d2ydx2f = ode2bvp.dG_d2ydx2f
        self.dG_d2ydx2f_v = np.vectorize(self.dG_d2ydx2f)

        # Initial conditions
        self.bc0 = ode2bvp.bc0
        self.bc1 = ode2bvp.bc1

        # Analytical solution (optional)
        if ode2bvp.yaf:
            self.yaf = ode2bvp.yaf
            self.yaf_v = np.vectorize(self.yaf)

        # Analytical derivative (optional)
        if ode2bvp.dya_dxf:
            self.dya_dxf = ode2bvp.dya_dxf
            self.dya_dxf_v = np.vectorize(self.dya_dxf)

        # Analytical 2nd derivative (optional)
        if ode2bvp.d2ya_dx2f:
            self.d2ya_dx2f = ode2bvp.d2ya_dx2f
            self.d2ya_dx2f_v = np.vectorize(self.d2ya_dx2f)

        # Create arrays to hold the weights and biases, initially all 0.
        self.w = np.zeros(nhid)
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

    def __str__(self):
        s = "NNODE2BVP:\n"
        s += "w = %s" % self.w + "\n"
        s += "u = %s" % self.u + "\n"
        s += "v = %s" % self.v + "\n"
        return s

    def train(self,
              x,                           # x-values for training points
              trainalg=DEFAULT_TRAINALG,   # Training algorithm
              nhid=DEFAULT_NHID,           # Node count in hidden layer
              maxepochs=DEFAULT_MAXEPOCHS, # Max training epochs
              eta=DEFAULT_ETA,             # Learning rate
              clamp=DEFAULT_CLAMP,         # Turn on parameter clamping
              wmin=DEFAULT_WMIN,           # Minimum hidden weight value
              wmax=DEFAULT_WMAX,           # Maximum hidden weight value
              umin=DEFAULT_UMIN,           # Minimum hidden bias value
              umax=DEFAULT_UMAX,           # Maximum hidden bias value
              vmin=DEFAULT_VMIN,           # Minimum output weight value
              vmax=DEFAULT_VMAX,           # Maximum output weight value
              debug=DEFAULT_DEBUG,
              verbose=DEFAULT_VERBOSE
              ):
        """Train the network to solve a 2nd-order ODE BVP. """
        print('trainalg =', trainalg)
        if trainalg == 'delta':
            print('Calling self.train_delta().')
            self.train_delta(x, nhid=nhid, maxepochs=maxepochs, eta=eta,
                             clamp=clamp,
                             wmin=wmin, wmax=wmax,
                             umin=umin, umax=umax,
                             vmin=vmin, vmax=vmax,
                             debug=debug, verbose=verbose)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS',
                          'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Calling self.train_minimize().')
            self.train_minimize(x, trainalg=trainalg,
                                wmin=wmin, wmax=wmax,
                                umin=umin, umax=umax,
                                vmin=vmin, vmax=vmax,
                                debug=debug, verbose=verbose)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(0)

    def train_delta(self,
                    x,                           # x-values for training points
                    nhid=DEFAULT_NHID,           # Node count in hidden layer
                    maxepochs=DEFAULT_MAXEPOCHS, # Max training epochs
                    eta=DEFAULT_ETA,             # Learning rate
                    clamp=DEFAULT_CLAMP,         # Turn on parameter clamping
                    wmin=DEFAULT_WMIN,           # Minimum hidden weight value
                    wmax=DEFAULT_WMAX,           # Maximum hidden weight value
                    umin=DEFAULT_UMIN,           # Minimum hidden bias value
                    umax=DEFAULT_UMAX,           # Maximum hidden bias value
                    vmin=DEFAULT_VMIN,           # Minimum output weight value
                    vmax=DEFAULT_VMAX,           # Maximum output weight value
                    debug=DEFAULT_DEBUG,
                    verbose=DEFAULT_VERBOSE
                    ):
        """Train the network to solve a 2nd-order ODE BVP. """
        if debug:
            print('x =', x)
            print('nhid =', nhid)
            print('maxepochs =', maxepochs)
            print('eta =', eta)
            print('clamp =', clamp)
            print('wmin =', wmin)
            print('wmax =', wmax)
            print('umin =', umin)
            print('umax =', umax)
            print('vmin =', vmin)
            print('vmax =', vmax)
            print('debug =', debug)
            print('verbose =', verbose)

        # Copy equation characteristics for local use.
        bc0 = self.bc0
        bc1 = self.bc1
        dG_dyf = self.dG_dyf
        dG_dydxf = self.dG_dydxf
        dG_d2ydx2f = self.dG_d2ydx2f
        if debug:
            print('bc0 =', bc0)
            print('bc1 =', bc1)
            print('dG_dyf =', dG_dyf)
            print('dG_dydxf =', dG_dydxf)
            print('dG_d2ydx2f =', dG_d2ydx2f)

        # Sanity-check arguments.
        assert x.any()
        assert nhid > 0
        assert maxepochs > 0
        assert eta > 0
        assert wmin < wmax
        assert umin < umax
        assert vmin < vmax

        #------------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)

        # Change notation for convenience.
        A = bc0
        B = bc1
        H = nhid

        if debug:
            print('n =', n)
            print('A =', A)
            print('B =', B)
            print('H =', H)

        #------------------------------------------------------------------------

        # Create the network.

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        self.w = np.random.uniform(wmin, wmax, H)

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        self.u = np.random.uniform(umin, umax, H)

        # Create an array to hold the weights connecting the hidden
        # nodes to the output node. The weights are initialized with a
        # uniform random distribution.
        self.v = np.random.uniform(vmin, vmax, H)

        if debug:
            print('w =', self.w)
            print('u =', self.u)
            print('v =', self.v)

        #------------------------------------------------------------------------

        # Initial parameter deltas are 0.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if debug:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            self.w -= eta*dE_dw
            self.u -= eta*dE_du
            self.v -= eta*dE_dv
            if debug:
                print('w =', self.w)
                print('u =', self.u)
                print('v =', self.v)

            # If desired, clamp the parameter values at the limits.
            if clamp:
                self.w[self.w < wmin] = wmin
                self.w[self.w > wmax] = wmax
                self.u[self.u < umin] = umin
                self.u[self.u > umax] = umax
                self.v[self.v < vmin] = vmin
                self.v[self.v > vmax] = vmax

            #--------------------------------------------------------------------

            # Compute the input, the sigmoid function, and its
            # derivatives, for each hidden node k, for each training
            # point i. Each hidden node has 1 weight w[k] (since there
            # is only 1 input x[i] per training sample) and 1 bias
            # u[k].

            # The weighted input to each hidden node is just z=w*x+u.
            # Since x and w are 1-D ndarray objects, np.outer() is used so
            # that each x is multiplied by each w, resulting in a 2-D array
            # with n rows and H columns. The biases u[] are then added to
            # each row of the resulting 2-D array.
            z = np.outer(x, self.w) + self.u

            # Each z[i,k] gets mapped to a s[i,k], so s, s1, s2, s3
            # are all n x H ndarray objects.
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)
            s3 = d3sigma_dz3_v(z)

            if debug:
                print('z =', z)
                print('s =', s)
                print('s1 =', s1)
                print('s2 =', s2)
                print('s3 =', s3)

            #--------------------------------------------------------------------

            # Compute the network output and its derivatives, for each
            # training point.

            # The network output N[i] for each training point x[i] is
            # the sum of the outputs s[i,k], weighted by the output
            # weights v[k]. N[] is thus a 1-D ndarray object of length
            # n.
            N = s.dot(self.v)

            # Since there is only one input x[i] for each training
            # sample, and a single output value N[i], the derivative
            # dN_dx[] is also a 1-D ndarray object of length n. For
            # each sample i, the dN_dx[i] is the dot product of s1[]
            # with the product of the output weights v[] and the
            # hidden weights w[].
            dN_dx = s1.dot(self.v*self.w)

            # s1[] is n x H, x[] is 1 x n, and v[] is 1 x H, so the
            # outer product of x[] and v[] is needed to get the
            # product of each x[i] with each v[k], resulting in a n x
            # H array, which is then multiplied by s1[], which is also
            # an n x H array.
            dN_dw = s1*np.outer(x, self.v)

            # s1[] is n x H, and v[] is 1 x H, so dN_du[] is n x H.
            dN_du = s1*self.v

            # There are n network outputs N[i], and H hidden nodes, so
            # the partials of N[] wrt the hidden node weights at the
            # output is a n x H ndarray object.
            dN_dv = s

            # v[] is 1 x H, s1[] and s2[] are n x H, x[] is 1 x N, and
            # w[] is 1 x H, so the outer product of x[] and wp[ gives
            # a n x H array, which is multipled by another n x H array
            # (s2[]) and then added to another n x H array
            # (s1[]). Each row of this array is then multiplied by the
            # row vector v[].
            d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))

            # v[] is 1 x H, s2[] is n x H, and w is[] 1 x H, so each
            # row of s1[] gets multiplied by the single row v[], which
            # then multiplies the single row w[], resulting in a n x H
            # array for the 2nd partial d2N_dudx[].
            d2N_dudx = self.v*s2*self.w

            # s1[] is n x H, and w[] is 1 x H, so each row of s1[]
            # gets multiplied by the single row v[], resulting in a n
            # x H array for the 2nd partial d2N_dvdx[].
            d2N_dvdx = s1*self.w

            # Since there is only one input x[i] for each training
            # sample, and a single output value N[i], the 2nd
            # derivative d2N_dx2[] is also a 1-D ndarray object of
            # length n. For each sample i, the d2N_dx2[i] is the dot
            # product of s2[] with the product of the output weights
            # v[] and the hidden weights w[] squared.
            d2N_dx2 = s2.dot(self.v*self.w**2)
            d3N_dwdx2 = self.v*(2*s2*self.w + s3*self.w**2*x)
            d3N_dudx2 = self.v*s3*self.w**2
            d3N_dvdx2 = s2*self.w**2

            if debug:
                print('N =', N)
                print('dN_dx =', dN_dx)
                print('dN_dw =', dN_dw)
                print('dN_du =', dN_du)
                print('dN_dv =', dN_dv)
                print('d2N_dwdx =', d2N_dwdx)
                print('d2N_dudx =', d2N_dudx)
                print('d2N_dvdx =', d2N_dvdx)
                print('d2N_dx2 =', d2N_dx2)
                print('d3N_dwdx2 =', d3N_dwdx2)
                print('d3N_dudx2 =', d3N_dudx2)
                print('d3N_dvdx2 =', d3N_dvdx2)

            #--------------------------------------------------------------------

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            yt = ytf_v(A, B, x, N)
            dyt_dx = dyt_dxf_v(A, B, x, N, dN_dx)
            d2yt_dx2 = d2yt_dx2f_v(A, B, x, N, dN_dx, d2N_dx2)
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

            if debug:
                print('yt =', yt)
                print('dyt_dx =', dyt_dx)
                print('d2yt_dx2 =', d2yt_dx2)
                print('dyt_dw =', dyt_dw)
                print('dyt_du =', dyt_du)
                print('dyt_dv =', dyt_dv)
                print('d2yt_dwdx =', d2yt_dwdx)
                print('d2yt_dudx =', d2yt_dudx)
                print('d2yt_dvdx =', d2yt_dvdx)
                print('d3yt_dwdx2 =', d3yt_dwdx2)
                print('d3yt_dudx2 =', d3yt_dudx2)
                print('d3yt_dvdx2 =', d3yt_dvdx2)

            #--------------------------------------------------------------------

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
            E = sum(G**2)

            # Compute the partial derivatives of the error with respect to
            # the network parameters.
            dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis=0)
            dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis=0)
            dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis=0)

            if debug:
                print('G =', G)
                print('dG_dyt =', dG_dyt)
                print('dG_dytdx =', dG_dytdx)
                print('dG_d2ytdx2 =', dG_d2ytdx2)
                print('dG_dw =', dG_dw)
                print('dG_du =', dG_du)
                print('dG_dv =', dG_dv)
                print('E =', E)
                print('dE_dw =', dE_dw)
                print('dE_du =', dE_du)
                print('dE_dv =', dE_dv)

            #--------------------------------------------------------------------

            # Record the current RMSE.
            rmse = sqrt(E/n)
            if verbose:
                print(epoch, rmse)

    def train_minimize(self,
                       x,                         # x-values for training points
                       trainalg=DEFAULT_TRAINALG, # Training algorithm
                       wmin=DEFAULT_WMIN,         # Minimum hidden weight value
                       wmax=DEFAULT_WMAX,         # Maximum hidden weight value
                       umin=DEFAULT_UMIN,         # Minimum hidden bias value
                       umax=DEFAULT_UMAX,         # Maximum hidden bias value
                       vmin=DEFAULT_VMIN,         # Minimum output weight value
                       vmax=DEFAULT_VMAX,         # Maximum output weight value
                       debug=DEFAULT_DEBUG,
                       verbose=DEFAULT_VERBOSE
                       ):
        """Train the network to solve a 2nd-order ODE BVP. """
        if debug:
            print('x =', x)
            print('trainalg =', trainalg)
            print('wmin =', wmin)
            print('wmax =', wmax)
            print('umin =', umin)
            print('umax =', umax)
            print('vmin =', vmin)
            print('vmax =', vmax)
            print('debug =', debug)
            print('verbose =', verbose)

        #-----------------------------------------------------------------------

        # Sanity-check arguments.
        assert x.any()
        assert trainalg
        assert vmin < vmax
        assert wmin < wmax
        assert umin < umax

        #------------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)

        # Change notation for convenience.
        H = len(self.w)

        if debug:
            print('n =', n)
            print('H =', H)

        #------------------------------------------------------------------------

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        self.w = np.random.uniform(wmin, wmax, H)

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        self.u = np.random.uniform(umin, umax, H)

        # Create an array to hold the weights connecting the hidden
        # nodes to the output node. The weights are initialized with a
        # uniform random distribution.
        self.v = np.random.uniform(vmin, vmax, H)

        if debug:
            print('w =', self.w)
            print('u =', self.u)
            print('v =', self.v)

        #------------------------------------------------------------------------

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        # p = [w, u, v]
        p = np.hstack((self.w, self.u, self.v))
        if debug:
            print('p =', p)

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            res = minimize(self.compute_error, p, method=trainalg,
                           args=(x, self.bc0, self.bc1))
        elif trainalg in ('Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            res = minimize(self.compute_error, p, method=trainalg,
                           jac=self.compute_error_gradient,
                           args=(x, self.bc0, self.bc1))
        if debug:
            print(res)

        # Unpack the optimized network parameters.
        self.w = res.x[0:H]
        self.u = res.x[H:2*H]
        self.v = res.x[2*H:3*H]

    def compute_error(self, p, x, A, B):
        """Compute the error function for 1 forward pass."""

        # Compute the number of hidden nodes.
        H = len(self.w)

        # Unpack the network parameters.
        w = p[0:H]
        u = p[H:2*H]
        v = p[2*H:3*H]

        # Compute the forward pass through the network.
        z = np.outer(x, w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        N = s.dot(v)

        # Compute the trial solution, derivative, and error function.
        yt = ytf_v(A, B, x, N)
        dN_dx = s1.dot(v*w)
        d2N_dx2 = s2.dot(v*w**2)
        dyt_dx = dyt_dxf_v(A, B, x, N, dN_dx)
        d2yt_dx2 = d2yt_dx2f_v(A, B, x, N, dN_dx, d2N_dx2)
        G = self.Gf_v(x, yt, dyt_dx, d2yt_dx2)
        E = sqrt(np.sum(G**2))
        return E

    def compute_error_gradient(self, p, x, A, B):
        """Compute the gradient of the error function wrt network
        parameters."""

        # Compute the number of training points.
        n = len(x)

        # Compute the number of hidden nodes.
        H = len(self.w)

        # Create the vector to hold the gradient.
        # grad = np.zeros(3*H)

        # Unpack the network parameters.
        w = p[0:H]
        u = p[H:2*H]
        v = p[2*H:3*H]

        # Individual parameter gradients
        # dE_dv = np.zeros(H)
        # dE_du = np.zeros(H)
        # dE_dw = np.zeros(H)

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
        d3N_dwdx2 = v*(2*s2*w + s3*w**2*x)
        d3N_dudx2 = v*s3*w**2
        d3N_dvdx2 = s2*w**2
        yt = ytf_v(A, B, x, N)
        dyt_dx = dyt_dxf_v(A, B, x, N, dN_dx)
        d2yt_dx2 = d2yt_dx2f_v(A, B, x, N, dN_dx, d2N_dx2)
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

    def run(self, x):
        """x is a single input value."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        N = s.dot(self.v)
        yt = ytf(self.bc0, self.bc1, x, N)
        return yt

    def run_derivative(self, x):
        """x is a single input value."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        # d2N_dx2 = s2.dot(self.v*self.w**2)
        dyt_dx = dyt_dxf(self.bc0, self.bc1, x, N, dN_dx)
        return dyt_dx

    def run_2nd_derivative(self, x):
        """x is a single input value."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        d2N_dx2 = s2.dot(self.v*self.w**2)
        d2yt_dx2 = d2yt_dx2f(self.bc0, self.bc1, x, N, dN_dx, d2N_dx2)
        return d2yt_dx2

#--------------------------------------------------------------------------------

if __name__ == '__main__':

    # Create training data.
    x_train = np.linspace(0, 1, 11)

    # Test each training algorithm on each equation.
    for ode in ('ode2_bvp_00',):
        print('Examining %s.' % ode)
        ode2bvp = ODE2BVP(ode)
        print(ode2bvp)

        # (Optional) analytical solution and derivative
        if ode2bvp.yaf:
            print('The analytical solution is:')
            print(np.vectorize(ode2bvp.yaf)(x_train))
        if ode2bvp.dya_dxf:
            print('The analytical derivative is:')
            print(np.vectorize(ode2bvp.dya_dxf)(x_train))
        if ode2bvp.d2ya_dx2f:
            print('The analytical 2nd derivative is:')
            print(np.vectorize(ode2bvp.d2ya_dx2f)(x_train))
        print()

        # Create and train the networks.
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Training using %s algorithm.' % trainalg)
            net = NNODE2BVP(ode2bvp)
            print(net)
            np.random.seed(0)
#            try:
            net.train(x_train, trainalg=trainalg)
#            except (OverflowError, ValueError) as e:
#                print('Error using %s algorithm on %s!' % (trainalg, ode))
#                print(e)
#                print()
#                continue
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

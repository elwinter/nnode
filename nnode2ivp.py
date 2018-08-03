# Notation notes:
# The suffix 'f' usually denotes an object that is a function.

# The suffix '_v' denotes a function that has been vectorized with
# np.vectorize().

from scipy.optimize import minimize
from math import sqrt
import numpy as np

from ode2ivp import ODE2IVP
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from slffnn import SLFFNN

# Default values for method parameters
default_clamp = False
default_debug = False
default_eta = 0.01
default_maxepochs = 1000
default_nhid = 10
default_ntrain = 10
default_ode = 'ode01ivp'
default_randomize = False
default_rmseout = 'rmse.dat'
default_seed = 0
default_trainalg = 'delta'
default_umax = 1
default_umin = -1
default_verbose = False
default_vmax = 1
default_vmin = -1
default_wmax = 1
default_wmin = -1

# Define the trial solution for a 2nd-order ODE IVP.
def ytf(A, Ap, x, N):
    return A + x*Ap + x**2*N

# Define the 1st trial derivative.
def dyt_dxf(A, Ap, x, N, dN_dx):
    return Ap + x**2*dN_dx + 2*x*N

# Define the 2nd trial derivative.
def d2yt_dx2f(A, Ap, x, N, dN_dx, d2N_dx2):
    return x**2*d2N_dx2 + 4*x*dN_dx + 2*N

# Vectorize the trial solution and derivatives.
ytf_v = np.vectorize(ytf)
dyt_dxf_v = np.vectorize(dyt_dxf)
d2yt_dx2f_v = np.vectorize(d2yt_dx2f)

# Vectorize sigma functions.
sigma_v = np.vectorize(sigma)
dsigma_dz_v = np.vectorize(dsigma_dz)
d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
d3sigma_dz3_v = np.vectorize(d3sigma_dz3)

class NNODE2IVP(SLFFNN):

    def __init__(self, ode2ivp, nhid=default_nhid):

        # ODE, in the form G(x,y,dy/dx,d2y/dx2)=0.
        self.Gf = ode2ivp.Gf
        self.Gf_v = np.vectorize(self.Gf)

        # dG/dy
        self.dG_dyf = ode2ivp.dG_dyf
        self.dG_dyf_v = np.vectorize(self.dG_dyf)

        # dG/d(dy/dx)
        self.dG_dydxf = ode2ivp.dG_dydxf
        self.dG_dydxf_v = np.vectorize(self.dG_dydxf)

        # dG/d(d2y/dx2)
        self.dG_d2ydx2f = ode2ivp.dG_d2ydx2f
        self.dG_d2ydx2f_v = np.vectorize(self.dG_d2ydx2f)

        # Initial conditions
        self.ic = ode2ivp.ic
        self.ic1 = ode2ivp.ic1

        # Analytical solution (optional)
        if ode2ivp.yaf:
            self.yaf = ode2ivp.yaf
            self.yaf_v = np.vectorize(self.yaf)

        # Analytical derivative (optional)
        if ode2ivp.dya_dxf:
            self.dya_dxf = ode2ivp.dya_dxf
            self.dya_dxf_v = np.vectorize(self.dya_dxf)

        # Analytical 2nd derivative (optional)
        if ode2ivp.d2ya_dx2f:
            self.d2ya_dx2f = ode2ivp.d2ya_dx2f
            self.d2ya_dx2f_v = np.vectorize(self.d2ya_dx2f)

        # Create arrays to hold the weights and biases, initially all 0.
        self.w = np.zeros(nhid)
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

    def __str__(self):
        s = "NNODE2IVP:\n"
        s += "w = %s" % self.w + "\n"
        s += "u = %s" % self.u + "\n"
        s += "v = %s" % self.v + "\n"
        return s

    def train(self,
              x,                             # x-values for training points
              trainalg = default_trainalg,   # Training algorithm
              nhid = default_nhid,           # Node count in hidden layer
              maxepochs = default_maxepochs, # Max training epochs
              eta = default_eta,             # Learning rate
              clamp = default_clamp,         # Turn on parameter clamping
              randomize = default_randomize, # Randomize training sample order
              wmin = default_wmin,           # Minimum hidden weight value
              wmax = default_wmax,           # Maximum hidden weight value
              umin = default_umin,           # Minimum hidden bias value
              umax = default_umax,           # Maximum hidden bias value
              vmin = default_vmin,           # Minimum output weight value
              vmax = default_vmax,           # Maximum output weight value 
              rmseout = default_rmseout,     # Output file for ODE RMS error
              debug = default_debug,
              verbose = default_verbose
    ):
        print('trainalg =', trainalg)
        if trainalg == 'delta':
            print('Calling self.train_delta().')
            self.train_delta(x, nhid = nhid, maxepochs = maxepochs, eta = eta,
                             clamp = clamp, randomize = randomize,
                             wmin = wmin, wmax = wmax,
                             umin = umin, umax = umax,
                             vmin = vmin, vmax = vmax,
                             rmseout = rmseout, debug = debug, verbose = verbose)
        elif trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS',
                          'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Calling self.train_minimize().')
            self.train_minimize(x, trainalg=trainalg,
                                wmin = wmin, wmax = wmax,
                                umin = umin, umax = umax,
                                vmin = vmin, vmax = vmax,
                                debug = debug, verbose = verbose)
        else:
            print('ERROR: Invalid training algorithm (%s)!' % trainalg)
            exit(0)

    def train_delta(self,
              x,                             # x-values for training points
              nhid = default_nhid,           # Node count in hidden layer
              maxepochs = default_maxepochs, # Max training epochs
              eta = default_eta,             # Learning rate
              clamp = default_clamp,         # Turn on parameter clamping
              randomize = default_randomize, # Randomize training sample order
              wmin = default_wmin,           # Minimum hidden weight value
              wmax = default_wmax,           # Maximum hidden weight value
              umin = default_umin,           # Minimum hidden bias value
              umax = default_umax,           # Maximum hidden bias value
              vmin = default_vmin,           # Minimum output weight value
              vmax = default_vmax,           # Maximum output weight value 
              rmseout = default_rmseout,     # Output file for ODE RMS error
              debug = default_debug,
              verbose = default_verbose
    ):
        """Train the network to solve a 2nd-order ODE IVP. """
        if debug: print('x =', x)
        if debug: print('nhid =', nhid)
        if debug: print('maxepochs =', maxepochs)
        if debug: print('eta =', eta)
        if debug: print('clamp =', clamp)
        if debug: print('randomize =', randomize)
        if debug: print('wmin =', wmin)
        if debug: print('wmax =', wmax)
        if debug: print('umin =', umin)
        if debug: print('umax =', umax)
        if debug: print('vmin =', vmin)
        if debug: print('vmax =', vmax)
        if debug: print('rmseout =', rmseout)
        if debug: print('debug =', debug)
        if debug: print('verbose =', verbose)

        # Copy equation characteristics for local use.
        ic = self.ic
        ic1 = self.ic1
        dG_dyf = self.dG_dyf
        dG_dydxf = self.dG_dydxf
        dG_d2ydx2f = self.dG_d2ydx2f
        if debug: print('ic =', ic)
        if debug: print('ic1 =', ic1)
        if debug: print('dG_dyf =', dG_dyf)
        if debug: print('dG_dydxf =', dG_dydxf)
        if debug: print('dG_d2ydx2f =', dG_d2ydx2f)

        # Sanity-check arguments.
        assert len(x) > 0
        assert nhid > 0
        assert maxepochs > 0
        assert eta > 0
        assert wmin < wmax
        assert umin < umax
        assert vmin < vmax
        assert rmseout

        #------------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)
        if debug: print('n =', n)

        # Change notation for convenience.
        A = ic
        if debug: print('A =', A)
        Ap = ic1
        if debug: print('Ap =', Ap)
        H = nhid
        if debug: print('H =', H)

        #------------------------------------------------------------------------

        # Create the network.

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        self.w = np.random.uniform(wmin, wmax, H)
        if debug: print('w =', self.w)

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        self.u = np.random.uniform(umin, umax, H)
        if debug: print('u =', self.u)

        # Create an array to hold the weights connecting the hidden
        # nodes to the output node. The weights are initialized with a
        # uniform random distribution.
        self.v = np.random.uniform(vmin, vmax, H)
        if debug: print('v =', self.v)

        # Create arrays to hold RMSE and parameter history.
        rmse_history = np.zeros(maxepochs)
        w_history = np.zeros((maxepochs, H))
        u_history = np.zeros((maxepochs, H))
        v_history = np.zeros((maxepochs, H))

        #------------------------------------------------------------------------

        # Initial parameter deltas are 0.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros(H)

        # Train the network.
        for epoch in range(maxepochs):
            if debug: print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            self.w -= eta*dE_dw
            self.u -= eta*dE_du
            self.v -= eta*dE_dv
            if debug: print('w =', self.w)
            if debug: print('u =', self.u)
            if debug: print('v =', self.v)

            # If desired, clamp the parameter values at the limits.
            if clamp:
                self.w[self.w < wmin] = wmin
                self.w[self.w > wmax] = wmax
                self.u[self.u < umin] = umin
                self.u[self.u > umax] = umax
                self.v[self.v < vmin] = vmin
                self.v[self.v > vmax] = vmax

            # Save the current parameter values in the history.
            if debug: print('Saving current parameter values.')
            w_history[epoch] = self.w
            u_history[epoch] = self.u
            v_history[epoch] = self.v

            # If the randomize flag is set, shuffle the order of the
            # training points.
            if randomize:
                if debug: print('Randomizing training sample order.')
                np.random.shuffle(x)

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
            if debug: print('z =', z)

            # Each z[i,k] gets mapped to a s[i,k], so s, s1, s2, s3
            # are all n x H ndarray objects.
            s = sigma_v(z)
            if debug: print('s =', s)
            s1 = dsigma_dz_v(z)
            if debug: print('s1 =', s1)
            s2 = d2sigma_dz2_v(z)
            if debug: print('s2 =', s2)
            s3 = d3sigma_dz3_v(z)
            if debug: print('s3 =', s3)

            #--------------------------------------------------------------------

            # Compute the network output and its derivatives, for each
            # training point.

            # The network output N[i] for each training point x[i] is
            # the sum of the outputs s[i,k], weighted by the output
            # weights v[k]. N[] is thus a 1-D ndarray object of length
            # n.
            N = s.dot(self.v)
            if debug: print('N =', N)

            # Since there is only one input x[i] for each training
            # sample, and a single output value N[i], the derivative
            # dN_dx[] is also a 1-D ndarray object of length n. For
            # each sample i, the dN_dx[i] is the dot product of s1[]
            # with the product of the output weights v[] and the
            # hidden weights w[].
            dN_dx = s1.dot(self.v*self.w)
            if debug: print('dN_dx =', dN_dx)

            # s1[] is n x H, x[] is 1 x n, and v[] is 1 x H, so the
            # outer product of x[] and v[] is needed to get the
            # product of each x[i] with each v[k], resulting in a n x
            # H array, which is then multiplied by s1[], which is also
            # an n x H array.
            dN_dw = s1*np.outer(x, self.v)
            if debug: print('dN_dw =', dN_dw)

            # s1[] is n x H, and v[] is 1 x H, so dN_du[] is n x H.
            dN_du = s1*self.v
            if debug: print('dN_du =', dN_du)

            # There are n network outputs N[i], and H hidden nodes, so
            # the partials of N[] wrt the hidden node weights at the
            # output is a n x H ndarray object.
            dN_dv = s
            if debug: print('dN_dv =', dN_dv)

            # v[] is 1 x H, s1[] and s2[] are n x H, x[] is 1 x N, and
            # w[] is 1 x H, so the outer product of x[] and wp[ gives
            # a n x H array, which is multipled by another n x H array
            # (s2[]) and then added to another n x H array
            # (s1[]). Each row of this array is then multiplied by the
            # row vector v[].
            d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))
            if debug: print('d2N_dwdx =', d2N_dwdx)

            # v[] is 1 x H, s2[] is n x H, and w is[] 1 x H, so each
            # row of s1[] gets multiplied by the single row v[], which
            # then multiplies the single row w[], resulting in a n x H
            # array for the 2nd partial d2N_dudx[].
            d2N_dudx = self.v*s2*self.w
            if debug: print('d2N_dudx =', d2N_dudx)

            # s1[] is n x H, and w[] is 1 x H, so each row of s1[]
            # gets multiplied by the single row v[], resulting in a n
            # x H array for the 2nd partial d2N_dvdx[].
            d2N_dvdx = s1*self.w
            if debug: print('d2N_dvdx =', d2N_dvdx)

            # Since there is only one input x[i] for each training
            # sample, and a single output value N[i], the 2nd
            # derivative d2N_dx2[] is also a 1-D ndarray object of
            # length n. For each sample i, the d2N_dx2[i] is the dot
            # product of s2[] with the product of the output weights
            # v[] and the hidden weights w[] squared.
            d2N_dx2 = s2.dot(self.v*self.w**2)
            if debug: print('d2N_dx2 =', d2N_dx2)

            d3N_dwdx2 = self.v*(2*s2*self.w + s3*self.w**2*x)
            if debug: print('d3N_dwdx2 =', d3N_dwdx2)
            d3N_dudx2 = self.v*s3*self.w**2
            if debug: print('d3N_dudx2 =', d3N_dudx2)
            d3N_dvdx2 = s2*self.w**2
            if debug: print('d3N_dvdx2 =', d3N_dvdx2)

            #--------------------------------------------------------------------

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            yt = ytf_v(A, Ap, x, N)
            if debug: print('yt =', yt)
            dyt_dx = dyt_dxf_v(A, Ap, x, N, dN_dx)
            if debug: print('dyt_dx =', dyt_dx)
            d2yt_dx2 = d2yt_dx2f_v(A, Ap, x, N, dN_dx, d2N_dx2)
            if debug: print('d2yt_dx2 =', d2yt_dx2)
            dyt_dw = np.broadcast_to(x**2, (H, n)).T*dN_dw
            if debug: print('dyt_dw =', dyt_dw)
            dyt_du = np.broadcast_to(x**2, (H, n)).T*dN_du
            if debug: print('dyt_du =', dyt_du)
            dyt_dv = np.broadcast_to(x**2, (H, n)).T*dN_dv
            if debug: print('dyt_dv =', dyt_dv)
            d2yt_dwdx = np.broadcast_to(x**2, (H, n)).T*d2N_dwdx \
                        + 2*np.broadcast_to(x, (H, n)).T*dN_dw
            if debug: print('d2yt_dwdx =', d2yt_dwdx)
            d2yt_dudx = np.broadcast_to(x**2, (H, n)).T*d2N_dudx \
                        + 2*np.broadcast_to(x, (H, n)).T*dN_du
            if debug: print('d2yt_dudx =', d2yt_dudx)
            d2yt_dvdx = np.broadcast_to(x**2, (H, n)).T*d2N_dvdx \
                        + 2*np.broadcast_to(x, (H, n)).T*dN_dv
            if debug: print('d2yt_dvdx =', d2yt_dvdx)
            d3yt_dwdx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dwdx2 \
                         + 4*np.broadcast_to(x, (H, n)).T*d2N_dwdx \
                         + 2*dN_dw
            if debug: print('d3yt_dwdx2 =', d3yt_dwdx2)
            d3yt_dudx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dudx2 \
                         + 4*np.broadcast_to(x, (H, n)).T*d2N_dudx \
                         + 2*dN_du
            if debug: print('d3yt_dudx2 =', d3yt_dudx2)
            d3yt_dvdx2 = np.broadcast_to(x**2, (H, n)).T*d3N_dvdx2 \
                         + 4*np.broadcast_to(x, (H, n)).T*d2N_dvdx \
                         + 2*dN_dv
            if debug: print('d3yt_dvdx2 =', d3yt_dvdx2)

            #--------------------------------------------------------------------

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = self.Gf_v(x, yt, dyt_dx, d2yt_dx2)
            if debug: print('G =', G)
            dG_dyt = self.dG_dyf_v(x, yt, dyt_dx, d2yt_dx2)
            if debug: print('dG_dyt =', dG_dyt)
            dG_dytdx = self.dG_dydxf_v(x, yt, dyt_dx, d2yt_dx2)
            if debug: print('dG_dytdx =', dG_dytdx)
            dG_d2ytdx2 = self.dG_d2ydx2f_v(x, yt, dyt_dx, d2yt_dx2)
            if debug: print('dG_d2ytdx2 =', dG_d2ytdx2)
            dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw \
                    + np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx \
                    + np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dwdx2
            if debug: print('dG_dw =', dG_dw)
            dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du \
                    + np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx \
                    + np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dudx2
            if debug: print('dG_du =', dG_du)
            dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv \
                    + np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx \
                    + np.broadcast_to(dG_d2ytdx2, (H, n)).T*d3yt_dvdx2
            if debug: print('dG_dv =', dG_dv)

            # Compute the error function for this epoch.
            E = sum(G**2)
            if debug: print('E =', E)

            # Compute the partial derivatives of the error with respect to
            # the network parameters.
            dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis = 0)
            if debug: print('dE_dw =', dE_dw)
            dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis = 0)
            if debug: print('dE_du =', dE_du)
            dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis = 0)
            if debug: print('dE_dv =', dE_dv)

            #--------------------------------------------------------------------

            # Record the current RMSE.
            rmse = sqrt(E/n)
            rmse_history[epoch] = rmse
            if verbose: print(epoch, rmse)

            # Save the error and parameter history.
            np.savetxt(rmseout, rmse_history)
            np.savetxt('w.dat', w_history)
            np.savetxt('v.dat', v_history)
            np.savetxt('u.dat', u_history)

    def train_minimize(self,
                       x,                          # x-values for training points
                       trainalg=default_trainalg,   # Training algorithm
                       wmin = default_wmin,         # Minimum hidden weight value
                       wmax = default_wmax,         # Maximum hidden weight value
                       umin = default_umin,         # Minimum hidden bias value
                       umax = default_umax,         # Maximum hidden bias value
                       vmin = default_vmin,         # Minimum output weight value
                       vmax = default_vmax,         # Maximum output weight value 
                       debug = default_debug,
                       verbose = default_verbose
    ):
        """Train the network to solve a 2nd-order ODE IVP. """
        if debug: print('x =', x)
        if debug: print('trainalg =', trainalg)
        if debug: print('wmin =', wmin)
        if debug: print('wmax =', wmax)
        if debug: print('umin =', umin)
        if debug: print('umax =', umax)
        if debug: print('vmin =', vmin)
        if debug: print('vmax =', vmax)
        if debug: print('debug =', debug)
        if debug: print('verbose =', verbose)

        #-----------------------------------------------------------------------

        # Sanity-check arguments.
        assert len(x) > 0
        assert trainalg
        assert vmin < vmax
        assert wmin < wmax
        assert umin < umax

        #------------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)
        if debug: print('n =', n)

        # Change notation for convenience.
        H = len(self.w)
        if debug: print('H =', H)

        #------------------------------------------------------------------------

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        self.w = np.random.uniform(wmin, wmax, H)
        if debug: print('w =', self.w)

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        self.u = np.random.uniform(umin, umax, H)
        if debug: print('u =', self.u)

        # Create an array to hold the weights connecting the hidden
        # nodes to the output node. The weights are initialized with a
        # uniform random distribution.
        self.v = np.random.uniform(vmin, vmax, H)
        if debug: print('v =', self.v)

        #------------------------------------------------------------------------

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        # p = [w, u, v]
        p = np.hstack((self.w, self.u, self.v))
        if debug: print('p =', p)

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            res = minimize(self.computeError, p, method=trainalg,
                           args = (x, self.ic, self.ic1))
        elif trainalg in ('Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            res = minimize(self.computeError, p, method=trainalg,
                           jac=self.computeErrorGradient, args = (x, self.ic,
                                                                  self.ic1))
        if debug: print(res)

        # Unpack the optimized network parameters.
        self.w = res.x[0:H]
        self.u = res.x[H:2*H]
        self.v = res.x[2*H:3*H]
        if debug: print('Final w =', self.w)
        if debug: print('Final u =', self.u)
        if debug: print('Final v =', self.v)

    def computeError(self, p, x, A, Ap):
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
        yt = ytf_v(A, Ap, x, N)
        dN_dx = s1.dot(v*w)
        d2N_dx2 = s2.dot(v*w**2)
        dyt_dx = dyt_dxf_v(A, Ap, x, N, dN_dx)
        d2yt_dx2 = d2yt_dx2f_v(A, Ap, x, N, dN_dx, d2N_dx2)
        G = self.Gf_v(x, yt, dyt_dx, d2yt_dx2)
        E = sqrt(np.sum(G**2))
        return E

    def computeErrorGradient(self, p, x, A, Ap):
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
        yt = ytf_v(A, Ap, x, N)
        dyt_dx = dyt_dxf_v(A, Ap, x, N, dN_dx)
        d2yt_dx2 = d2yt_dx2f_v(A, Ap, x, N, dN_dx, d2N_dx2)
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
        E = np.sum(G**2)
        dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis = 0)
        dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis = 0)
        dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis = 0)

        jac = np.hstack((dE_dw, dE_du, dE_dv))
        return jac

    def run(self, x):
        """x is a single input value."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        N = s.dot(self.v)
        yt = ytf(self.ic, self.ic1, x, N)
        return yt

    def run_derivative(self, x):
        """x is a single input value."""
        z = np.outer(x, self.w) + self.u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        # d2N_dx2 = s2.dot(self.v*self.w**2)
        dyt_dx = dyt_dxf(self.ic, self.ic1, x, N, dN_dx)
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
        d2yt_dx2 = d2yt_dx2f(self.ic, self.ic1, x, N, dN_dx, d2N_dx2)
        return d2yt_dx2

#--------------------------------------------------------------------------------

if __name__ == '__main__':

    # Create training data.
    x_train = np.linspace(0, 1, 10)
    print('x_train =', x_train)

    # ode2ivp = ODE2IVP('ode01ivp')
    # nnode2ivp = NNODE2IVP(ode2ivp)
    # np.random.seed(0)
    # nnode2ivp.train(x_train, trainalg='delta', verbose=True)

    # Test each training algorithm on each equation.
    for ode in ('ode01ivp',):
    # for ode in ('ode01ivp', 'lagaris03ivp'):
        print('Examining %s.' % ode)
        ode2ivp = ODE2IVP(ode)
        net = NNODE2IVP(ode2ivp)
        ya = net.yaf_v(x_train)
        dya_dx = net.dya_dxf_v(x_train)
        d2ya_dx2 = net.d2ya_dx2f_v(x_train)
        print('The analytical solution is:')
        print('ya =', ya)
        print('The analytical derivative is:')
        print('dya_dx =', dya_dx)
        print('The analytical 2nd derivative is:')
        print('d2ya_dx2 =', d2ya_dx2)
        print()
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Training using %s algorithm.' % trainalg)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg)
            except Exception as e:
                print('Error using %s algorithm on %s!' % (trainalg, ode))
                print(e)
                print()
                continue
            yt = net.run(x_train)
            dyt_dx = net.run_derivative(x_train)
            d2yt_dx2 = net.run_2nd_derivative(x_train)
            # print('The optimized network parameters are:')
            # print('w =', net.w)
            # print('u =', net.u)
            # print('v =', net.v)
            print('The trained solution is:')
            print('yt =', yt)
            print('The trained derivative is:')
            print('dyt_dx =', dyt_dx)
            print('The trained 2nd derivative is:')
            print('d2yt_dx2 =', d2yt_dx2)
            print()

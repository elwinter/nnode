from math import sqrt
import numpy as np

from ode1ivp import ODE1IVP
from sigma import sigma, dsigma_dz, d2sigma_dz2
from slffnn import SLFFNN

# Default values for program parameters
default_clamp = False
default_debug = False
default_eta = 0.01
default_maxepochs = 1000
default_nhid = 10
default_ntest = 10
default_ntrain = 10
default_ode = 'ode00'
default_randomize = False
default_rmseout = 'rmse.dat'
default_seed = 0
default_testout = 'testpoints.dat'
default_trainout = 'trainpoints.dat'
default_umax = 1
default_umin = -1
default_verbose = False
default_vmax = 1
default_vmin = -1
default_wmax = 1
default_wmin = -1

# Define the trial solution for a 1st-order ODE IVP.
def ytf(A, x, N):
    return A + x*N

# Define the 1st trial derivative.
def dyt_dxf(x, N, dN_dx):
    return x*dN_dx + N

class NNODE1IVP(SLFFNN):

    def __init__(self, ode1ivp):
        self.ode1ivp = ode1ivp

    def __str__(self):
        # Add a name to distinguish multiple nets.
        s = "NNODE1IVP:\n"
        s += "w = %s" % self.w + "\n"
        s += "u = %s" % self.u + "\n"
        s += "v = %s" % self.v + "\n"
        return s

    def train(self,
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
        """Train the network to solve a 1st-order ODE IVP. """
        if debug: print('x =', x)
        if debug: print('maxepochs =', maxepochs)
        if debug: print('eta =', eta)
        if debug: print('clamp =', clamp)
        if debug: print('randomize =', randomize)
        if debug: print('rmseout =', rmseout)
        if debug: print('wmin =', wmin)
        if debug: print('wmax =', wmax)
        if debug: print('umin =', umin)
        if debug: print('umax =', umax)
        if debug: print('vmin =', vmin)
        if debug: print('vmax =', vmax)
        if debug: print('debug =', debug)
        if debug: print('verbose =', verbose)

        #-----------------------------------------------------------------------

        # Copy equation characteristics for local use.
        # assert ode1ivp
        Gf = self.ode1ivp.Gf
        ic = self.ode1ivp.ic
        dG_dyf = self.ode1ivp.dG_dyf
        dG_dydxf = self.ode1ivp.dG_dydxf
        if debug: print('Gf =', Gf)
        if debug: print('ic =', ic)
        if debug: print('dG_dyf =', dG_dyf)
        if debug: print('dG_dydxf =', dG_dydxf)

        # Sanity-check arguments.
        assert Gf
        assert ic != None
        assert dG_dyf
        assert dG_dydxf
        assert len(x) > 0
        assert maxepochs > 0
        assert eta > 0
        assert vmin < vmax
        assert wmin < wmax
        assert umin < umax
        assert rmseout

        #------------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)
        if debug: print('n =', n)

        # Change notation for convenience.
        A = ic
        if debug: print('A =', A)
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

        # Vectorize the functions used by the network.
        sigma_v = np.vectorize(sigma)
        dsigma_dz_v = np.vectorize(dsigma_dz)
        d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
        ytf_v = np.vectorize(ytf)
        dyt_dxf_v = np.vectorize(dyt_dxf)
        Gf_v = np.vectorize(Gf)
        dG_dyf_v = np.vectorize(dG_dyf)
        dG_dydxf_v = np.vectorize(dG_dydxf)

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
            # derivatives, for each hidden node k, for each training point
            # i. Each hidden node has 1 weight w[k] (since there is only 1
            # input x[i] per training sample) and 1 bias u[k].

            # The weighted input to each hidden node is just z=w*x+u.
            # Since x and w are 1-D ndarray objects, np.outer() is used so
            # that each x is multiplied by each w, resulting in a 2-D array
            # with n rows and H columns. The biases u[] are then added to
            # each row of the resulting 2-D array.
            z = np.outer(x, self.w) + self.u
            if debug: print('z =', z)

            # Each z[i,k] gets mapped to a s[i,k], so s, s1, s2 are all n
            # x H ndarray objects.
            s = sigma_v(z)
            if debug: print('s =', s)
            s1 = dsigma_dz_v(z)
            if debug: print('s1 =', s1)
            s2 = d2sigma_dz2_v(z)
            if debug: print('s2 =', s2)

            #--------------------------------------------------------------------

            # Compute the network output and its derivatives, for each
            # training point.

            # The network output N[i] for each training point x[i] is the
            # sum of the outputs s[i,k], weighted by the output weights
            # v[k]. N is thus a 1-D ndarray object of length n.
            N = s.dot(self.v)
            if debug: print('N =', N)

            # Since there is only one input x[i] for each training
            # sample, and a single output value N[i], the derivative
            # dN_dx[] is also a 1-D ndarray object of length n. For
            # each sample i, the dN_dx[i] is the dot product of s1
            # with the product of the output weights v[] and the
            # hidden weights w[].
            dN_dx = s1.dot(self.v*self.w)
            if debug: print('dN_dx =', dN_dx)

            # s1[] is n x H, x is 1 x n, and v is 1 x H, so the outer
            # product of x and v is needed to get the product of each x[i]
            # with each v[k], resulting in a n x H array, which is then
            # multiplied by s1[], which is also an n x H array.
            dN_dw = s1*np.outer(x,self.v)
            if debug: print('dN_dw =', dN_dw)

            # s1[] is n x H, and v is 1 x H, so dN_du[] is n x H.
            dN_du = s1*self.v
            if debug: print('dN_du =', dN_du)

            # There are n network outputs N[i], and H hidden nodes, so the
            # partials of N wrt the hidden node weights at the output is a
            # n x H ndarray object.
            dN_dv = s
            if debug: print('dN_dv =', dN_dv)

            # v is 1 x H, s1[] and s2[] are n x H, x is 1 x N, and w is 1
            # x H, so the outer product of x and w gives a n x H array,
            # which is multipled by another n x H array (s2) and then
            # added to another n x H array (s1). Each row of this array is
            # then multiplied by the row vector v.
            d2N_dwdx = self.v*(s1 + s2*np.outer(x, self.w))
            if debug: print('d2N_dwdx =', d2N_dwdx)

            # v is 1 x H, s2[] is n x H, and w is 1 x H, so each row of
            # s1[] gets multiplied by the single row v[], which then
            # multiplies the single row w[], resulting in a n x H array
            # for the 2nd partial d2N_dudx[].
            d2N_dudx = self.v*s2*self.w
            if debug: print('d2N_dudx =', d2N_dudx)

            # s1[] is n x H, and w is 1 x H, so each row of s1[] gets
            # multiplied by the single row v[], resulting in a n x H
            # array for the 2nd partial d2N_dvdx[].
            d2N_dvdx = s1*self.w
            if debug: print('d2N_dvdx =', d2N_dvdx)

            #--------------------------------------------------------------------

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            yt = ytf_v(A, x, N)
            if debug: print('yt =', yt)
            dyt_dx = dyt_dxf_v(x, N, dN_dx)
            if debug: print('dyt_dx =', dyt_dx)
            dyt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
            if debug: print('dyt_dw =', dyt_dw)
            dyt_du = np.broadcast_to(x, (H, n)).T*dN_du
            if debug: print('dyt_du =', dyt_du)
            dyt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
            if debug: print('dyt_dv =', dyt_dv)
            d2yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
            if debug: print('d2yt_dwdx =', d2yt_dwdx)
            d2yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
            if debug: print('d2yt_dudx =', d2yt_dudx)
            d2yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv
            if debug: print('d2yt_dvdx =', d2yt_dvdx)

            #--------------------------------------------------------------------

            # Compute the value of the original differential equation for
            # each training point, and its derivatives.
            G = Gf_v(x, yt, dyt_dx)
            if debug: print('G =', G)
            dG_dyt = dG_dyf_v(x, yt, dyt_dx)
            if debug: print('dG_dyt =', dG_dyt)
            dG_dytdx = dG_dydxf_v(x, yt, dyt_dx)
            if debug: print('dG_dytdx =', dG_dytdx)
            dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
                    np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx
            if debug: print('dG_dw =', dG_dw)
            dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
                    np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx
            if debug: print('dG_du =', dG_du)
            dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
                    np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx
            if debug: print('dG_dv =', dG_dv)

            # Compute the error function for this epoch.
            E = np.sum(G**2)
            if debug: print('E =', E)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
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
        np.savetxt(rmseout, rmse_history, fmt = '%.6E', header = 'rmse')
        np.savetxt('w.dat', w_history, fmt = '%.6E', header = 'w')
        np.savetxt('u.dat', u_history, fmt = '%.6E', header = 'u')
        np.savetxt('v.dat', v_history, fmt = '%.6E', header = 'v')

    def run(self, x):
        """x is a single input value."""
        z = x*self.w + self.u
        s = np.vectorize(sigma)(z)
        N = np.dot(self.v, s)
        yt = ytf(self.ode1ivp.ic, x, N)
        return yt

    def run_derivative(self, x):
        """x is a single input value."""
        z = x*self.w + self.u
        s = np.vectorize(sigma)(z)
        s1 = np.vectorize(dsigma_dz)(z)
        N = s.dot(self.v)
        dN_dx = s1.dot(self.v*self.w)
        dyt_dx = dyt_dxf(x, N, dN_dx)
        return dyt_dx

#--------------------------------------------------------------------------------

if __name__ == '__main__':

    # Import the ODE module.
    ode = 'ode00'
    ode1ivp = ODE1IVP(ode)
    print(ode1ivp)

    # Create training data.
    x_train = np.linspace(0, 1, 10)
    print(x_train)

    # Train the network using all defaults.
    np.random.seed(0)
    net = NNODE1IVP(ode1ivp)
    net.train(x_train, verbose = True)

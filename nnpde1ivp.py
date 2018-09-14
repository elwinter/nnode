"""
NNPDE1IVP - Class to solve 2-D 1st-order partial differential equation initial
value problems using a neural network

This module provides the functionality to solve 2-D 1sr-order partial
differential equation initial value problems using a neural network.

Example:
    Create an empty NNPDE1IVP object.
        net = NNPDE1IVP()
    Create an NNPDE1IVP object for a PDE1IVP object.
        net = NNPDE1IVP(pde1ivp_obj)

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

from pde1ivp import PDE1IVP
from sigma import sigma, dsigma_dz, d2sigma_dz2
from slffnn import SLFFNN

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


class NNPDE1IVP(SLFFNN):
    """Solve a 2-D 1st-order PDE IVP with a neural network."""

    # Public methods

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()
        self.eq = eq
        self.w = np.zeros(nhid)
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)
        # <HACK>
        self.delAf = (self.dA_dxf, self.dA_dyf)
        self.delPf = (self.dP_dxf, self.dP_dyf)
        # </HACK>

    def __str__(self):
        s = ''
        s += "NNPDE1IVP:\n"
        s += "%s\n" % self.eq
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

    def train(self, x, trainalg=DEFAULT_TRAINALG, opts=DEFAULT_OPTS):
        """Train the network."""
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

    # Internal methods below this point

    def Af(self, xy):
        (x, y) = xy
        (f0f, g0f) = self.eq.bcf
        A = (1 - x)*f0f(y) + (1 - y)*(g0f(x) - (1 - x)*g0f(0))
        return A

    def dA_dxf(self, xy):
        (x, y) = xy
        (f0f, g0f) = self.eq.bcf
        (df0_dyf, dg0_dxf) = self.eq.bcdf
        dA_dx = -f0f(y) + (1 - y)*(dg0_dxf(x) + g0f(0))
        return dA_dx

    def dA_dyf(self, xy):
        (x, y) = xy
        (f0f, g0f) = self.eq.bcf
        (df0_dyf, dg0_dxf) = self.eq.bcdf
        dA_dy = (1 - x)*df0_dyf(y) - g0f(x) + (1 - x)*g0f(0)
        return dA_dy

    def Pf(self, xy):
        (x, y) = xy
        P = x*y
        return P

    def dP_dxf(self, xy):
        (x, y) = xy
        dP_dx = y
        return dP_dx

    def dP_dyf(self, xy):
        (x, y) = xy
        dP_dy = x
        return dP_dy

    def Ytf(self, xy, N):
        A = self.Af(xy)
        P = self.Pf(xy)
        Yt = A + P*N
        return Yt

    def __train_delta(self, x, opts=DEFAULT_OPTS):
        """Train the network with the delta method."""

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
        m = len(self.eq.bcf)
        H = opts['nhid']

        # Create the hidden node weights, biases, and output node weights.
        self.w = np.random.uniform(opts['wmin'], opts['wmax'], (2, H))
        self.u = np.random.uniform(opts['umin'], opts['umax'], H)
        self.v = np.random.uniform(opts['vmin'], opts['vmax'], H)

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((2, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Train the network.
#        for epoch in range(opts['maxepochs']):
        for epoch in range(1):
            if opts['debug']:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            self.w -= opts['eta']*dE_dw
            self.u -= opts['eta']*dE_du
            self.v -= opts['eta']*dE_dv

            # Compute the input, the sigmoid function, and its
            # derivatives, for each training point.
            z = np.broadcast_to(self.u, (n, H)) + np.dot(x, self.w)
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)
            print(s2)

            # These arrays are broadcast ('_b' suffix) to the same shape
            # (all are shape (n,m,H)) to facilitate calculations done
            # later.
            x_b = np.broadcast_to(x.T, (H, m, n)).T
            w_b = np.broadcast_to(self.w, (n, m, H))
            v_b = np.broadcast_to(np.broadcast_to(self.v, (m, H)), (n, m, H))
            s1_b = np.broadcast_to(np.expand_dims(s1, axis=1), (n, m, H))
            s2_b = np.broadcast_to(np.expand_dims(s2, axis=1), (n, m, H))

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(self.v)
            dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)
            dN_dw = v_b*s1_b*x_b
            dN_du = s1*self.v
            dN_dv = s
            d2N_dwdx = v_b*s1_b + v_b*s2_b*w_b*x_b
            d2N_dudx = v_b*s2_b*w_b
            d2N_dvdx = s1_b*w_b

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            dA_dx = np.zeros((n, m))
            P = np.zeros(n)
            dP_dx = np.zeros((n, m))
            Yt = np.zeros(n)
#            d2Yt_dudx = np.zeros((n, m, H))
#            d2Yt_dvdx = np.zeros((n, m, H))
            for i in range(n):
                P[i] = self.Pf(x[i])
                Yt[i] = self.Ytf(x[i], N[i])
                for j in range(m):
                    dA_dx[i, j] = self.delAf[j](x[i])
                    dP_dx[i, j] = self.delPf[j](x[i])
#                for k in range(H):
#                    for j in range(m):
#                        d2Yt_dvdx[i,j,k] = (
#                            P[i]*d2N_dvdx[i,j,k] + dP_dx[i,j]*dN_dv[i,k]
#                        )
#                        d2Yt_dudx[i,j,k] = (
#                            P[i]*d2N_dudx[i,j,k] + dP_dx[i,j]*dN_du[i,k]
#                        )
#                        P_b = np.tile(P, (2, 1)).T
#                        N_b = np.tile(N, (2, 1)).T
#                        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
#                        dYt_dw = dN_dw*np.broadcast_to(P.T, (H, m, n)).T
#                        dYt_du = dN_du*np.broadcast_to(P.T, (H, n)).T
#                        dYt_dv = dN_dv*np.broadcast_to(P.T, (H, n)).T
#                        d2Yt_dwdx = np.broadcast_to(P.T, (H, m, n)).T*d2N_dwdx +\
#                                    np.broadcast_to(dP_dx.T, (H, m, n)).T*dN_dw
            print(Yt)

#            # Compute the value of the original differential equation
#            # for each training point, and its derivatives.
#            G = np.zeros(n)
#            dG_dYt = np.zeros(n)
#            dG_ddelYt = np.zeros((n, m))
#            dG_dv = np.zeros((n, H))
#            dG_du = np.zeros((n, H))
#            dG_dw = np.zeros((n, m, H))
#            for i in range(n):
#                G[i] = self.Gf(x[i], Yt[i], dYt_dx[i])
#                dG_dYt[i] = self.dG_dYf(x[i], Yt[i], dYt_dx[i])
#                for j in range(m):
#                    dG_ddelYt[i,j] = self.dG_ddelYf[j](x[i], Yt[i], dYt_dx[i])
#                for k in range(H):
#                    dG_dv[i,k] = dG_dYt[i]*dYt_dv[i,k]
#                    dG_du[i,k] = dG_dYt[i]*dYt_du[i,k]
#                    for j in range(m):
#                        dG_dv[i,k] += dG_ddelYt[i,j]*d2Yt_dvdx[i,j,k]
#                        dG_du[i,k] += dG_ddelYt[i,j]*d2Yt_dudx[i,j,k]
#                        dG_dw[i,j,k] = (
#                            dG_dYt[i]*dYt_dw[i,j,k] +
#                            dG_ddelYt[i,j]*d2Yt_dwdx[i,j,k]
#                        )
#
#            # Compute the error function for this epoch.
#            E = np.sum(G**2)
#            if debug: print('E =', E)
#
#            # Compute the partial derivatives of the error with
#            # respect to the network parameters.
#            dE_dv = np.zeros(H)
#            dE_du = np.zeros(H)
#            dE_dw = np.zeros((m, H))
#            for k in range(H):
#                for i in range(n):
#                    dE_dv[k] += 2*G[i]*dG_dv[i,k]
#                    dE_du[k] += 2*G[i]*dG_du[i,k]
#                for j in range(m):
#                    for i in range(n):
#                        dE_dw[j,k] += 2*G[i]*dG_dw[i,j,k]
#
#            #--------------------------------------------------------------------
#
#            # Record the current RMSE.
#            rmse = sqrt(E/n)
#            rmse_history[epoch] = rmse
#            if verbose: print(epoch, rmse)

#    def train_minimize(self,
#                       x,  # x-values for training points
#                       trainalg=default_trainalg,  # Training algorithm
#                       wmin = default_wmin,  # Minimum hidden weight value
#                       wmax = default_wmax,  # Maximum hidden weight value
#                       umin = default_umin,  # Minimum hidden bias value
#                       umax = default_umax,  # Maximum hidden bias value
#                       vmin = default_vmin,  # Minimum output weight value
#                       vmax = default_vmax,  # Maximum output weight value
#                       debug = default_debug,
#                       verbose = default_verbose
#    ):
#        """Train the network to solve a 2nd-order ODE IVP. """
#
#        if debug: print('x =', x)
#        if debug: print('trainalg =', trainalg)
#        if debug: print('wmin =', wmin)
#        if debug: print('wmax =', wmax)
#        if debug: print('umin =', umin)
#        if debug: print('umax =', umax)
#        if debug: print('vmin =', vmin)
#        if debug: print('vmax =', vmax)
#        if debug: print('debug =', debug)
#        if debug: print('verbose =', verbose)
#
#        #-----------------------------------------------------------------------
#
#        # Sanity-check arguments.
#        assert len(x) > 0
#        assert trainalg
#        assert vmin < vmax
#        assert wmin < wmax
#        assert umin < umax
#
#        #------------------------------------------------------------------------
#
#        # Determine the number of training points.
#        n = len(x)
#        if debug: print('n =', n)
#
#        # Change notation for convenience.
#        m = len(self.bcf)
#        if debug: print('m =', m)
#        H = len(self.w[0])
#        if debug: print('H =', H)
#
#        #------------------------------------------------------------------------
#
#        # Create an array to hold the weights connecting the input
#        # node to the hidden nodes. The weights are initialized with a
#        # uniform random distribution.
#        w = np.random.uniform(wmin, wmax, (m, H))
#        if debug: print('w =', w)
#
#        # Create an array to hold the biases for the hidden nodes. The
#        # biases are initialized with a uniform random distribution.
#        u = np.random.uniform(umin, umax, H)
#        if debug: print('u =', u)
#
#        # Create an array to hold the weights connecting the hidden
#        # nodes to the output node. The weights are initialized with a
#        # uniform random distribution.
#        v = np.random.uniform(vmin, vmax, H)
#        if debug: print('v =', v)
#
#        #------------------------------------------------------------------------
#
#        # Assemble the network parameters into a single 1-D vector for
#        # use by the minimize() method.
#        # p = [w, u, v]
#        p = np.hstack((w[0], w[1], u, v))
#
#        debug = True
#
#        # Minimize the error function to get the new parameter values.
#        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
#            res = minimize(self.computeError, p, method=trainalg,
#                           args = (x),
#                           options = {'maxiter': 20000, 'disp': True})
#                           callback=print_progress)
#        elif trainalg in ('Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
#            res = minimize(self.computeError, p, method=trainalg,
#                           jac=self.computeErrorGradient, args = (x))
#        if debug: print(res)
#
#        # Unpack the optimized network parameters.
#        self.w[0] = res.x[0:H]
#        self.w[1] = res.x[H:2*H]
#        self.u = res.x[2*H:3*H]
#        self.v = res.x[3*H:4*H]
#        if debug: print('Final w =', self.w)
#        if debug: print('Final u =', self.u)
#        if debug: print('Final v =', self.v)
#
#    def computeError(self, p, x):
#
#        n = len(x)
#        m = len(x[0])
#        H = int(len(p)/4)
#
#        w = np.zeros((m, H))
#        w[0] = p[0:H]
#        w[1] = p[H:2*H]
#        u = p[2*H:3*H]
#        v = p[3*H:4*H]
#
#        z = np.broadcast_to(u,(n,H)) + np.dot(x,w)
#        s = np.vectorize(sigma)(z)
#        s1 = np.vectorize(dsigma_dz)(z)
#        s2 = np.vectorize(d2sigma_dz2)(z)
#
#        w_b = np.broadcast_to(w, (n, m, H))
#        v_b = np.broadcast_to(np.broadcast_to(v,(m,H)),(n,m,H))
#        s1_b = np.broadcast_to(np.expand_dims(s1,axis=1),(n,m,H))
#
#        N = s.dot(v)
#        dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)
#
#        dA_dx = np.zeros((n, m))
#        P = np.zeros(n)
#        dP_dx = np.zeros((n, m))
#        Yt = np.zeros(n)
#        for i in range(n):
#            P[i] = Pf(x[i])
#            Yt[i] = Ytf(x[i], N[i], self.bcf)
#            for j in range(m):
#                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
#                dP_dx[i,j] = delPf[j](x[i])
#        P_b = np.tile(P, (2, 1)).T
#        N_b = np.tile(N, (2, 1)).T
#        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
#
#        G = np.zeros(n)
#        for i in range(n):
#            G[i] = self.Gf(x[i], Yt[i], dYt_dx[i])
#
#        E = np.sum(G**2)
#        return E
#
#    def computeErrorGradient(self, p, x):
#
#        n = len(x)
#        m = len(x[0])
#        H = int(len(p)/4)
#
#        w = np.zeros((m, H))
#        w[0] = p[0:H]
#        w[1] = p[H:2*H]
#        u = p[2*H:3*H]
#        v = p[3*H:4*H]
#
#        z = np.broadcast_to(u,(n,H)) + np.dot(x,w)
#        s = np.vectorize(sigma)(z)
#        s1 = np.vectorize(dsigma_dz)(z)
#        s2 = np.vectorize(d2sigma_dz2)(z)
#
#        x_b = np.broadcast_to(x.T, (H,m,n)).T
#        w_b = np.broadcast_to(w, (n, m, H))
#        v_b = np.broadcast_to(np.broadcast_to(v,(m,H)),(n,m,H))
#        s1_b = np.broadcast_to(np.expand_dims(s1,axis=1),(n,m,H))
#        s2_b = np.broadcast_to(np.expand_dims(s2,axis=1),(n,m,H))

#        N = s.dot(v)
#        dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)
#        dN_dw = v_b*s1_b*x_b
#        dN_du = s1*v
#        dN_dv = s
#        d2N_dwdx = v_b*s1_b + v_b*s2_b*w_b*x_b
#        d2N_dudx = v_b*s2_b*w_b
#        d2N_dvdx = s1_b*w_b
#
#        dA_dx = np.zeros((n, m))
#        P = np.zeros(n)
#        dP_dx = np.zeros((n, m))
#        Yt = np.zeros(n)
#        d2Yt_dudx = np.zeros((n, m, H))
#        d2Yt_dvdx = np.zeros((n, m, H))
#        for i in range(n):
#            P[i] = Pf(x[i])
#            Yt[i] = Ytf(x[i], N[i], self.bcf)
#            for j in range(m):
#                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
#                dP_dx[i,j] = delPf[j](x[i])
#            for k in range(H):
#                for j in range(m):
#                    d2Yt_dudx[i,j,k] = (
#                        P[i]*d2N_dudx[i,j,k] + dP_dx[i,j]*dN_du[i,k]
#                    )
#                    d2Yt_dvdx[i,j,k] = (
#                        P[i]*d2N_dvdx[i,j,k] + dP_dx[i,j]*dN_dv[i,k]
#                    )
#        P_b = np.tile(P, (2, 1)).T
#        N_b = np.tile(N, (2, 1)).T
#        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
#        dYt_dw = dN_dw*np.broadcast_to(P.T, (H, m, n)).T
#        dYt_du = dN_du*np.broadcast_to(P.T, (H, n)).T
#        dYt_dv = dN_dv*np.broadcast_to(P.T, (H, n)).T
#        d2Yt_dwdx = np.broadcast_to(P.T, (H, m, n)).T*d2N_dwdx + \
#                    np.broadcast_to(dP_dx.T, (H, m, n)).T*dN_dw
#
#        G = np.zeros(n)
#        dG_dYt = np.zeros(n)
#        dG_ddelYt = np.zeros((n, m))
#        dG_dw = np.zeros((n, m, H))
#        dG_du = np.zeros((n, H))
#        dG_dv = np.zeros((n, H))
#        for i in range(n):
#            G[i] = self.Gf(x[i], Yt[i], dYt_dx[i])
#            dG_dYt[i] = self.dG_dYf(x[i], Yt[i], dYt_dx[i])
#            for j in range(m):
#                dG_ddelYt[i,j] = self.dG_ddelYf[j](x[i], Yt[i], dYt_dx[i])
#            for k in range(H):
#                dG_dv[i,k] = dG_dYt[i]*dYt_dv[i,k]
#                dG_du[i,k] = dG_dYt[i]*dYt_du[i,k]
#                for j in range(m):
#                    dG_dw[i,j,k] = (
#                        dG_dYt[i]*dYt_dw[i,j,k] +
#                        dG_ddelYt[i,j]*d2Yt_dwdx[i,j,k]
#                    )
#                    dG_du[i,k] += dG_ddelYt[i,j]*d2Yt_dudx[i,j,k]
#                    dG_dv[i,k] += dG_ddelYt[i,j]*d2Yt_dvdx[i,j,k]
#
#        dE_dw = np.zeros((m, H))
#        dE_du = np.zeros(H)
#        dE_dv = np.zeros(H)
#        for k in range(H):
#            for i in range(n):
#                dE_dv[k] += 2*G[i]*dG_dv[i,k]
#                dE_du[k] += 2*G[i]*dG_du[i,k]
#            for j in range(m):
#                for i in range(n):
#                    dE_dw[j,k] += 2*G[i]*dG_dw[i,j,k]
#
#        jac = np.hstack((dE_dw[0], dE_dw[1], dE_du, dE_dv))
#        return jac
#
#    def print_progress(xk):
#        print('xk =', xk)
#
#    def run(self, x):
#        n = len(x)
#        m = len(x[0])
#        H = len(self.w[0])
#        z = np.broadcast_to(self.u, (n, H)) + np.dot(x, self.w)
#        s = np.vectorize(sigma)(z)
#        N = s.dot(self.v)
#        P = np.zeros(n)
#        Yt = np.zeros(n)
#        for i in range(n):
#            P[i] = Pf(x[i])
#            Yt[i] = Ytf(x[i], N[i], self.bcf)
#        return Yt
#
#    def run_derivative(self, x):
#        n = len(x)
#        m = len(x[0])
#        H = len(self.w[0])
#
#        z = np.broadcast_to(self.u,(n,H)) + np.dot(x,self.w)
#        s = np.vectorize(sigma)(z)
#        s1 = np.vectorize(dsigma_dz)(z)
#        s2 = np.vectorize(d2sigma_dz2)(z)
#
#        x_b = np.broadcast_to(x.T, (H,m,n)).T
#        w_b = np.broadcast_to(self.w, (n, m, H))
#        v_b = np.broadcast_to(np.broadcast_to(self.v,(m,H)),(n,m,H))
#        s1_b = np.broadcast_to(np.expand_dims(s1,axis=1),(n,m,H))
#        N = s.dot(self.v)
#        dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)
#
#        dA_dx = np.zeros((n, m))
#        P = np.zeros(n)
#        dP_dx = np.zeros((n, m))
#        for i in range(n):
#            P[i] = Pf(x[i])
#            for j in range(m):
#                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
#                dP_dx[i,j] = delPf[j](x[i])
#        P_b = np.tile(P, (2, 1)).T
#        N_b = np.tile(N, (2, 1)).T
#        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
#        return dYt_dx


if __name__ == '__main__':

    # Create training data.
    nx = ny = 10
    xt = np.linspace(0, 1, nx)
    yt = np.linspace(0, 1, nx)
    x_train = np.array(list(zip(np.tile(xt, 10), np.repeat(yt, 10))))
    n = len(x_train)

    # Test each training algorithm on each equation.
    for pde in ('pde1_ivp_00',):
        print('Examining %s.' % pde)
        pde1ivp = PDE1IVP(pde)
        print(pde1ivp)

        # (Optional) analytical solution and gradient
        if pde1ivp.Yaf:
            print('The analytical solution is:')
            for i in range(n):
                print(pde1ivp.Yaf(x_train[i]))
        if pde1ivp.delYaf:
            print('The analytical gradient is:')
            for i in range(n):
                print(pde1ivp.delYaf[0](x_train[i]),
                      pde1ivp.delYaf[1](x_train[i]))
        print()

        # Create and train the networks.
#        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
#                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
        for trainalg in ('delta',):
            print('Training using %s algorithm.' % trainalg)
            net = NNPDE1IVP(pde1ivp)
            np.random.seed(0)
#            try:
            net.train(x_train, trainalg=trainalg)
#            except (OverflowError, ValueError) as e:
#                print('Error using %s algorithm on %s!' % (trainalg, pde))
#                print(e)
#                print()
#                continue
#            print('The optimized network is:')
#            print(net)
#            Yt = net.run(x_train)
#            delYt = net.run_derivative(x_train)
#            print('The trained solution is:')
#            print('Yt =', Yt)
#            print('The trained gradient is:')
#            print('delYt =', delYt)
#            print()

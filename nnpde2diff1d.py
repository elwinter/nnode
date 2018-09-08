"""
NNODE2DIFF1D - Class to solve 1-D diffusion problems using a neural network

This module provides the functionality to solve 1-D diffusion problems using
a neural network.

Example:
    Create an empty NNODE2DIFF1D object.
        net = NNODE2DIFF1D()
    Create an NNODE2DIFF1D object for a PDE2DIFF1D object.
        net = NNODE2DIFF1D(pde2diff1d_obj)

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

Todo:
    * Expand base functionality.
"""

from math import sqrt
import numpy as np
from scipy.optimize import minimize

from kdelta import kdelta
from pde2diff1d import PDE2DIFF1D
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from slffnn import SLFFNN

# Default values for method parameters
DEFAULT_CLAMP = False
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

# Define the coefficient functions for the trial solution, and their
# derivatives.
def Af(xt, bcf):
    """BC function for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    return (1 - x)*f0(t) + x*f1(t) + \
        (1 - t)*(g0(x) - (1 - x)*g0(0) - x*g0(1))

def dA_dxf(xt, bcf, bcdf):
    """BC function x-derivative for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    ((df0_dt, dg0_dx), (df1_dt, dg1_dx)) = bcdf
    return f1(t) - f0(t) + (1 - t)*(g0(0) - g0(1) + dg0_dx(x))

def dA_dtf(xt, bcf, bcdf):
    """BC function t-derivative for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    ((df0_dt, dg0_dx), (df1_dt, dg1_dx)) = bcdf
    return g0(0)*(1 - x) + x*g0(1) - g0(x) + (1 - x)*df0_dt(t) + x*df1_dt(t)

delAf = (dA_dxf, dA_dtf)

def d2A_dxdxf(xt, bcf, bcdf, bcd2f):
    """BC function xx-derivative for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    ((df0_dt, dg0_dx), (df1_dt, dg1_dx)) = bcdf
    ((d2f0_dt2, d2g0_dx2), (d2f1_dt2, d2g1_dx2)) = bcd2f
    return (1 - t)*d2g0_dx2(x)

def d2A_dxdtf(xt, bcf, bcdf, bcd2f):
    """BC function xt-derivative for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    ((df0_dt, dg0_dx), (df1_dt, dg1_dx)) = bcdf
    ((d2f0_dt2, d2g0_dx2), (d2f1_dt2, d2g1_dx2)) = bcd2f
    return g0(1) - g0(0) + df1_dt(t) - df0_dt(t) - dg0_dx(x)

def d2A_dtdxf(xt, bcf, bcdf, bcd2f):
    """BC function tx-derivative for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    ((df0_dt, dg0_dx), (df1_dt, dg1_dx)) = bcdf
    ((d2f0_dt2, d2g0_dx2), (d2f1_dt2, d2g1_dx2)) = bcd2f
    return g0(1) - g0(0) + df1_dt(t) - df0_dt(t) - dg0_dx(x)

def d2A_dtdtf(xt, bcf, bcdf, bcd2f):
    """BC function tt-derivative for trial solution"""
    (x, t) = xt
    ((f0, g0), (f1, g1)) = bcf
    ((df0_dt, dg0_dx), (df1_dt, dg1_dx)) = bcdf
    ((d2f0_dt2, d2g0_dx2), (d2f1_dt2, d2g1_dx2)) = bcd2f
    return (1 - x)*d2f0_dt2(t) + x*d2f1_dt2(t)

deldelAf = ((d2A_dxdxf, d2A_dxdtf),
            (d2A_dtdxf, d2A_dtdtf))

def Pf(xt):
    """Coefficient function for trial solution"""
    (x, t) = xt
    return x*(1 - x)*t

def dP_dxf(xt):
    """Coefficient function x-derivative for trial solution"""
    (x, t) = xt
    return (1 - 2*x)*t

def dP_dtf(xt):
    """Coefficient function t-derivative for trial solution"""
    (x, t) = xt
    return x*(1 - x)

delPf = (dP_dxf, dP_dtf)

def d2P_dxdxf(xt):
    """Coefficient function xx-derivative for trial solution"""
    (x, t) = xt
    return -2*t

def d2P_dxdtf(xt):
    """Coefficient function xt-derivative for trial solution"""
    (x, t) = xt
    return 1 - 2*x

def d2P_dtdxf(xt):
    """Coefficient function tx-derivative for trial solution"""
    (x, t) = xt
    return 1 - 2*x

def d2P_dtdtf(xt):
    """Coefficient function tt-derivative for trial solution"""
    (x, t) = xt
    return 0

deldelPf = ((d2P_dxdxf, d2P_dxdtf),
            (d2P_dtdxf, d2P_dtdtf))

def Ytf(xt, N, bcf):
    """Trial solution"""
    return Af(xt, bcf) + Pf(xt)*N

# Vectorize the functions used by the network.
sigma_v = np.vectorize(sigma)
dsigma_dz_v = np.vectorize(dsigma_dz)
d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
d3sigma_dz3_v = np.vectorize(d3sigma_dz3)

def print_progress(xk):
    """Callback to print progress message from optimizer"""
    print('xk =', xk)

class NNPDE2DIFF1D(SLFFNN):
    """Solve a 1-D diffusion problem with a neural network"""

    def __init__(self, eq, nhid=DEFAULT_NHID):
        super().__init__()
        self.pde = eq
        self.w = np.zeros((2, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

    def __str__(self):
        s = ''
        s += "NNPDEDIFF1D:\n"
        s += "%s\n" % self.pde
        s += "w = %s\n" % self.w
        s += "u = %s\n" % self.u
        s += "v = %s\n" % self.v
        return s.rstrip()

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
        """Train the network to solve a 1-D diffusion problem"""
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
        """Train using the delta method."""
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

        #---------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)

        # Change notation for convenience.
        ndim = len(self.pde.bcf)
        m = ndim
        H = nhid

        if debug:
            print('n =', n)
            print('m =', m)  # Will always be 2 in this code.
            print('H =', H)

        #---------------------------------------------------------------------

        # Create the network.

        # Create an array to hold the weights connecting the 2
        # input nodes to the hidden nodes. The weights are
        # initialized with a uniform random distribution.
        w = np.random.uniform(wmin, wmax, (m, H))

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        u = np.random.uniform(umin, umax, H)

        # Create an array to hold the weights connecting the hidden nodes
        # to the output node. The weights are initialized with a uniform
        # random distribution.
        v = np.random.uniform(vmin, vmax, H)

        # Create arrays to hold RMSE and parameter history.
        rmse_history = np.zeros(maxepochs)
        w_history = np.zeros((maxepochs, m, H))
        u_history = np.zeros((maxepochs, H))
        v_history = np.zeros((maxepochs, H))

        if debug:
            print('w =', w)
            print('u =', u)
            print('v =', v)

        #---------------------------------------------------------------------

        # Copy to locals to ease notation.
        Gf = self.pde.Gf
        dG_dYf = self.pde.dG_dYf
        dG_ddelYf = self.pde.dG_ddelYf
        dG_ddeldelYf = self.pde.dG_ddeldelYf
        bcf = self.pde.bcf
        bcdf = self.pde.bcdf
        bcd2f = self.pde.bcd2f

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        # Run the network.
        for epoch in range(maxepochs):
            if debug:
                print('Starting epoch %d.' % epoch)

            # Compute the new values of the network parameters.
            w -= eta*dE_dw
            u -= eta*dE_du
            v -= eta*dE_dv
            if debug:
                print('w =', w)
                print('u =', u)
                print('v =', v)

            # If desired, clamp the parameter values at the limits.
            if clamp:
                w[w < wmin] = wmin
                w[w > wmax] = wmax
                u[u < umin] = umin
                u[u > umax] = umax
                v[v < vmin] = vmin
                v[v > vmax] = vmax

            # Save the current parameter values in the history.
            w_history[epoch] = w
            u_history[epoch] = u
            v_history[epoch] = v

            # Compute the net input, the sigmoid function and its derivatives,
            # for each hidden node and each training point.
            z = u + x.dot(w)
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

            #-----------------------------------------------------------------

            # Compute the network output and its derivatives, for each
            # training point.
            N = s.dot(v)
            dN_dx = np.zeros((n, m))
            dN_dv = s
            dN_du = s1*v
            dN_dw = np.zeros((n, m, H))
            d2N_dvdx = np.zeros((n, m, H))
            d2N_dudx = np.zeros((n, m, H))
            d2N_dwdx = np.zeros((n, m, m, H))
            d2N_dxdy = np.zeros((n, m, m))
            d3N_dvdxdy = np.zeros((n, m, m, H))
            d3N_dudxdy = np.zeros((n, m, m, H))
            d3N_dwdxdy = np.zeros((n, m, m, m, H))
            for i in range(n):
                for k in range(H):
                    for j in range(m):
                        dN_dx[i, j] += v[k]*s1[i, k]*w[j, k]
                        dN_dw[i, j, k] = v[k]*s1[i, k]*x[i, j]
                        d2N_dvdx[i, j, k] = s1[i, k]*w[j, k]
                        d2N_dudx[i, j, k] = v[k]*s2[i, k]*w[j, k]
                        for jj in range(m):
                            d2N_dxdy[i, j, jj] += v[k]*s2[i, k]*w[j, k]*w[jj, k]
                            d2N_dwdx[i, j, jj, k] = \
                            v[k]*(s1[i, k]*kdelta(j, jj) + s2[i, k]*w[jj, k]*x[i, j])
                            d3N_dvdxdy[i, j, jj, k] = s2[i, k]*w[j, k]*w[jj, k]
                            d3N_dudxdy[i, j, jj, k] = v[k]*s3[i, k]*w[j, k]*w[jj, k]
                            for jjj in range(m):
                                d3N_dwdxdy[i, j, jj, jjj, k] = \
                                v[k]*s2[i, k]*(w[jj, k]*kdelta(j, jjj) + \
                                 w[jjj, k]*kdelta(j, jj)) + \
                                 v[k]*s3[i, k]*w[jj, k]*w[jjj, k]*x[i, j]
            if debug:
                print('N =', N)
                print('dN_dx =', dN_dx)
                print('dN_dv =', dN_dv)
                print('dN_du =', dN_du)
                print('dN_dw =', dN_dw)
                print('d2N_dvdx =', d2N_dvdx)
                print('d2N_dudx =', d2N_dudx)
                print('d2N_dwdx =', d2N_dwdx)
                print('d2N_dxdy =', d2N_dxdy)
                print('d3N_dvdxdy =', d3N_dvdxdy)
                print('d3N_dudxdy =', d3N_dudxdy)
                print('d3N_dwdxdy =', d3N_dwdxdy)

            #-----------------------------------------------------------------

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            A = np.zeros(n)
            dA_dx = np.zeros((n, m))
            d2A_dxdy = np.zeros((n, m, m))
            P = np.zeros(n)
            dP_dx = np.zeros((n, m))
            d2P_dxdy = np.zeros((n, m, m))
            Yt = np.zeros(n)
            dYt_dx = np.zeros((n, m))
            dYt_dv = np.zeros((n, H))
            dYt_du = np.zeros((n, H))
            dYt_dw = np.zeros((n, m, H))
            d2Yt_dvdx = np.zeros((n, m, H))
            d2Yt_dudx = np.zeros((n, m, H))
            d2Yt_dwdx = np.zeros((n, m, m, H))
            d2Yt_dxdy = np.zeros((n, m, m))
            d3Yt_dvdxdy = np.zeros((n, m, m, H))
            d3Yt_dudxdy = np.zeros((n, m, m, H))
            d3Yt_dwdxdy = np.zeros((n, m, m, m, H))
            for i in range(n):
                A[i] = Af(x[i], bcf)
                P[i] = Pf(x[i])
                Yt[i] = Ytf(x[i], N[i], bcf)
                for j in range(m):
                    dA_dx[i, j] = delAf[j](x[i], bcf, bcdf)
                    dP_dx[i, j] = delPf[j](x[i])
                    dYt_dx[i, j] = dA_dx[i, j] + P[i]*dN_dx[i, j] + dP_dx[i, j]*N[i]
                    for jj in range(m):
                        d2A_dxdy[i, j, jj] = deldelAf[j][jj](x[i], bcf, bcdf, bcd2f)
                        d2P_dxdy[i, j, jj] = deldelPf[j][jj](x[i])
                        d2Yt_dxdy[i, j, jj] = (
                            d2A_dxdy[i, j, jj] +
                            P[i]*d2N_dxdy[i, j, jj] +
                            dP_dx[i, j]*dN_dx[i, jj] +
                            dP_dx[i, jj]*dN_dx[i, j] +
                            d2P_dxdy[i, j, jj]*N[i]
                        )
                for k in range(H):
                    dYt_dv[i, k] = P[i]*dN_dv[i, k]
                    dYt_du[i, k] = P[i]*dN_du[i, k]
                    for j in range(m):
                        dYt_dw[i, j, k] = P[i]*dN_dw[i, j, k]
                        d2Yt_dvdx[i, j, k] = (
                            P[i]*d2N_dvdx[i, j, k] + dP_dx[i, j]*dN_dv[i, k]
                        )
                        d2Yt_dudx[i, j, k] = (
                            P[i]*d2N_dudx[i, j, k] + dP_dx[i, j]*dN_du[i, k]
                        )
                        for jj in range(m):
                            d2Yt_dwdx[i, jj, j, k] = (
                                P[i]*d2N_dwdx[i, jj, j, k] + dP_dx[i, j]*dN_dw[i, j, k]
                            )
                            d3Yt_dvdxdy[i, j, jj, k] = (
                                P[i]*d3N_dvdxdy[i, j, jj, k] +
                                dP_dx[i, j]*d2N_dvdx[i, jj, k] +
                                dP_dx[i, jj]*d2N_dvdx[i, j, k] +
                                d2P_dxdy[i, j, jj]*dN_dv[i, k]
                            )
                            d3Yt_dudxdy[i, j, jj, k] = (
                                P[i]*d3N_dudxdy[i, j, jj, k] +
                                dP_dx[i, j]*d2N_dudx[i, jj, k] +
                                dP_dx[i, jj]*d2N_dudx[i, j, k] +
                                d2P_dxdy[i, j, jj]*dN_du[i, k]
                            )
                            for jjj in range(m):
                                d3Yt_dwdxdy[i, jjj, j, jj, k] = (
                                    P[i]*d3N_dwdxdy[i, jjj, j, jj, k] +
                                    dP_dx[i, j]*d2N_dwdx[i, jjj, jj, k] +
                                    dP_dx[i, jj]*d2N_dwdx[i, jj, j, k] +
                                    d2P_dxdy[i, j, jj]*dN_dw[i, jjj, k]
                                )
            if debug:
                print('A =', A)
                print('dA_dx =', dA_dx)
                print('d2A_dxdy =', d2A_dxdy)
                print('P =', P)
                print('dP_dx =', dP_dx)
                print('d2P_dxdy =', d2P_dxdy)
                print('Yt =', Yt)
                print('dYt_dx =', dYt_dx)
                print('dYt_dv =', dYt_dv)
                print('dYt_du =', dYt_du)
                print('dYt_dw =', dYt_dw)
                print('d2Yt_dvdx =', d2Yt_dvdx)
                print('d2Yt_dudx =', d2Yt_dudx)
                print('d2Yt_dwdx =', d2Yt_dwdx)
                print('d2Yt_dxdy =', d2Yt_dxdy)
                print('d3Yt_dvdxdy =', d3Yt_dvdxdy)
                print('d3Yt_dudxdy =', d3Yt_dudxdy)
                print('d3Yt_dwdxdy =', d3Yt_dwdxdy)

            #-----------------------------------------------------------------

            # Compute the value of the original differential equation
            # for each training point, and its derivatives.
            G = np.zeros(n)
            dG_dYt = np.zeros(n)
            dG_ddelYt = np.zeros((n, m))
            dG_ddeldelYt = np.zeros((n, m, m))
            dG_dv = np.zeros((n, H))
            dG_du = np.zeros((n, H))
            dG_dw = np.zeros((n, m, H))
            for i in range(n):
                G[i] = Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
                dG_dYt[i] = dG_dYf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
                for j in range(m):
                    dG_ddelYt[i, j] = \
                    dG_ddelYf[j](x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
                    for jj in range(m):
                        dG_ddeldelYt[i, j, jj] = \
                        dG_ddeldelYf[j][jj](x[i], Yt[i], dYt_dx[i],
                                            d2Yt_dxdy[i])
                for k in range(H):
                    dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                    dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                    for j in range(m):
                        dG_dv[i, k] += dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
                        dG_du[i, k] += dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
                        dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]
                        for l in range(m):
                            dG_dw[i, j, k] += dG_ddelYt[i, l]*d2Yt_dwdx[i, j, l, k]
                            for ll in range(m):
                                dG_dw[i, j, k] += (
                                    dG_ddeldelYt[i, l, ll]*d3Yt_dwdxdy[i, j, l, ll, k]
                                )
                        for jj in range(m):
                            dG_dv[i, k] += dG_ddeldelYt[i, j, jj]*d3Yt_dvdxdy[i, j, jj, k]
                            dG_du[i, k] += dG_ddeldelYt[i, j, jj]*d3Yt_dudxdy[i, j, jj, k]
            if debug:
                print('G =', G)
                print('dG_dYt =', dG_dYt)
                print('dG_ddelYt =', dG_ddelYt)
                print('dG_ddeldelYt =', dG_ddeldelYt)
                print('dG_dv =', dG_dv)
                print('dG_du =', dG_du)
                print('dG_dw =', dG_dw)

            #-----------------------------------------------------------------

            # Compute the error function for this pass.
            E = sum(G**2)
            if debug:
                print('E =', E)

            # Compute the partial derivatives of the error with respect to the
            # network parameters.
            dE_dv = np.zeros(H)
            dE_du = np.zeros(H)
            dE_dw = np.zeros((m, H))
            for k in range(H):
                for i in range(n):
                    dE_dv[k] += 2*G[i]*dG_dv[i, k]
                    dE_du[k] += 2*G[i]*dG_du[i, k]
                for j in range(m):
                    for i in range(n):
                        dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]
            if debug:
                print('dE_dv =', dE_dv)
                print('dE_du =', dE_du)
                print('dE_dw =', dE_dw)

            #-----------------------------------------------------------------

            # Record the current RMSE.
            rmse = sqrt(E/n)
            rmse_history[epoch] = rmse
            if verbose:
                print(epoch, rmse)

        # Save the final values of the network parameters.
        self.w = w
        self.u = u
        self.v = v

    def train_minimize(self,
                       x,                          # x-values for training points
                       trainalg=DEFAULT_TRAINALG,   # Training algorithm
                       wmin=DEFAULT_WMIN,         # Minimum hidden weight value
                       wmax=DEFAULT_WMAX,         # Maximum hidden weight value
                       umin=DEFAULT_UMIN,         # Minimum hidden bias value
                       umax=DEFAULT_UMAX,         # Maximum hidden bias value
                       vmin=DEFAULT_VMIN,         # Minimum output weight value
                       vmax=DEFAULT_VMAX,         # Maximum output weight value
                       debug=DEFAULT_DEBUG,
                       verbose=DEFAULT_VERBOSE
                       ):
        """Train using the scipy minimize() function"""
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

        #---------------------------------------------------------------------

        # Sanity-check arguments.
        assert x.any()
        assert trainalg
        assert vmin < vmax
        assert wmin < wmax
        assert umin < umax

        #---------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)

        # Change notation for convenience.
        m = len(self.pde.bcf)
        H = len(self.w[0])

        if debug:
            print('n =', n)
            print('m =', m)
            print('H =', H)

        #---------------------------------------------------------------------

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        w = np.random.uniform(wmin, wmax, (m, H))

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        u = np.random.uniform(umin, umax, H)

        # Create an array to hold the weights connecting the hidden
        # nodes to the output node. The weights are initialized with a
        # uniform random distribution.
        v = np.random.uniform(vmin, vmax, H)

        if debug:
            print('w =', w)
            print('u =', u)
            print('v =', v)

        #---------------------------------------------------------------------

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        # p = [w, u, v]
        p = np.hstack((w[0], w[1], u, v))

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            res = minimize(self.computeError, p, method=trainalg,
                           args=(x),
                           options={'disp': True},)
#                            callback=print_progress)
        elif trainalg in ('Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            res = minimize(self.computeError, p, method=trainalg,
                           jac=self.computeErrorGradient, args=(x),)
#                           callback=print_progress)
        if debug:
            print(res)

        # Unpack the optimized network parameters.
        self.w[0] = res.x[0:H]
        self.w[1] = res.x[H:2*H]
        self.u = res.x[2*H:3*H]
        self.v = res.x[3*H:4*H]

    def computeError(self, p, x):
        """Compute the current error in the trained solution."""

        # Copy to locals to ease notation.
        Gf = self.pde.Gf
        bcf = self.pde.bcf
        bcdf = self.pde.bcdf
        bcd2f = self.pde.bcd2f

        n = len(x)
        m = len(x[0])
        H = int(len(p)/(m + 2))

        w = np.zeros((m, H))
        w[0] = p[0:H]
        w[1] = p[H:2*H]
        u = p[2*H:3*H]
        v = p[3*H:4*H]

        z = u + x.dot(w)
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)

        N = s.dot(v)
        dN_dx = np.zeros((n, m))
        d2N_dxdy = np.zeros((n, m, m))
        for i in range(n):
            for k in range(H):
                for j in range(m):
                    dN_dx[i, j] += v[k]*s1[i, k]*w[j, k]
                    for jj in range(m):
                        d2N_dxdy[i, j, jj] += v[k]*s2[i, k]*w[j, k]*w[jj, k]

        dA_dx = np.zeros((n, m))
        d2A_dxdy = np.zeros((n, m, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        d2P_dxdy = np.zeros((n, m, m))
        Yt = np.zeros(n)
        dYt_dx = np.zeros((n, m))
        d2Yt_dxdy = np.zeros((n, m, m))
        for i in range(n):
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], bcf)
            for j in range(m):
                dA_dx[i, j] = delAf[j](x[i], bcf, bcdf)
                dP_dx[i, j] = delPf[j](x[i])
                dYt_dx[i, j] = dA_dx[i, j] + P[i]*dN_dx[i, j] + dP_dx[i, j]*N[i]
                for jj in range(m):
                    d2A_dxdy[i, j, jj] = deldelAf[j][jj](x[i], bcf, bcdf, bcd2f)
                    d2P_dxdy[i, j, jj] = deldelPf[j][jj](x[i])
                    d2Yt_dxdy[i, j, jj] = (
                        d2A_dxdy[i, j, jj] +
                        P[i]*d2N_dxdy[i, j, jj] +
                        dP_dx[i, j]*dN_dx[i, jj] +
                        dP_dx[i, jj]*dN_dx[i, j] +
                        d2P_dxdy[i, j, jj]*N[i]
                    )

        G = np.zeros(n)
        for i in range(n):
            G[i] = Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])

        E = sum(G**2)
        return E

    def computeErrorGradient(self, p, x):
        """Compute the error gradient in the trained solution."""

        # Copy to locals to ease notation.
        Gf = self.pde.Gf
        dG_dYf = self.pde.dG_dYf
        dG_ddelYf = self.pde.dG_ddelYf
        bcf = self.pde.bcf
        bcdf = self.pde.bcdf
        bcd2f = self.pde.bcd2f

        n = len(x)
        m = len(x[0])
        H = int(len(p)/(m + 2))

        w = np.zeros((m, H))
        w[0] = p[0:H]
        w[1] = p[H:2*H]
        u = p[2*H:3*H]
        v = p[3*H:4*H]

        z = u + x.dot(w)
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)

        N = s.dot(v)
        dN_dx = np.zeros((n, m))
        dN_dv = s
        dN_du = s1*v
        dN_dw = np.zeros((n, m, H))
        d2N_dvdx = np.zeros((n, m, H))
        d2N_dudx = np.zeros((n, m, H))
        d2N_dwdx = np.zeros((n, m, m, H))
        d2N_dxdy = np.zeros((n, m, m))
        for i in range(n):
            for k in range(H):
                for j in range(m):
                    dN_dx[i, j] += v[k]*s1[i, k]*w[j, k]
                    dN_dw[i, j, k] = v[k]*s1[i, k]*x[i, j]
                    d2N_dvdx[i, j, k] = s1[i, k]*w[j, k]
                    d2N_dudx[i, j, k] = v[k]*s2[i, k]*w[j, k]
                    for jj in range(m):
                        d2N_dxdy[i, j, jj] += v[k]*s2[i, k]*w[j, k]*w[jj, k]
                        d2N_dwdx[i, j, jj, k] = v[k]*(s1[i, k]*kdelta(j, jj) \
                                + s2[i, k]*w[jj, k]*x[i, j])
        dA_dx = np.zeros((n, m))
        d2A_dxdy = np.zeros((n, m, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        d2P_dxdy = np.zeros((n, m, m))
        Yt = np.zeros(n)
        dYt_dx = np.zeros((n, m))
        dYt_dv = np.zeros((n, H))
        dYt_du = np.zeros((n, H))
        dYt_dw = np.zeros((n, m, H))
        d2Yt_dvdx = np.zeros((n, m, H))
        d2Yt_dudx = np.zeros((n, m, H))
        d2Yt_dwdx = np.zeros((n, m, m, H))
        d2Yt_dxdy = np.zeros((n, m, m))
        for i in range(n):
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], bcf)
            for j in range(m):
                dA_dx[i, j] = delAf[j](x[i], bcf, bcdf)
                dP_dx[i, j] = delPf[j](x[i])
                dYt_dx[i, j] = dA_dx[i, j] + P[i]*dN_dx[i, j] + dP_dx[i, j]*N[i]
                for jj in range(m):
                    d2A_dxdy[i, j, jj] = deldelAf[j][jj](x[i], bcf, bcdf, bcd2f)
                    d2P_dxdy[i, j, jj] = deldelPf[j][jj](x[i])
                    d2Yt_dxdy[i, j, jj] = (
                        d2A_dxdy[i, j, jj] +
                        P[i]*d2N_dxdy[i, j, jj] +
                        dP_dx[i, j]*dN_dx[i, jj] +
                        dP_dx[i, jj]*dN_dx[i, j] +
                        d2P_dxdy[i, j, jj]*N[i]
                    )
            for k in range(H):
                dYt_dv[i, k] = P[i]*dN_dv[i, k]
                dYt_du[i, k] = P[i]*dN_du[i, k]
                for j in range(m):
                    dYt_dw[i, j, k] = P[i]*dN_dw[i, j, k]
                    d2Yt_dvdx[i, j, k] = (
                        P[i]*d2N_dvdx[i, j, k] + dP_dx[i, j]*dN_dv[i, k]
                    )
                    d2Yt_dudx[i, j, k] = (
                        P[i]*d2N_dudx[i, j, k] + dP_dx[i, j]*dN_du[i, k]
                    )
                    for jj in range(m):
                        d2Yt_dwdx[i, jj, j, k] = (
                            P[i]*d2N_dwdx[i, jj, j, k] + dP_dx[i, j]*dN_dw[i, j, k]
                        )

        G = np.zeros(n)
        dG_dYt = np.zeros(n)
        dG_ddelYt = np.zeros((n, m))
        dG_dv = np.zeros((n, H))
        dG_du = np.zeros((n, H))
        dG_dw = np.zeros((n, m, H))
        for i in range(n):
            G[i] = Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
            dG_dYt[i] = dG_dYf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
            for j in range(m):
                dG_ddelYt[i, j] = dG_ddelYf[j](x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
            for k in range(H):
                dG_dv[i, k] = dG_dYt[i]*dYt_dv[i, k]
                dG_du[i, k] = dG_dYt[i]*dYt_du[i, k]
                for j in range(m):
                    dG_dv[i, k] += dG_ddelYt[i, j]*d2Yt_dvdx[i, j, k]
                    dG_du[i, k] += dG_ddelYt[i, j]*d2Yt_dudx[i, j, k]
                    dG_dw[i, j, k] = dG_dYt[i]*dYt_dw[i, j, k]

        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros((m, H))
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2*G[i]*dG_dv[i, k]
                dE_du[k] += 2*G[i]*dG_du[i, k]
            for j in range(m):
                for i in range(n):
                    dE_dw[j, k] += 2*G[i]*dG_dw[i, j, k]

        jac = np.hstack((dE_dw[0], dE_dw[1], dE_du, dE_dv))
        return jac

    def run(self, x):
        """Compute the trained solution."""
        n = len(x)
        m = len(x[0])
        H = len(self.w[0])
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = self.u[k]
                for j in range(m):
                    z[i, k] += self.w[j, k]*x[i, j]
        s = np.vectorize(sigma)(z)
        N = s.dot(self.v)
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = Ytf(x[i], N[i], self.pde.bcf)
        return Yt

    def run_derivative(self, x):
        """Compute the trained derivative."""
        n = len(x)
        m = len(x[0])
        H = len(self.w[0])

        z = np.zeros((n, H))
        s = np.zeros((n, H))
        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i, k] = self.u[k]
                for j in range(m):
                    z[i, k] += self.w[j, k]*x[i, j]
                    s[i, k] = sigma(z[i, k])
                    s1[i, k] = dsigma_dz(z[i, k])

        N = np.zeros(n)
        dN_dx = np.zeros((n, m))
        for i in range(n):
            for k in range(H):
                N[i] += self.v[k]*s[i, k]
                for j in range(m):
                    dN_dx[i, j] += self.v[k]*s1[i, k]*self.w[j, k]

        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        dYt_dx = np.zeros((n, m))
        for i in range(n):
            P[i] = Pf(x[i])
            for j in range(m):
                dA_dx[i, j] = delAf[j](x[i], self.pde.bcf, self.pde.bcdf)
                dP_dx[i, j] = delPf[j](x[i])
                dYt_dx[i, j] = dA_dx[i, j] + P[i]*dN_dx[i, j] + dP_dx[i, j]*N[i]

        return dYt_dx


if __name__ == '__main__':

    # Create training data.
    nx = 11
    nt = 10
    xt = np.linspace(0, 1, nx)
    yt = np.linspace(0, 1, nt)
    x_train = np.array(list(zip(np.tile(xt, nt), np.repeat(yt, nx))))
    print('x_train =', x_train)
    n = len(x_train)

    # Test each training algorithm on each equation.
    for pde in ('diff1d_0', 'diff1d_flat', 'diff1d_rampup', 'diff1d_rampdown',
                'diff1d_sine', 'diff1d_triangle', 'diff1d_increase',
                'diff1d_decrease', 'diff1d_sinewave'):
        print('Examining %s.' % pde)
        pde2diff1d = PDE2DIFF1D(pde)
        print(pde2diff1d)
        m = len(pde2diff1d.bcdf)
        net = NNPDE2DIFF1D(pde2diff1d)
        print(net)
        Ya = np.zeros(n)
        for i in range(n):
            Ya[i] = net.pde.Yaf(x_train[i])
        print('The analytical solution is:')
        print('Ya =', Ya.reshape(nx, nt))
        print()
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Training using %s algorithm.' % trainalg)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg)
            except (OverflowError, ValueError) as e:
                print('Error using %s algorithm on %s!' % (trainalg, pde))
                print(e)
                print()
                continue
            Yt = net.run(x_train)
            print('The trained solution is:')
            print('Yt =', Yt.reshape(nt, nx))
            print()
            print('The error in the trained solution is:')
            print('Yt - Ya =', (Yt - Ya).reshape(nt, nx))
            print()

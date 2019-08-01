# Notation notes:
# The suffix 'f' usually denotes an object that is a function.

# The suffix '_v' denotes a function that has been vectorized with
# np.vectorize().

from scipy.optimize import minimize
from math import sqrt
import numpy as np

from kdelta import kdelta
from pde2bvp import PDE2BVP
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3
from slffnn import SLFFNN

#********************************************************************************

# # Default values for program parameters
default_clamp = False
default_debug = False
default_eta = 0.01
default_maxepochs = 1000
default_nhid = 10
default_randomize = False
default_rmseout = 'rmse.dat'
default_trainalg = 'delta'
default_umax = 1
default_umin = -1
default_verbose = False
default_vmax = 1
default_vmin = -1
default_wmax = 1
default_wmin = -1

#********************************************************************************

# # The domain of the trial solution is assumed to be [[0, 1], [0, 1]].

# Define the coefficient functions for the trial solution, and their derivatives.
def Af(xy, bcf):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    A = (
        (1 - x)*f0f(y) + x*f1f(y)
        + (1 - y)*(g0f(x) - (1 - x)*g0f(0) - x*g0f(1))
        + y*(g1f(x) - (1 - x)*g1f(0) - x*g1f(1))
    )
    return A

def dA_dxf(xy, bcf, bcdf):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf)) = bcdf
    dA_dx = (
        -f0f(y) + f1f(y) + (1 - y)*(dg0_dxf(x) + g0f(0) - g0f(1))
        + y*(dg1_dxf(x) + g1f(0) - g1f(1))
        )
    return dA_dx

def dA_dyf(xy, bcf, bcdf):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf)) = bcdf
    dA_dy = (
        (1 - x)*df0_dyf(y) + x*df1_dyf(y)
        - (g0f(x) - (1 - x)*g0f(0) - x*g0f(1))
        + (g1f(x) - (1 - x)*g1f(0) - x*g1f(1))
    )
    return dA_dy

delAf = (dA_dxf, dA_dyf)

def d2A_dxdxf(xy, bcf, bcdf, bcd2f):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf)) = bcdf
    ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f)) = bcd2f
    return (1 - y)*d2g0_dx2f(x) + y*d2g1_dx2f(x)

def d2A_dxdyf(xy, bcf, bcdf, bcd2f):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf)) = bcdf
    ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f)) = bcd2f
    return (
        -df0_dyf(y) + df1_dyf(y) - dg0_dxf(x) + dg1_dxf(x) -
        g0f(0) + g0f(1) + g1f(0) - g1f(1)
    )

def d2A_dydxf(xy, bcf, bcdf, bcd2f):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf)) = bcdf
    ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f)) = bcd2f
    return (
        -df0_dyf(y) + df1_dyf(y) - dg0_dxf(x) + dg1_dxf(x) -
        g0f(0) + g0f(1) + g1f(0) - g1f(1)
    )

def d2A_dydyf(xy, bcf, bcdf, bcd2f):
    (x, y) = xy
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dyf, dg0_dxf), (df1_dyf, dg1_dxf)) = bcdf
    ((d2f0_dy2f, d2g0_dx2f), (d2f1_dy2f, d2g1_dx2f)) = bcd2f
    return (1 - x)*d2f0_dy2f(y) + y*d2f1_dy2f(y)

deldelAf = ((d2A_dxdxf, d2A_dxdyf),
            (d2A_dydxf, d2A_dydyf))

def Pf(xy):
    (x, y) = xy
    P = x*(1 - x)*y*(1 - y)
    return P

def dP_dxf(xy):
    (x, y) = xy
    dP_dx = (1 - 2*x)*y*(1 - y)
    return dP_dx

def dP_dyf(xy):
    (x, y) = xy
    dP_dy = x*(1 - x)*(1 - 2*y)
    return dP_dy

delPf = (dP_dxf, dP_dyf)

def d2P_dxdxf(xy):
    (x, y) = xy
    return -2*y*(1 - y)

def d2P_dxdyf(xy):
    (x, y) = xy
    return (1 - 2*x)*(1 - 2*y)

def d2P_dydxf(xy):
    (x, y) = xy
    return (1 - 2*x)*(1 - 2*y)

def d2P_dydyf(xy):
    (x, y) = xy
    return -2*x*(1 - x)

deldelPf = ((d2P_dxdxf, d2P_dxdyf),
            (d2P_dydxf, d2P_dydyf))

# Define the trial solution.
def Ytf(xy, N, bcf):
    A = Af(xy, bcf)
    P = Pf(xy)
    Yt = A + P*N
    return Yt

# Vectorize sigma functions.
# sigma_v = np.vectorize(sigma)
# dsigma_dz_v = np.vectorize(dsigma_dz)
# d2sigma_dz2_v = np.vectorize(d2sigma_dz2)

class NNPDE2BVP(SLFFNN):

    def __init__(self, pde2bvp, nhid=default_nhid):

        # PDE, in the form G(x,Y,delY,deldelY)=0.
        self.Gf = pde2bvp.Gf

        # dG/dY
        self.dG_dYf = pde2bvp.dG_dYf

        # dG/d(delY)
        self.dG_ddelYf = pde2bvp.dG_ddelYf

        # dG/d(deldelY)
        self.dG_ddeldelYf = pde2bvp.dG_ddeldelYf

        # Boundary condition functions
        self.bcf = pde2bvp.bcf

        # Boundary condition derivatives
        self.bcdf = pde2bvp.bcdf

        # Boundary condition 2nd derivatives
        self.bcd2f = pde2bvp.bcd2f

        # Analytical solution (optional)
        if pde2bvp.Yaf:
            self.Yaf = pde2bvp.Yaf

        # Analytical gradient (optional)
        if pde2bvp.delYaf:
            self.delYaf = pde2bvp.delYaf

        # Analytical Hessian (optional)
        if pde2bvp.deldelYaf:
            self.deldelYaf = pde2bvp.deldelYaf

        # Create arrays to hold the weights and biases, initially all 0.
        self.w = np.zeros((2, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

    def __str__(self):
        s = "NNPDE2BVP:\n"
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
        bcf = self.bcf
        bcdf = self.bcdf
        bcd2f = self.bcd2f
        dG_dYf = self.dG_dYf
        dG_ddelYf = self.dG_ddelYf
        dG_ddeldelYf = self.dG_ddeldelYf

        #------------------------------------------------------------------------

        # Determine the number of training points.
        n = len(x)
        if debug: print('n =', n)

        # Change notation for convenience.
        m = len(self.bcf)
        if debug: print('m =', m)  # Will always be 2 in this code.
        H = nhid
        if debug: print('H =', H)

        #------------------------------------------------------------------------

        # Create the network.

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        self.w = np.random.uniform(wmin, wmax, (2, H))
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
        w_history = np.zeros((maxepochs, 2, H))
        u_history = np.zeros((maxepochs, H))
        v_history = np.zeros((maxepochs, H))

        #------------------------------------------------------------------------

        # Initial parameter deltas are 0.
        dE_dw = np.zeros((2, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)

        verbose = True

        # Run the network.
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

            # Compute the net input, the sigmoid function and its derivatives,
            # for each hidden node and each training point.
            z =  np.zeros((n, H))
            s =  np.zeros((n, H))
            s1 = np.zeros((n, H))
            s2 = np.zeros((n, H))
            s3 = np.zeros((n, H))
            for i in range(n):
                for k in range(H):
                    z[i,k] = self.u[k]
                    for j in range(m):
                        z[i,k] += self.w[j,k]*x[i,j]
                        s[i,k] = sigma(z[i,k])
                        s1[i,k] = dsigma_dz(z[i,k])
                        s2[i,k] = d2sigma_dz2(z[i,k])
                        s3[i,k] = d3sigma_dz3(z[i,k])
            if debug: print('z =', z)
            if debug: print('s =', s)
            if debug: print('s1 =', s1)
            if debug: print('s2 =', s2)
            if debug: print('s3 =', s3)

            #--------------------------------------------------------------------

            # Compute the network output and its derivatives, for each
            # training point.
            N = np.zeros(n)
            dN_dx = np.zeros((n, m))
            dN_dv = np.zeros((n, H))
            dN_du = np.zeros((n, H))
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
                    N[i] += self.v[k]*s[i,k]
                    dN_dv[i,k] = s[i,k]
                    dN_du[i,k] = self.v[k]*s1[i,k]
                    for j in range(m):
                        dN_dx[i,j] += self.v[k]*s1[i,k]*self.w[j,k]
                        dN_dw[i,j,k] = self.v[k]*s1[i,k]*x[i,j]
                        d2N_dvdx[i,j,k] = s1[i,k]*self.w[j,k]
                        d2N_dudx[i,j,k] = self.v[k]*s2[i,k]*self.w[j,k]
                        for jj in range(m):
                            d2N_dxdy[i,j,jj] += self.v[k]*s2[i,k]*self.w[j,k]*self.w[jj,k]
                            d2N_dwdx[i,j,jj,k] = self.v[k]*(s1[i,k]*kdelta(j,jj) +
                                                       s2[i,k]*self.w[jj,k]*x[i,j])
                            d3N_dvdxdy[i,j,jj,k] = s2[i,k]*self.w[j,k]*self.w[jj,k]
                            d3N_dudxdy[i,j,jj,k] = self.v[k]*s3[i,k]*self.w[j,k]*self.w[jj,k]
                            for jjj in range(m):
                                d3N_dwdxdy[i,j,jj,jjj,k] = (
                                    self.v[k]*s2[i,k]*(self.w[jj,k]*kdelta(j,jjj)
                                                  + self.w[jjj,k]*kdelta(j,jj)) +
                                    self.v[k]*s3[i,k]*self.w[jj,k]*self.w[jjj,k]*x[i,j]
                                )
            if debug: print('N =', N)
            if debug: print('dN_dx =', dN_dx)
            if debug: print('dN_dv =', dN_dv)
            if debug: print('dN_du =', dN_du)
            if debug: print('dN_dw =', dN_dw)
            if debug: print('d2N_dvdx =', d2N_dvdx)
            if debug: print('d2N_dudx =', d2N_dudx)
            if debug: print('d2N_dwdx =', d2N_dwdx)
            if debug: print('d2N_dxdy =', d2N_dxdy)
            if debug: print('d3N_dvdxdy =', d3N_dvdxdy)
            if debug: print('d3N_dudxdy =', d3N_dudxdy)
            if debug: print('d3N_dwdxdy =', d3N_dwdxdy)

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
                    dA_dx[i,j] = delAf[j](x[i], bcf, bcdf)
                    dP_dx[i,j] = delPf[j](x[i])
                    dYt_dx[i,j] = dA_dx[i,j] + P[i]*dN_dx[i,j] + dP_dx[i,j]*N[i]
                    for jj in range(m):
                        d2A_dxdy[i,j,jj] = deldelAf[j][jj](x[i], bcf, bcdf, bcd2f)
                        d2P_dxdy[i,j,jj] = deldelPf[j][jj](x[i])
                        d2Yt_dxdy[i,j,jj] = (
                            d2A_dxdy[i,j,jj] +
                            P[i]*d2N_dxdy[i,j,jj] +
                            dP_dx[i,j]*dN_dx[i,jj] +
                            dP_dx[i,jj]*dN_dx[i,j] +
                            d2P_dxdy[i,j,jj]*N[i]
                        )
                for k in range(H):
                    dYt_dv[i,k] = P[i]*dN_dv[i,k]
                    dYt_du[i,k] = P[i]*dN_du[i,k]
                    for j in range(m):
                        dYt_dw[i,j,k] = P[i]*dN_dw[i,j,k]
                        d2Yt_dvdx[i,j,k] = (
                            P[i]*d2N_dvdx[i,j,k] + dP_dx[i,j]*dN_dv[i,k]
                        )
                        d2Yt_dudx[i,j,k] = (
                            P[i]*d2N_dudx[i,j,k] + dP_dx[i,j]*dN_du[i,k]
                        )
                        for jj in range(m):
                            d2Yt_dwdx[i,jj,j,k] = (
                                P[i]*d2N_dwdx[i,jj,j,k] + dP_dx[i,j]*dN_dw[i,j,k]
                            )
                            d3Yt_dvdxdy[i,j,jj,k] = (
                                P[i]*d3N_dvdxdy[i,j,jj,k] +
                                dP_dx[i,j]*d2N_dvdx[i,jj,k] +
                                dP_dx[i,jj]*d2N_dvdx[i,j,k] +
                                d2P_dxdy[i,j,jj]*dN_dv[i,k]
                            )
                            d3Yt_dudxdy[i,j,jj,k] = (
                                P[i]*d3N_dudxdy[i,j,jj,k] +
                                dP_dx[i,j]*d2N_dudx[i,jj,k] +
                                dP_dx[i,jj]*d2N_dudx[i,j,k] +
                                d2P_dxdy[i,j,jj]*dN_du[i,k]
                            )
                            for jjj in range(m):
                                d3Yt_dwdxdy[i,jjj,j,jj,k] = (
                                    P[i]*d3N_dwdxdy[i,jjj,j,jj,k] +
                                    dP_dx[i,j]*d2N_dwdx[i,jjj,jj,k] +
                                    dP_dx[i,jj]*d2N_dwdx[i,jj,j,k] +
                                    d2P_dxdy[i,j,jj]*dN_dw[i,jjj,k]
                                )
            if debug: print('A =', A)
            if debug: print('dA_dx =', dA_dx)
            if debug: print('d2A_dxdy =', d2A_dxdy)
            if debug: print('P =', P)
            if debug: print('dP_dx =', dP_dx)
            if debug: print('d2P_dxdy =', d2P_dxdy)
            if debug: print('Yt =', Yt)
            if debug: print('dYt_dx =', dYt_dx)
            if debug: print('dYt_dv =', dYt_dv)
            if debug: print('dYt_du =', dYt_du)
            if debug: print('dYt_dw =', dYt_dw)
            if debug: print('d2Yt_dvdx =', d2Yt_dvdx)
            if debug: print('d2Yt_dudx =', d2Yt_dudx)
            if debug: print('d2Yt_dwdx =', d2Yt_dwdx)
            if debug: print('d2Yt_dxdy =', d2Yt_dxdy)
            if debug: print('d3Yt_dvdxdy =', d3Yt_dvdxdy)
            if debug: print('d3Yt_dudxdy =', d3Yt_dudxdy)
            if debug: print('d3Yt_dwdxdy =', d3Yt_dwdxdy)

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
                G[i] = self.Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
                dG_dYt[i] = self.dG_dYf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
                for j in range(m):
                    dG_ddelYt[i,j] = self.dG_ddelYf[j](x[i], Yt[i],
                                                  dYt_dx[i], d2Yt_dxdy[i])
                    for jj in range(m):
                        dG_ddeldelYt[i,j,jj] = self.dG_ddeldelYf[j][jj](x[i], Yt[i],
                                                                   dYt_dx[i],
                                                                   d2Yt_dxdy[i])
                for k in range(H):
                    dG_dv[i,k] = dG_dYt[i]*dYt_dv[i,k]
                    dG_du[i,k] = dG_dYt[i]*dYt_du[i,k]
                    for j in range(m):
                        dG_dv[i,k] += dG_ddelYt[i,j]*d2Yt_dvdx[i,j,k]
                        dG_du[i,k] += dG_ddelYt[i,j]*d2Yt_dudx[i,j,k]
                        dG_dw[i,j,k] = dG_dYt[i]*dYt_dw[i,j,k]
                        for l in range(m):
                            dG_dw[i,j,k] += dG_ddelYt[i,l]*d2Yt_dwdx[i,j,l,k]
                            for ll in range(m):
                                dG_dw[i,j,k] += (
                                    dG_ddeldelYt[i,l,ll]*d3Yt_dwdxdy[i,j,l,ll,k]
                                )
                        for jj in range(m):
                            dG_dv[i,k] += dG_ddeldelYt[i,j,jj]*d3Yt_dvdxdy[i,j,jj,k]
                            dG_du[i,k] += dG_ddeldelYt[i,j,jj]*d3Yt_dudxdy[i,j,jj,k]
            if debug: print('G =', G)
            if debug: print('dG_dYt =', dG_dYt)
            if debug: print('dG_ddelYt =', dG_ddelYt)
            if debug: print('dG_ddeldelYt =', dG_ddeldelYt)
            if debug: print('dG_dv =', dG_dv)
            if debug: print('dG_du =', dG_du)
            if debug: print('dG_dw =', dG_dw)

            # Compute the error function for this pass.
            E = sum(G**2)
            if debug: print('E =', E)

            # Compute the partial derivatives of the error with
            # respect to the network parameters.
            dE_dv = np.zeros(H)
            dE_du = np.zeros(H)
            dE_dw = np.zeros((m, H))
            for k in range(H):
                for i in range(n):
                    dE_dv[k] += 2*G[i]*dG_dv[i,k]
                    dE_du[k] += 2*G[i]*dG_du[i,k]
                for j in range(m):
                    for i in range(n):
                        dE_dw[j,k] += 2*G[i]*dG_dw[i,j,k]
            if debug: print('dE_dv =', dE_dv)
            if debug: print('dE_du =', dE_du)
            if debug: print('dE_dw =', dE_dw)

            # Record the current RMSE.
            rmse = sqrt(E/n)
            rmse_history[epoch] = rmse
            if verbose: print(epoch, rmse)

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
        m = len(self.bcf)
        if debug: print('m =', m)
        H = len(self.w[0])
        if debug: print('H =', H)

        #------------------------------------------------------------------------

        # Create an array to hold the weights connecting the input
        # node to the hidden nodes. The weights are initialized with a
        # uniform random distribution.
        w = np.random.uniform(wmin, wmax, (m, H))
        if debug: print('w =', w)

        # Create an array to hold the biases for the hidden nodes. The
        # biases are initialized with a uniform random distribution.
        u = np.random.uniform(umin, umax, H)
        if debug: print('u =', u)

        # Create an array to hold the weights connecting the hidden
        # nodes to the output node. The weights are initialized with a
        # uniform random distribution.
        v = np.random.uniform(vmin, vmax, H)
        if debug: print('v =', v)

        #------------------------------------------------------------------------

        # Assemble the network parameters into a single 1-D vector for
        # use by the minimize() method.
        # p = [w, u, v]
        p = np.hstack((w[0], w[1], u, v))

        # Minimize the error function to get the new parameter values.
        if trainalg in ('Nelder-Mead', 'Powell', 'CG', 'BFGS'):
            res = minimize(self.computeError, p, method=trainalg,
                           args = (x),
                           options = {'maxiter': 20000, 'disp': True})
#                           callback=print_progress)
        elif trainalg in ('Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            res = minimize(self.computeError, p, method=trainalg,
                           jac=self.computeErrorGradient, args = (x))
        if debug: print(res)

        # Unpack the optimized network parameters.
        self.w[0] = res.x[0:H]
        self.w[1] = res.x[H:2*H]
        self.u = res.x[2*H:3*H]
        self.v = res.x[3*H:4*H]
        if debug: print('Final w =', self.w)
        if debug: print('Final u =', self.u)
        if debug: print('Final v =', self.v)

    def computeError(self, p, x):

        n = len(x)
        m = len(x[0])
        H = int(len(p)/4)

        w = np.zeros((m, H))
        w[0] = p[0:H]
        w[1] = p[H:2*H]
        u = p[2*H:3*H]
        v = p[3*H:4*H]

        z =  np.zeros((n, H))
        s =  np.zeros((n, H))
        s1 = np.zeros((n, H))
        s2 = np.zeros((n, H))
        # s3 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i,k] = u[k]
                for j in range(m):
                    z[i,k] += w[j,k]*x[i,j]
                s[i,k] = sigma(z[i,k])
                s1[i,k] = dsigma_dz(z[i,k])
                s2[i,k] = d2sigma_dz2(z[i,k])

        N = np.zeros(n)
        dN_dx = np.zeros((n, m))
        d2N_dxdy = np.zeros((n, m, m))
        for i in range(n):
            for k in range(H):
                N[i] += v[k]*s[i,k]
                for j in range(m):
                    dN_dx[i,j] += v[k]*s1[i,k]*w[j,k]
                    for jj in range(m):
                        d2N_dxdy[i,j,jj] += v[k]*s2[i,k]*w[j,k]*w[jj,k]

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
            Yt[i] = Ytf(x[i], N[i], self.bcf)
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                dP_dx[i,j] = delPf[j](x[i])
                dYt_dx[i,j] = dA_dx[i,j] + P[i]*dN_dx[i,j] + dP_dx[i,j]*N[i]
                for jj in range(m):
                    d2A_dxdy[i,j,jj] = deldelAf[j][jj](x[i], self.bcf, self.bcdf, self.bcd2f)
                    d2P_dxdy[i,j,jj] = deldelPf[j][jj](x[i])
                    d2Yt_dxdy[i,j,jj] = (
                        d2A_dxdy[i,j,jj] +
                        P[i]*d2N_dxdy[i,j,jj] +
                        dP_dx[i,j]*dN_dx[i,jj] +
                        dP_dx[i,jj]*dN_dx[i,j] +
                        d2P_dxdy[i,j,jj]*N[i]
                    )

        G = np.zeros(n)
        for i in range(n):
            G[i] =self.Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])

        E = np.sum(G**2)
        return E

    def computeErrorGradient(self, p, x):

        n = len(x)
        m = len(x[0])
        H = int(len(p)/4)

        w = np.zeros((m, H))
        w[0] = p[0:H]
        w[1] = p[H:2*H]
        u = p[2*H:3*H]
        v = p[3*H:4*H]

        z =  np.zeros((n, H))
        s =  np.zeros((n, H))
        s1 = np.zeros((n, H))
        s2 = np.zeros((n, H))
        s3 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i,k] = u[k]
                for j in range(m):
                    z[i,k] += w[j,k]*x[i,j]
                s[i,k] = sigma(z[i,k])
                s1[i,k] = dsigma_dz(z[i,k])
                s2[i,k] = d2sigma_dz2(z[i,k])
                s3[i,k] = d3sigma_dz3(z[i,k])

        N = np.zeros(n)
        dN_dx = np.zeros((n, m))
        dN_dv = np.zeros((n, H))
        dN_du = np.zeros((n, H))
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
                N[i] += v[k]*s[i,k]
                dN_dv[i,k] = s[i,k]
                dN_du[i,k] = v[k]*s1[i,k]
                for j in range(m):
                    dN_dx[i,j] += v[k]*s1[i,k]*w[j,k]
                    dN_dw[i,j,k] = v[k]*s1[i,k]*x[i,j]
                    d2N_dvdx[i,j,k] = s1[i,k]*w[j,k]
                    d2N_dudx[i,j,k] = v[k]*s2[i,k]*w[j,k]
                    for jj in range(m):
                        d2N_dxdy[i,j,jj] += v[k]*s2[i,k]*w[j,k]*w[jj,k]
                        d2N_dwdx[i,j,jj,k] = v[k]*(s1[i,k]*kdelta(j,jj) +
                                                   s2[i,k]*w[jj,k]*x[i,j])
                        d3N_dvdxdy[i,j,jj,k] = s2[i,k]*w[j,k]*w[jj,k]
                        d3N_dudxdy[i,j,jj,k] = v[k]*s3[i,k]*w[j,k]*w[jj,k]
                        for jjj in range(m):
                            d3N_dwdxdy[i,j,jj,jjj,k] = (
                                v[k]*s2[i,k]*(w[jj,k]*kdelta(j,jjj)
                                              + w[jjj,k]*kdelta(j,jj)) +
                                v[k]*s3[i,k]*w[jj,k]*w[jjj,k]*x[i,j]
                            )

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
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], self.bcf)
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                dP_dx[i,j] = delPf[j](x[i])
                dYt_dx[i,j] = dA_dx[i,j] + P[i]*dN_dx[i,j] + dP_dx[i,j]*N[i]
                for jj in range(m):
                    d2A_dxdy[i,j,jj] = deldelAf[j][jj](x[i], self.bcf, self.bcdf, self.bcd2f)
                    d2P_dxdy[i,j,jj] = deldelPf[j][jj](x[i])
                    d2Yt_dxdy[i,j,jj] = (
                        d2A_dxdy[i,j,jj] +
                        P[i]*d2N_dxdy[i,j,jj] +
                        dP_dx[i,j]*dN_dx[i,jj] +
                        dP_dx[i,jj]*dN_dx[i,j] +
                        d2P_dxdy[i,j,jj]*N[i]
                    )
            for k in range(H):
                dYt_dv[i,k] = P[i]*dN_dv[i,k]
                dYt_du[i,k] = P[i]*dN_du[i,k]
                for j in range(m):
                    dYt_dw[i,j,k] = P[i]*dN_dw[i,j,k]
                    d2Yt_dvdx[i,j,k] = (
                        P[i]*d2N_dvdx[i,j,k] + dP_dx[i,j]*dN_dv[i,k]
                    )
                    d2Yt_dudx[i,j,k] = (
                        P[i]*d2N_dudx[i,j,k] + dP_dx[i,j]*dN_du[i,k]
                    )
                    for jj in range(m):
                        d2Yt_dwdx[i,jj,j,k] = (
                            P[i]*d2N_dwdx[i,jj,j,k] + dP_dx[i,j]*dN_dw[i,j,k]
                        )
                        d3Yt_dvdxdy[i,j,jj,k] = (
                            P[i]*d3N_dvdxdy[i,j,jj,k] +
                            dP_dx[i,j]*d2N_dvdx[i,jj,k] +
                            dP_dx[i,jj]*d2N_dvdx[i,j,k] +
                            d2P_dxdy[i,j,jj]*dN_dv[i,k]
                        )
                        d3Yt_dudxdy[i,j,jj,k] = (
                            P[i]*d3N_dudxdy[i,j,jj,k] +
                            dP_dx[i,j]*d2N_dudx[i,jj,k] +
                            dP_dx[i,jj]*d2N_dudx[i,j,k] +
                            d2P_dxdy[i,j,jj]*dN_du[i,k]
                        )
                        for jjj in range(m):
                            d3Yt_dwdxdy[i,jjj,j,jj,k] = (
                                P[i]*d3N_dwdxdy[i,jjj,j,jj,k] +
                                dP_dx[i,j]*d2N_dwdx[i,jjj,jj,k] +
                                dP_dx[i,jj]*d2N_dwdx[i,jj,j,k] +
                                d2P_dxdy[i,j,jj]*dN_dw[i,jjj,k]
                            )

        G = np.zeros(n)
        dG_dYt = np.zeros(n)
        dG_ddelYt = np.zeros((n, m))
        dG_ddeldelYt = np.zeros((n, m, m))
        dG_dv = np.zeros((n, H))
        dG_du = np.zeros((n, H))
        dG_dw = np.zeros((n, m, H))
        for i in range(n):
            G[i] = self.Gf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
            dG_dYt[i] = self.dG_dYf(x[i], Yt[i], dYt_dx[i], d2Yt_dxdy[i])
            for j in range(m):
                dG_ddelYt[i,j] = self.dG_ddelYf[j](x[i], Yt[i],
                                              dYt_dx[i], d2Yt_dxdy[i])
                for jj in range(m):
                    dG_ddeldelYt[i,j,jj] = self.dG_ddeldelYf[j][jj](x[i], Yt[i],
                                                               dYt_dx[i],
                                                               d2Yt_dxdy[i])
            for k in range(H):
                dG_dv[i,k] = dG_dYt[i]*dYt_dv[i,k]
                dG_du[i,k] = dG_dYt[i]*dYt_du[i,k]
                for j in range(m):
                    dG_dv[i,k] += dG_ddelYt[i,j]*d2Yt_dvdx[i,j,k]
                    dG_du[i,k] += dG_ddelYt[i,j]*d2Yt_dudx[i,j,k]
                    dG_dw[i,j,k] = dG_dYt[i]*dYt_dw[i,j,k]
                    for l in range(m):
                        dG_dw[i,j,k] += dG_ddelYt[i,l]*d2Yt_dwdx[i,j,l,k]
                        for ll in range(m):
                            dG_dw[i,j,k] += (
                                dG_ddeldelYt[i,l,ll]*d3Yt_dwdxdy[i,j,l,ll,k]
                            )
                    for jj in range(m):
                        dG_dv[i,k] += dG_ddeldelYt[i,j,jj]*d3Yt_dvdxdy[i,j,jj,k]
                        dG_du[i,k] += dG_ddeldelYt[i,j,jj]*d3Yt_dudxdy[i,j,jj,k]

        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros((m, H))
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2*G[i]*dG_dv[i,k]
                dE_du[k] += 2*G[i]*dG_du[i,k]
            for j in range(m):
                for i in range(n):
                    dE_dw[j,k] += 2*G[i]*dG_dw[i,j,k]

        jac = np.hstack((dE_dw[0], dE_dw[1], dE_du, dE_dv))
        return jac

    def print_progress(xk):
        print('xk =', xk)

    def run(self, x):
        n = len(x)
        m = len(x[0])
        H = len(self.w[0])
        z = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i,k] = self.u[k]
                for j in range(m):
                    z[i,k] += self.w[j,k]*x[i,j]
        s = np.vectorize(sigma)(z)
        N = s.dot(self.v)
        Yt = np.zeros(n)
        for i in range(n):
            Yt[i] = Ytf(x[i], N[i], self.bcf)
        return Yt

    def run_derivative(self, x):
        n = len(x)
        m = len(x[0])
        H = len(self.w[0])

        z =  np.zeros((n, H))
        s =  np.zeros((n, H))
        s1 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i,k] = self.u[k]
                for j in range(m):
                    z[i,k] += self.w[j,k]*x[i,j]
                    s[i,k] = sigma(z[i,k])
                    s1[i,k] = dsigma_dz(z[i,k])

        N = np.zeros(n)
        dN_dx = np.zeros((n, m))
        for i in range(n):
            for k in range(H):
                N[i] += self.v[k]*s[i,k]
                for j in range(m):
                    dN_dx[i,j] += self.v[k]*s1[i,k]*self.w[j,k]

        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        dYt_dx = np.zeros((n, m))
        for i in range(n):
            P[i] = Pf(x[i])
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                dP_dx[i,j] = delPf[j](x[i])
                dYt_dx[i,j] = dA_dx[i,j] + P[i]*dN_dx[i,j] + dP_dx[i,j]*N[i]

        return dYt_dx

#--------------------------------------------------------------------------------

if __name__ == '__main__':

    # Create training data.
    xt = np.linspace(0, 1, 10)
    yt = np.linspace(0, 1, 10)
    x_train = np.array(list(zip(np.tile(xt, 10),np.repeat(yt, 10))))
    n = len(x_train)

    # Test each training algorithm on each equation.
    for pde in ('pde02bvp',):
        print('Examining %s.' % pde)
        pde2bvp = PDE2BVP(pde)
        print(pde2bvp)
        m = len(pde2bvp.bcdf)
        net = NNPDE2BVP(pde2bvp)
        print(net)
        Ya = np.zeros(n)
        for i in range(n):
            Ya[i] = net.Yaf(x_train[i])
        dYa_dx = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dYa_dx[i,j] = net.delYaf[j](x_train[i])
        d2Ya_dxdy = np.zeros((n, m, m))
        for i in range(n):
            for j in range(m):
                for jj in range(m):
                    d2Ya_dxdy[i, j, jj] = net.deldelYaf[j][jj](x_train[i])
        print('The analytical solution is:')
        print('Ya =', Ya)
        print('The analytical gradient is:')
        print('dYa_dx =', dYa_dx)
        print('The analytical Hessian is:')
        # print('d2Ya_dxdy =', d2Ya_dxdy)
        print()
        for trainalg in ('delta', 'Nelder-Mead', 'Powell', 'CG', 'BFGS',
                         'Newton-CG', 'L-BFGS-B', 'TNC', 'SLSQP'):
            print('Training using %s algorithm.' % trainalg)
            np.random.seed(0)
            try:
                net.train(x_train, trainalg=trainalg)
            except Exception as e:
                print('Error using %s algorithm on %s!' % (trainalg, pde))
                print(e)
                print()
                continue
            Yt = net.run(x_train)
            dYt_dx = net.run_derivative(x_train)
            # d2yt_dx2 = net.run_2nd_derivative(x_train)
        #     print('The optimized network parameters are:')
        #     print('w =', net.w)
        #     print('u =', net.u)
        #     print('v =', net.v)
            print('The trained solution is:')
            print('Yt =', Yt)
            print('The trained derivative is:')
            print('dYt_dx =', dYt_dx)
            print()

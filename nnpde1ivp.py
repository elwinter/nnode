# Notation notes:
# The suffix 'f' usually denotes an object that is a function.

# The suffix '_v' denotes a function that has been vectorized with
# np.vectorize().

from scipy.optimize import minimize
from math import sqrt
import numpy as np

from pde1ivp import PDE1IVP
from sigma import sigma, dsigma_dz, d2sigma_dz2
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

# Define the coefficient functions for the trial solution, and their
# derivatives.
def Af(xy, bcf):
    (x, y) = xy
    (f0f, g0f) = bcf
    A = (1 - x)*f0f(y) + (1 - y)*(g0f(x) - (1 - x)*g0f(0))
    return A

def dA_dxf(xy, bcf, bcdf):
    (x, y) = xy
    (f0f, g0f) = bcf
    (df0_dyf, dg0_dxf) = bcdf
    dA_dx = -f0f(y) + (1 - y)*(dg0_dxf(x) + g0f(0))
    return dA_dx

def dA_dyf(xy, bcf, bcdf):
    (x, y) = xy
    (f0f, g0f) = bcf
    (df0_dyf, dg0_dxf) = bcdf
    dA_dy = (1 - x)*df0_dyf(y) - g0f(x) + (1 - x)*g0f(0)
    return dA_dy

delAf = (dA_dxf, dA_dyf)

def Pf(xy):
    (x, y) = xy
    P = x*y
    return P

def dP_dxf(xy):
    (x, y) = xy
    dP_dx = y
    return dP_dx

def dP_dyf(xy):
    (x, y) = xy
    dP_dy = x
    return dP_dy

delPf = (dP_dxf, dP_dyf)

# # Define the trial solution.
def Ytf(xy, N, bcf):
    A = Af(xy, bcf)
    P = Pf(xy)
    Yt = A + P*N
    return Yt

# Vectorize sigma functions.
sigma_v = np.vectorize(sigma)
dsigma_dz_v = np.vectorize(dsigma_dz)
d2sigma_dz2_v = np.vectorize(d2sigma_dz2)

class NNPDE1IVP(SLFFNN):

    def __init__(self, pde1ivp, nhid=default_nhid):


        # PDE, in the form G(x,Y,delY)=0.
        self.Gf = pde1ivp.Gf

        # dG/dY
        self.dG_dYf = pde1ivp.dG_dYf

        # dG/d(delY)
        self.dG_ddelYf = pde1ivp.dG_ddelYf

        # Initial condition functions
        self.bcf = pde1ivp.bcf

        # Initial condition derivatives
        self.bcdf = pde1ivp.bcdf

        # Analytical solution (optional)
        if pde1ivp.Yaf:
            self.Yaf = pde1ivp.Yaf

        # Analytical gradient (optional)
        if pde1ivp.delYaf:
            self.delYaf = pde1ivp.delYaf

        # Create arrays to hold the weights and biases, initially all 0.
        self.w = np.zeros((2, nhid))
        self.u = np.zeros(nhid)
        self.v = np.zeros(nhid)

    def __str__(self):
        s = "NNPDE1IVP:\n"
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

            # Compute the input, the sigmoid function, and its
            # derivatives, for each hidden node k, for each training
            # point i. Each hidden node has 2 weights w[j,k] (since
            # there are 2 inputs x[i,j] per training sample) and 1
            # bias u[k].

            # Compute the input, the sigmoid function and its derivatives,
            # for each hidden node.
            z = np.broadcast_to(self.u, (n, H)) + np.dot(x, self.w)
            s = sigma_v(z)
            s1 = dsigma_dz_v(z)
            s2 = d2sigma_dz2_v(z)
            if debug: print('z =', z)
            if debug: print('s =', s)
            if debug: print('s1 =', s1)
            if debug: print('s2 =', s2)

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
            if debug: print('N =', N)
            if debug: print('dN_dx =', dN_dx)
            if debug: print('dN_dw =', dN_dw)
            if debug: print('dN_du =', dN_du)
            if debug: print('dN_dv =', dN_dv)
            if debug: print('d2N_dwdx =', d2N_dwdx)
            if debug: print('d2N_dudx =', d2N_dudx)
            if debug: print('d2N_dvdx =', d2N_dvdx)

            #--------------------------------------------------------------------

            # Compute the value of the trial solution and its derivatives,
            # for each training point.
            dA_dx = np.zeros((n, m))
            P = np.zeros(n)
            dP_dx = np.zeros((n, m))
            Yt = np.zeros(n)
            d2Yt_dudx = np.zeros((n, m, H))
            d2Yt_dvdx = np.zeros((n, m, H))
            for i in range(n):
                P[i] = Pf(x[i])
                Yt[i] = Ytf(x[i], N[i], self.bcf)
                for j in range(m):
                    dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                    dP_dx[i,j] = delPf[j](x[i])
                for k in range(H):
                    for j in range(m):
                        d2Yt_dvdx[i,j,k] = (
                            P[i]*d2N_dvdx[i,j,k] + dP_dx[i,j]*dN_dv[i,k]
                        )
                        d2Yt_dudx[i,j,k] = (
                            P[i]*d2N_dudx[i,j,k] + dP_dx[i,j]*dN_du[i,k]
                        )
                        P_b = np.tile(P, (2, 1)).T
                        N_b = np.tile(N, (2, 1)).T
                        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
                        dYt_dw = dN_dw*np.broadcast_to(P.T, (H, m, n)).T
                        dYt_du = dN_du*np.broadcast_to(P.T, (H, n)).T
                        dYt_dv = dN_dv*np.broadcast_to(P.T, (H, n)).T
                        d2Yt_dwdx = np.broadcast_to(P.T, (H, m, n)).T*d2N_dwdx +\
                                    np.broadcast_to(dP_dx.T, (H, m, n)).T*dN_dw
            if debug: print('dA_dx =', dA_dx)
            if debug: print('P =', P)
            if debug: print('dP_dx =', dP_dx)
            if debug: print('Yt =', Yt)
            if debug: print('dYt_dx =', dYt_dx)
            if debug: print('dYt_dw =', dYt_dw)
            if debug: print('dYt_du =', dYt_du)
            if debug: print('dYt_dv =', dYt_dv)
            if debug: print('d2Yt_dwdx =', d2Yt_dwdx)
            if debug: print('d2Yt_dudx =', d2Yt_dudx)
            if debug: print('d2Yt_dvdx =', d2Yt_dvdx)

            # Compute the value of the original differential equation
            # for each training point, and its derivatives.
            G = np.zeros(n)
            dG_dYt = np.zeros(n)
            dG_ddelYt = np.zeros((n, m))
            dG_dv = np.zeros((n, H))
            dG_du = np.zeros((n, H))
            dG_dw = np.zeros((n, m, H))
            for i in range(n):
                G[i] = self.Gf(x[i], Yt[i], dYt_dx[i])
                dG_dYt[i] = self.dG_dYf(x[i], Yt[i], dYt_dx[i])
                for j in range(m):
                    dG_ddelYt[i,j] = self.dG_ddelYf[j](x[i], Yt[i], dYt_dx[i])
                for k in range(H):
                    dG_dv[i,k] = dG_dYt[i]*dYt_dv[i,k]
                    dG_du[i,k] = dG_dYt[i]*dYt_du[i,k]
                    for j in range(m):
                        dG_dv[i,k] += dG_ddelYt[i,j]*d2Yt_dvdx[i,j,k]
                        dG_du[i,k] += dG_ddelYt[i,j]*d2Yt_dudx[i,j,k]
                        dG_dw[i,j,k] = (
                            dG_dYt[i]*dYt_dw[i,j,k] +
                            dG_ddelYt[i,j]*d2Yt_dwdx[i,j,k]
                        )
            if debug: print('G =', G)
            if debug: print('dG_dYt =', dG_dYt)
            if debug: print('dG_ddelYt =', dG_ddelYt)
            if debug: print('dG_dv =', dG_dv)
            if debug: print('dG_du =', dG_du)
            if debug: print('dG_dw =', dG_dw)

            # Compute the error function for this epoch.
            E = np.sum(G**2)
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

            #--------------------------------------------------------------------

            # Record the current RMSE.
            rmse = sqrt(E/n)
            rmse_history[epoch] = rmse
            if verbose: print(epoch, rmse)

            # Save the error and parameter history.
            np.savetxt(rmseout, rmse_history)
            np.savetxt('w0.dat', w_history[:,0,:])
            np.savetxt('w1.dat', w_history[:,1,:])
            np.savetxt('u.dat', u_history)
            np.savetxt('v.dat', v_history)

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

        debug = True

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

        z = np.broadcast_to(u,(n,H)) + np.dot(x,w)
        s = np.vectorize(sigma)(z)
        s1 = np.vectorize(dsigma_dz)(z)
        s2 = np.vectorize(d2sigma_dz2)(z)

        w_b = np.broadcast_to(w, (n, m, H))
        v_b = np.broadcast_to(np.broadcast_to(v,(m,H)),(n,m,H))
        s1_b = np.broadcast_to(np.expand_dims(s1,axis=1),(n,m,H))

        N = s.dot(v)
        dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)

        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        Yt = np.zeros(n)
        for i in range(n):
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], self.bcf)
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                dP_dx[i,j] = delPf[j](x[i])
        P_b = np.tile(P, (2, 1)).T
        N_b = np.tile(N, (2, 1)).T
        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx

        G = np.zeros(n)
        for i in range(n):
            G[i] = self.Gf(x[i], Yt[i], dYt_dx[i])

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

        z = np.broadcast_to(u,(n,H)) + np.dot(x,w)
        s = np.vectorize(sigma)(z)
        s1 = np.vectorize(dsigma_dz)(z)
        s2 = np.vectorize(d2sigma_dz2)(z)

        x_b = np.broadcast_to(x.T, (H,m,n)).T
        w_b = np.broadcast_to(w, (n, m, H))
        v_b = np.broadcast_to(np.broadcast_to(v,(m,H)),(n,m,H))
        s1_b = np.broadcast_to(np.expand_dims(s1,axis=1),(n,m,H))
        s2_b = np.broadcast_to(np.expand_dims(s2,axis=1),(n,m,H))
        
        N = s.dot(v)
        dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)
        dN_dw = v_b*s1_b*x_b
        dN_du = s1*v
        dN_dv = s
        d2N_dwdx = v_b*s1_b + v_b*s2_b*w_b*x_b
        d2N_dudx = v_b*s2_b*w_b
        d2N_dvdx = s1_b*w_b

        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        Yt = np.zeros(n)
        d2Yt_dudx = np.zeros((n, m, H))
        d2Yt_dvdx = np.zeros((n, m, H))
        for i in range(n):
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], self.bcf)
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                dP_dx[i,j] = delPf[j](x[i])
            for k in range(H):
                for j in range(m):
                    d2Yt_dudx[i,j,k] = (
                        P[i]*d2N_dudx[i,j,k] + dP_dx[i,j]*dN_du[i,k]
                    )
                    d2Yt_dvdx[i,j,k] = (
                        P[i]*d2N_dvdx[i,j,k] + dP_dx[i,j]*dN_dv[i,k]
                    )
        P_b = np.tile(P, (2, 1)).T
        N_b = np.tile(N, (2, 1)).T
        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
        dYt_dw = dN_dw*np.broadcast_to(P.T, (H, m, n)).T
        dYt_du = dN_du*np.broadcast_to(P.T, (H, n)).T
        dYt_dv = dN_dv*np.broadcast_to(P.T, (H, n)).T
        d2Yt_dwdx = np.broadcast_to(P.T, (H, m, n)).T*d2N_dwdx + \
                    np.broadcast_to(dP_dx.T, (H, m, n)).T*dN_dw

        G = np.zeros(n)
        dG_dYt = np.zeros(n)
        dG_ddelYt = np.zeros((n, m))
        dG_dw = np.zeros((n, m, H))
        dG_du = np.zeros((n, H))
        dG_dv = np.zeros((n, H))
        for i in range(n):
            G[i] = self.Gf(x[i], Yt[i], dYt_dx[i])
            dG_dYt[i] = self.dG_dYf(x[i], Yt[i], dYt_dx[i])
            for j in range(m):
                dG_ddelYt[i,j] = self.dG_ddelYf[j](x[i], Yt[i], dYt_dx[i])
            for k in range(H):
                dG_dv[i,k] = dG_dYt[i]*dYt_dv[i,k]
                dG_du[i,k] = dG_dYt[i]*dYt_du[i,k]
                for j in range(m):
                    dG_dw[i,j,k] = (
                        dG_dYt[i]*dYt_dw[i,j,k] +
                        dG_ddelYt[i,j]*d2Yt_dwdx[i,j,k]
                    )
                    dG_du[i,k] += dG_ddelYt[i,j]*d2Yt_dudx[i,j,k]
                    dG_dv[i,k] += dG_ddelYt[i,j]*d2Yt_dvdx[i,j,k]

        dE_dw = np.zeros((m, H))
        dE_du = np.zeros(H)
        dE_dv = np.zeros(H)
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
        z = np.broadcast_to(self.u, (n, H)) + np.dot(x, self.w)
        s = np.vectorize(sigma)(z)
        N = s.dot(self.v)
        P = np.zeros(n)
        Yt = np.zeros(n)
        for i in range(n):
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], self.bcf)
        return Yt

    def run_derivative(self, x):
        n = len(x)
        m = len(x[0])
        H = len(self.w[0])

        z = np.broadcast_to(self.u,(n,H)) + np.dot(x,self.w)
        s = np.vectorize(sigma)(z)
        s1 = np.vectorize(dsigma_dz)(z)
        s2 = np.vectorize(d2sigma_dz2)(z)

        x_b = np.broadcast_to(x.T, (H,m,n)).T
        w_b = np.broadcast_to(self.w, (n, m, H))
        v_b = np.broadcast_to(np.broadcast_to(self.v,(m,H)),(n,m,H))
        s1_b = np.broadcast_to(np.expand_dims(s1,axis=1),(n,m,H))
        
        N = s.dot(self.v)
        dN_dx = np.sum(v_b*s1_b*w_b, axis = 2)

        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        for i in range(n):
            P[i] = Pf(x[i])
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], self.bcf, self.bcdf)
                dP_dx[i,j] = delPf[j](x[i])
        P_b = np.tile(P, (2, 1)).T
        N_b = np.tile(N, (2, 1)).T
        dYt_dx = dA_dx + dN_dx*P_b + N_b*dP_dx
        return dYt_dx

#--------------------------------------------------------------------------------

if __name__ == '__main__':

    # Create training data.
    xt = np.linspace(0, 1, 10)
    yt = np.linspace(0, 1, 10)
    x_train = np.array(list(zip(np.tile(xt, 10),np.repeat(yt, 10))))
    n = len(x_train)

    # Test each training algorithm on each equation.
    for ode in ('pde00', 'pde01',):
        print('Examining %s.' % ode)
        pde1ivp = PDE1IVP(ode)
        m = len(pde1ivp.bcdf)
        net = NNPDE1IVP(pde1ivp)
        Ya = np.zeros(n)
        for i in range(n):
            Ya[i] = net.Yaf(x_train[i])
        dYa_dx = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dYa_dx[i,j] = net.delYaf[j](x_train[i])
        print('The analytical solution is:')
        print('Ya =', Ya)
        print('The analytical derivative is:')
        print('dYa_dx =', dYa_dx)
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
            Yt = net.run(x_train)
            dYt_dx = net.run_derivative(x_train)
            # d2yt_dx2 = net.run_2nd_derivative(x_train)
            print('The optimized network parameters are:')
            print('w =', net.w)
            print('u =', net.u)
            print('v =', net.v)
            print('The trained solution is:')
            print('Yt =', Yt)
            print('The trained derivative is:')
            print('dYt_dx =', dYt_dx)
            print()

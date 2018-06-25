#!/usr/bin/env python

# Use a neural network to solve a 1-D diffusion problem:

# dY/dt = -D del2(Y)

# Notation notes:

# 1. Names that end in 'f' are usually functions, or containers of functions.

# 2. Underscores separate the numerator and denominator in a name
# which represents a derivative.

# 3. Names beginning with 'del' are gradients of another function, in
# the form of function lists. Similarly for del2.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
from importlib import import_module
from math import sqrt
import numpy as np
from kdelta import kdelta
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3

#********************************************************************************

# Default values for program parameters
default_clamp = False
default_debug = False
default_eta = 0.01
default_lambdamom = 0
default_maxepochs = 1000
default_nhid = 10
default_ntest = 10
default_ntrain = 10
default_pde = 'diffusion1d'
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

#********************************************************************************

# The domain of the trial solution is assumed to be [[0,1],[0,1]].

# Define the coefficient functions for the trial solution, and their derivatives.
def Af(xt, bcf):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    return f0f(t)*(1 - x)*t + f1f(t)*x*t + (1 - t)*g0f(x)

def dA_dxf(xt, bcf, bcdf):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf)) = bcdf
    return (f1f(t) - f0f(t))*t + (1 - t)*dg0_dxf(x)

def dA_dtf(xt, bcf, bcdf):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf)) = bcdf
    return (1 - x)*(f0f(t) + df0_dtf(t)*t) + x*(f1f(t) + df1_dtf(t)*t) - g0f(x)

delAf = (dA_dxf, dA_dtf)

def d2A_dxdxf(xt, bcf, bcdf, bcd2f):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf)) = bcdf
    ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f)) = bcd2f
    return (1 - t)*d2g0_dx2f(x)

def d2A_dxdtf(xt, bcf, bcdf, bcd2f):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf)) = bcdf
    ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f)) = bcd2f
    return -f0f(t) - df0_dtf(t) + f1f(t) + df1_dtf(t) - dg0_dxf(x)

def d2A_dtdxf(xt, bcf, bcdf, bcd2f):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf)) = bcdf
    ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f)) = bcd2f
    return -f0f(t) - df0_dtf(t) + f1f(t) + df1_dtf(t) - dg0_dxf(x)

def d2A_dtdtf(xt, bcf, bcdf, bcd2f):
    (x, t) = xt
    ((f0f, g0f), (f1f, g1f)) = bcf
    ((df0_dtf, dg0_dxf), (df1_dtf, dg1_dxf)) = bcdf
    ((d2f0_dt2f, d2g0_dx2f), (d2f1_dt2f, d2g1_dx2f)) = bcd2f
    return (
        (1 - x)*(2*df0_dtf(t) + d2f0_dt2f(t)*t) +
        x*(2*df1_dtf(t) + d2f1_dt2f(t)*t)
    )

deldelAf = ((d2A_dxdxf, d2A_dxdtf),
            (d2A_dtdxf, d2A_dtdtf))

def Pf(xt):
    (x, t) = xt
    return x*(1 - x)*t

def dP_dxf(xt):
    (x, t) = xt
    return (1 - 2*x)*t

def dP_dtf(xt):
    (x, t) = xt
    return x*(1 - x)

delPf = (dP_dxf, dP_dtf)

def d2P_dxdxf(xt):
    (x, t) = xt
    return -2*t

def d2P_dxdtf(xt):
    (x, t) = xt
    return 1 - 2*x

def d2P_dtdxf(xt):
    (x, t) = xt
    return 1 - 2*x

def d2P_dtdtf(xt):
    (x, t) = xt
    return 0

deldelPf = ((d2P_dxdxf, d2P_dxdtf),
            (d2P_dtdxf, d2P_dtdtf))

# Define the trial solution.
def Ytf(xt, N, bcf):
    return Af(xt, bcf) + Pf(xt)*N

#********************************************************************************

# Function to solve a 2-variable, 2nd-order PDE BVP with Dirichlet BC,
# using a single-hidden-layer feedforward neural network with 2 input
# nodes and a single output node.

def nnpde2bvp(
        Gf,                            # 2-variable, 2nd-order PDE BVP
        dG_dYf,                        # Partial of G wrt Y
        dG_ddelYf,                     # Partials of G wrt del Y
        dG_ddeldelYf,                  # Partials of G wrt del del Y
        bcf,                           # BC functions
        bcdf,                          # BC function derivatives
        bcd2f,                         # BC function 2nd derivatives
        x,                             # Training points as pairs
        nhid = default_nhid,           # Node count in hidden layer
        maxepochs = default_maxepochs, # Max training epochs
        eta = default_eta,             # Learning rate
        clamp = default_clamp,         # Turn on/off parameter clamping
        randomize = default_randomize, # Randomize training sample order
        vmax = default_vmax,           # Maximum initial output weight value
        vmin = default_vmin,           # Minimum initial output weight value
        wmax = default_wmax,           # Maximum initial hidden weight value
        wmin = default_wmin,           # Minimum initial hidden weight value
        umax = default_umax,           # Maximum initial hidden bias value
        umin = default_umin,           # Minimum initial hidden bias value
        lambdamom = default_lambdamom, # Momentum coefficient
        rmseout = default_rmseout,     # Output file for ODE RMS error
        debug = default_debug,
        verbose = default_verbose
):
    if debug: print('Gf =', Gf)
    if debug: print('dG_dYf =', dG_dYf)
    if debug: print('dG_ddelYf =', dG_ddelYf)
    if debug: print('dG_ddeldelYf =', dG_ddeldelYf)
    if debug: print('bcf =', bcf)
    if debug: print('bcdf =', bcdf)
    if debug: print('bcd2f =', bcd2f)
    if debug: print('x =', x)
    if debug: print('nhid =', nhid)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('clamp =', clamp)
    if debug: print('randomize =', randomize)
    if debug: print('vmin =', vmin)
    if debug: print('vmax =', vmax)
    if debug: print('wmin =', wmin)
    if debug: print('wmax =', wmax)
    if debug: print('umin =', umin)
    if debug: print('umax =', umax)
    if debug: print('lambdamom =', lambdamom)
    if debug: print('rmseout =', rmseout)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments. UPDATE THESE CHECKS!
    assert Gf
    assert dG_dYf
    ndim = len(dG_ddelYf)
    assert ndim == 2
    assert len(dG_ddeldelYf) == ndim
    for j in range(ndim):
        assert len(dG_ddeldelYf[j]) == ndim
    assert len(bcf) == ndim
    for j in range(ndim):
        assert len(bcf[j]) == ndim
    assert len(bcdf) == ndim
    for j in range(ndim):
        assert len(bcdf[j]) == ndim
    assert len(bcd2f) == ndim
    for j in range(ndim):
        assert len(bcd2f[j]) == ndim
    assert nhid > 0
    assert maxepochs > 0
    assert eta > 0
    assert lambdamom >= 0
    assert rmseout
    assert vmin < vmax
    assert wmin < wmax
    assert umin < umax

    #----------------------------------------------------------------------------

    # Determine the number of training points.
    n = len(x)
    if debug: print('n =', n)

    # Change notation for convenience.
    m = ndim
    if debug: print('m =', m)  # Will always be 2 in this code.
    H = nhid
    if debug: print('H =', H)

    #----------------------------------------------------------------------------

    # Create the network.

    # Create an array to hold the weights connecting the 2 input nodes
    # to the hidden nodes. The weights are initialized with a uniform
    # random distribution.
    w = np.random.uniform(wmin, wmax, (m, H))
    if debug: print('w =', w)

    # Create an array to hold the biases for the hidden nodes. The
    # biases are initialized with a uniform random distribution.
    u = np.random.uniform(umin, umax, H)
    if debug: print('u =', u)

    # Create an array to hold the weights connecting the hidden nodes
    # to the output node. The weights are initialized with a uniform
    # random distribution.
    v = np.random.uniform(vmin, vmax, H)
    if debug: print('v =', v)

    # Create arrays to hold RMSE and parameter history.
    rmse_history = np.zeros(maxepochs)
    w_history = np.zeros((maxepochs, m, H))
    u_history = np.zeros((maxepochs, H))
    v_history = np.zeros((maxepochs, H))

    #----------------------------------------------------------------------------

    # Vectorize the functions used by the network.
    sigma_v = np.vectorize(sigma)
    dsigma_dz_v = np.vectorize(dsigma_dz)
    d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
    d3sigma_dz3_v = np.vectorize(d3sigma_dz3)

    # Initialize the momentum terms.
    delta_v_old = np.zeros(H)
    delta_w_old = np.zeros((m, H))
    delta_u_old = np.zeros(H)

    # 0 <= i < n (n = # of sample points)
    # 0 <= j,jj,jjj < m (m = # independent variables)
    # 0 <= k < H (H = # hidden nodes)

    # Run the network.
    for epoch in range(maxepochs):
        if debug: print('Starting epoch %d.' % epoch)

        # Save the current parameter values in the history.
        w_history[epoch] = w
        u_history[epoch] = u
        v_history[epoch] = v

        # If the randomize flag is set, shuffle the order of the training points.
        if randomize:
            if debug: print('Randomizing training sample order.')
            np.random.shuffle(x)

        # Compute the net input, the sigmoid function and its derivatives,
        # for each hidden node and each training point.
        z = u + x.dot(w)
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        s3 = d3sigma_dz3_v(z)
        if debug: print('z =', z)
        if debug: print('s =', s)
        if debug: print('s1 =', s1)
        if debug: print('s2 =', s2)
        if debug: print('s3 =', s3)

        #------------------------------------------------------------------------

        # Compute the network output and its derivatives, for each
        # training point.
        # N = np.zeros(n)
        N = s.dot(v)
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
                # N[i] += v[k]*s[i,k]
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

        #------------------------------------------------------------------------

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

        #------------------------------------------------------------------------

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
                dG_ddelYt[i,j] = dG_ddelYf[j](x[i], Yt[i],
                                              dYt_dx[i], d2Yt_dxdy[i])
                for jj in range(m):
                    dG_ddeldelYt[i,j,jj] = dG_ddeldelYf[j][jj](x[i], Yt[i],
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

        #------------------------------------------------------------------------

        # Compute the error function for this pass.
        E = sum(G**2)
        if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to the
        # network parameters.
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

        #------------------------------------------------------------------------

        # Compute the current deltas.
        delta_v = -eta*dE_dv + lambdamom*delta_v_old
        delta_u = -eta*dE_du + lambdamom*delta_u_old
        delta_w = -eta*dE_dw + lambdamom*delta_w_old
        if debug: print('delta_v =', delta_v)
        if debug: print('delta_u =', delta_u)
        if debug: print('delta_w =', delta_w)
    
        # Compute the new values of the network parameters.
        v_new = v + delta_v
        u_new = u + delta_u
        w_new = w + delta_w
        if debug: print('v_new =', v_new)
        if debug: print('u_new =', u_new)
        if debug: print('w_new =', w_new)

        # Save the current deltas.
        delta_v_old = delta_v
        delta_u_old = delta_u
        delta_w_old = delta_w

        # Clamp the values at +/-1.
        if clamp:
            w_new[w_new < wmin] = wmin
            w_new[w_new > wmax] = wmax
            u_new[u_new < umin] = umin
            u_new[u_new > umax] = umax
            v_new[v_new < vmin] = vmin
            v_new[v_new > vmax] = vmax

        # Compute the RMS error in the differential equation.
        rmse = sqrt(E/n)
        rmse_history[epoch] = rmse
        if verbose: print(epoch, rmse)

        # Save the new weights and biases.
        v = v_new
        u = u_new
        w = w_new

    # Save the error and parameter history.
    np.savetxt(rmseout, rmse_history, fmt = '%.6E', header = 'rmse')
    np.savetxt('v.dat', v_history, fmt = '%.6E', header = 'v')
    np.savetxt('w0.dat', w_history[:,0,:], fmt = '%.6E', header = 'w0')
    np.savetxt('w1.dat', w_history[:,1,:], fmt = '%.6E', header = 'w1')
    np.savetxt('u.dat', u_history, fmt = '%.6E', header = 'u')

    # Return the final solution.
    return (Yt, dYt_dx, d2Yt_dxdy, v, w, u)

#--------------------------------------------------------------------------------

def create_argument_parser():

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description = 'Solve a 1-D diffusion problem with a neural net',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = 'Experiment with the settings to find what works.'
    )
    assert parser

    # Add command line arguments.
    parser.add_argument('--clamp', '-c',
                        action = 'store_true',
                        default = default_clamp,
                        help = 'Clamp parameter values at limits.')
    parser.add_argument('--debug', '-d',
                        action = 'store_true',
                        default = default_debug,
                        help = 'Produce debugging output')
    parser.add_argument('--eta', type = float,
                        default = default_eta,
                        help = 'Learning rate for parameter adjustment')
    parser.add_argument('--lambdamom', type = float,
                        default = default_lambdamom,
                        help = 'Momentum coefficient for parameter adjustment')
    parser.add_argument('--maxepochs', type = int,
                        default = default_maxepochs,
                        help = 'Maximum number of training epochs')
    parser.add_argument('--nhid', type = int,
                        default = default_nhid,
                        help = 'Number of hidden-layer nodes to use')
    parser.add_argument('--ntest', type = int,
                        default = default_ntest,
                        help = 'Number of evenly-spaced test points to use in (x,t)')
    parser.add_argument('--ntrain', type = int,
                        default = default_ntrain,
                        help = 'Number of evenly-spaced training points to use in (x,t)')
    parser.add_argument('--pde', type = str,
                        default = default_pde,
                        help = 'Name of module containing PDE to solve')
    parser.add_argument('--randomize', '-r',
                        action = 'store_true',
                        default = default_randomize,
                        help = 'Randomize training sample order')
    parser.add_argument('--rmseout', type = str,
                        default = default_rmseout,
                        help = 'Name of file to hold ODE RMS error')
    parser.add_argument('--seed', type = int,
                        default = default_seed,
                        help = 'Random number generator seed')
    parser.add_argument('--testout', type = str,
                        default = default_testout,
                        help = 'Name of file to hold results at test points')
    parser.add_argument('--trainout', type = str,
                        default = default_trainout,
                        help = 'Name of file to hold results at training points')
    parser.add_argument('--umax', type = float,
                        default = default_umax,
                        help = 'Maximum initial hidden bias value')
    parser.add_argument('--umin', type = float,
                        default = default_umin,
                        help = 'Minimum initial hidden bias value')
    parser.add_argument('--verbose', '-v',
                        action = 'store_true',
                        default = default_verbose,
                        help = 'Produce verbose output')
    parser.add_argument('--version',
                        action = 'version',
                        version = '%(prog)s 0.0')
    parser.add_argument('--vmax', type = float,
                        default = default_vmax,
                        help = 'Maximum initial output weight value')
    parser.add_argument('--vmin', type = float,
                        default = default_vmin,
                        help = 'Minimum initial output weight value')
    parser.add_argument('--wmax', type = float,
                        default = default_wmax,
                        help = 'Maximum initial hidden weight value')
    parser.add_argument('--wmin', type = float,
                        default = default_wmin,
                        help = 'Minimum initial hidden weight value')

    # Return the argument parser.
    return parser

#--------------------------------------------------------------------------------

# Begin main program.

if __name__ == '__main__':

    # Create the command line parser.
    parser = create_argument_parser()
    assert parser

    # Process the command line.
    args = parser.parse_args()
    if args.debug: print('args =', args)

    # Extract the processed options.
    clamp = args.clamp
    debug = args.debug
    eta = args.eta
    lambdamom = args.lambdamom
    maxepochs = args.maxepochs
    nhid = args.nhid
    ntest = args.ntest
    ntrain = args.ntrain
    pde = args.pde
    randomize = args.randomize
    rmseout = args.rmseout
    seed = args.seed
    testout = args.testout
    trainout = args.trainout
    umax = args.umax
    umin = args.umin
    verbose = args.verbose
    vmax = args.vmax
    vmin = args.vmin
    wmax = args.wmax
    wmin = args.wmin
    if debug: print('clamp =', clamp)
    if debug: print('debug =', debug)
    if debug: print('eta =', eta)
    if debug: print('lambdamom =', lambdamom)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('nhid =', nhid)
    if debug: print('ntest =', ntest)
    if debug: print('ntrain =', ntrain)
    if debug: print('pde =', pde)
    if debug: print('randomize =', randomize)
    if debug: print('rmseout =', rmseout)
    if debug: print('seed =', seed)
    if debug: print('testout =', testout)
    if debug: print('trainout =', trainout)
    if debug: print('umax =', umax)
    if debug: print('umin =', umin)
    if debug: print('verbose =', verbose)
    if debug: print('vmax =', vmax)
    if debug: print('vmin =', vmin)
    if debug: print('wmax =', wmax)
    if debug: print('wmin =', wmin)

    # Perform basic sanity checks on the command-line options.
    assert eta > 0
    assert lambdamom >= 0
    assert maxepochs > 0
    assert nhid > 0
    assert ntest > 0
    assert ntrain > 0
    assert pde
    assert rmseout
    assert seed >= 0
    assert testout
    assert trainout
    assert umin < umax
    assert vmin < vmax
    assert wmin < wmax

    #----------------------------------------------------------------------------

    # Initialize the random number generator to ensure repeatable results.
    if verbose:
        print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the specified PDE module.
    if verbose:
        print('Importing PDE module %s.' % pde)
    pdemod = import_module(pde)

    # Validate the module contents.
    assert pdemod.Gf
    assert pdemod.dG_dYf
    ndim = len(pdemod.dG_ddelYf)
    assert ndim == 2
    assert len(pdemod.dG_ddeldelYf) == ndim
    for j in range(ndim):
        assert len(pdemod.dG_ddeldelYf[j]) == ndim
    assert len(pdemod.bcf) == ndim
    for j in range(ndim):
        assert len(pdemod.bcf[j]) == ndim
    assert len(pdemod.bcdf) == ndim
    for j in range(ndim):
        assert len(pdemod.bcdf[j]) == ndim
    assert len(pdemod.bcd2f) == ndim
    for j in range(ndim):
        assert len(pdemod.bcd2f[j]) == ndim
    assert pdemod.Yaf

    # FIND A BETTER WAY TO MAKE THIS SET OF POINTS.

    # Create the array of evenly-spaced training points. Use the same
    # values of the training points for each dimension.
    if verbose: print('Computing training points in [[0,1],[0,1]].')
    xt = np.linspace(0, 1, ntrain)
    if debug: print('xt =', xt)
    yt = xt
    if debug: print('yt =', yt)

    # Compute the total number of training points, replacing ntrain.
    nxt = len(xt)
    if debug: print('nxt =', nxt)
    nyt = len(yt)
    if debug: print('nyt =', nyt)
    ntrain = nxt*nyt
    if debug: print('ntrain =', ntrain)

    # Create the list of training points.
    #((x0,y0),(x1,y0),(x2,y0),...
    # (x0,y1),(x1,y1),(x2,y1),...
    x_train = np.zeros((ntrain, ndim))
    for j in range(nyt):
        for i in range(nxt):
            k = j*nxt + i
            x_train[k,0] = xt[i]
            x_train[k,1] = yt[j]
    if debug: print('x_train =', x_train)

    #----------------------------------------------------------------------------

    # Compute the 2nd-order PDE solution using the neural network.
    (Yt_train, delYt_train, deldelYt_train, v, w, u) = nnpde2bvp(
        pdemod.Gf,             # 2-variable, 2nd-order PDE BVP to solve
        pdemod.dG_dYf,         # Partial of G wrt Y
        pdemod.dG_ddelYf,      # Partials of G wrt del Y (wrt gradient)
        pdemod.dG_ddeldelYf,   # Partials of G wrt del del Y (wrt Hessian)
        pdemod.bcf,            # BC functions
        pdemod.bcdf,           # BC function derivatives
        pdemod.bcd2f,          # BC function 2nd derivatives
        x_train,               # Training points as pairs
        nhid = nhid,           # Node count in hidden layer
        maxepochs = maxepochs, # Max training epochs
        eta = eta,             # Learning rate
        clamp = clamp,         # Turn on/off parameter clamping
        randomize = randomize, # Randomize training sample order
        vmax = vmax,           # Maximum initial output weight value
        vmin = vmin,           # Minimum initial output weight value
        wmax = wmax,           # Maximum initial hidden weight value
        wmin = wmin,           # Minimum initial hidden weight value
        umax = umax,           # Maximum initial hidden bias value
        umin = umin,           # Minimum initial hidden bias value
        lambdamom = lambdamom, # Momentum coefficient
        rmseout = rmseout,     # Output file for ODE RMS error
        debug = debug,
        verbose = verbose
    )

    #----------------------------------------------------------------------------

    # Compute the analytical solution at the training points.
    Ya_train = np.zeros(ntrain)
    for i in range(ntrain):
        Ya_train[i] = pdemod.Yaf(x_train[i])
    if debug: print('Ya_train =', Ya_train)

    # Compute the analytical gradient at the training points.
    # delYa = np.zeros((ntrain, len(x[1])))
    # for i in range(ntrain):
    #     for j in range(ndim):
    #         delYa[i,j] = pdemod.delYaf[j](x[i])
    # if debug: print('delYa =', delYa)

    # Compute the analytical Hessian at the training points.
    # deldelYa = np.zeros((ntrain, len(x[1]), len(x[1])))
    # for i in range(ntrain):
    #     for j in range(ndim):
    #         for jj in range(ndim):
    #             deldelYa[i,j, jj] = pdemod.deldelYaf[j][jj](x[i])
    # if debug: print('deldelYa =', deldelYa)

    # Compute the RMSE of the trial solution.
    Yerr = Yt_train - Ya_train
    if debug: print('Yerr =', Yerr)
    rmseY = sqrt(sum(Yerr**2)/ntrain)
    if debug: print('rmseY =', rmseY)

    # Compute the RMSEs of the gradient.
    # delYerr = delYt - delYa
    # if debug: print('delYerr =', delYerr)
    # rmsedelY = np.zeros(ndim)
    # e2sum = np.zeros(ndim)
    # for j in range(ndim):
    #     for i in range(ntrain):
    #         e2sum[j] += delYerr[i,j]**2
    #     rmsedelY[j] = sqrt(e2sum[j]/ntrain)
    # if debug: print('rmsedelY =', rmsedelY)

    # Compute the RMSEs of the Hessian.
    # deldelYerr = deldelYt - deldelYa
    # if debug: print('deldelYerr =', deldelYerr)
    # rmsedeldelY = np.zeros((ndim, ndim))
    # e2sum = np.zeros((ndim, ndim))
    # for j in range(ndim):
    #     for jj in range(ndim):
    #         for i in range(ntrain):
    #             e2sum[j,jj] += deldelYerr[i,j,jj]**2
    #         rmsedeldelY[j,jj] = sqrt(e2sum[j,jj]/ntrain)
    # if debug: print('rmsedeldelY =', rmsedeldelY)

    # Print the report.
    # print('    x        t       Yt_train      Ya')
    # for i in range(ntrain)):
    #     print('%.6f %.6f %.6f %.6f' % (x[i,0], x[i,1], Yt_train[i], Ya[i]))
    # print(rmseY)
    np.savetxt(trainout,
               list(zip(x_train[:,0], x_train[:,1], Yt_train, Ya_train)),
               header = 'x_train t_train Yt_train Ya_train',
               fmt = '%.6E'
    )

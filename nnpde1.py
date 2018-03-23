#!/usr/bin/env python

# Use a neural network to solve a 2-variable, 1st-order PDE IVP. Note
# that any 2-variable 1st-order PDE BVP can be mapped to a
# corresponding IVP with initial values at 0, so this is the only
# solution form needed.

# The general form of such equations is:

# G(x[], psi, del_psi[]) = 0

# Notation notes:

# 0. Notation is developed to mirror my derivations and notes.

# 1. Names that end in 'f' are usually functions, or containers of functions.

# 2. Underscores separate the numerator and denominator in a name
# which represents a derivative.

# 3. del_psit[i,j] is the derivative of the trial solution psit[i] wrt
# x[i][j].

# 4. Names beginning with 'del_' are gradients of another function, in
# the form of function lists.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
from importlib import import_module
from math import sqrt
import numpy as np
from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3

#********************************************************************************

# Default values for program parameters
default_debug = False
default_eta = 0.01
default_maxepochs = 1000
default_nhid = 10
default_ntrain = 10
default_pde = 'pde00'
default_seed = 0
default_verbose = False

# Default ranges for weights and biases
w_min = -1
w_max = 1
u_min = -1
u_max = 1
v_min = -1
v_max = 1

#********************************************************************************

# The range of the trial solution is assumed to be [0, 1].
# N.B. ASSUMES ONLY 2 DIMENSIONS!

# Define the coefficient functions for the trial solution, and their derivatives.
def Af(xy, bcf):
    (x, y) = xy
    (f0f, g0f) = bcf
    A = (1 - x) * f0f(y) + (1 - y) * (g0f(x) - (1 - x) * g0f(0))
    return A

def dA_dxf(xy, bcf, bcdf):
    (x, y) = xy
    (f0f, g0f) = bcf
    (df0_dyf, dg0_dxf) = bcdf
    dA_dx = -f0f(y) + (1 - y) * (dg0_dxf(x) + g0f(0))
    return dA_dx

def dA_dyf(xy, bcf, bcdf):
    (x, y) = xy
    (f0f, g0f) = bcf
    (df0_dyf, dg0_dxf) = bcdf
    dA_dy = (1 - x) * df0_dyf(y) - g0f(x) + (1 - x) * g0f(0)
    return dA_dy

del_Af = (dA_dxf, dA_dyf)

def Pf(xy):
    (x, y) = xy
    P = x * y
    return P

def dP_dxf(xy):
    (x, y) = xy
    dP_dx = y
    return dP_dx

def dP_dyf(xy):
    (x, y) = xy
    dP_dy = x
    return dP_dy

del_Pf = ( dP_dxf, dP_dyf )

# Define the trial solution and its derivatives.
def psitf(xy, N, bcf):
    A = Af(xy, bcf)
    P = Pf(xy)
    psit = A + P * N
    return psit

#********************************************************************************

# Function to solve a 2-variable, 1st-order PDE IVP using a single-hidden-layer
# feedforward neural network with 2 input nodes and a single output node.

def nnpde1(
        Gf,                            # 2-variable, 1st-order PDE IVP
        bcf,                           # BC functions
        bcdf,                          # BC function derivatives
        dG_dpsif,                      # Partial of G wrt psi
        del_Gf,                        # Gradient of G
        dG_ddel_psif,                  # Partials of G wrt del psi
        d2G_dpsi2f,                    # 2nd partial wrt psi
        d2G_ddel_psi_dpsif,            # Cross-partials
        d2G_dpsi_ddel_psif,            # Cross-partials
        d2G_ddel_psi2f,                # 2nd partials wrt del psi
        x,                             # Training points as pairs
        nhid,                          # Node count in hidden layer
        maxepochs = default_maxepochs, # Max training epochs
        eta = default_eta,             # Learning rate
        debug = default_debug,
        verbose = default_verbose
):
    if debug: print('Starting nnpde1().')
    if debug: print('Gf =', Gf)
    if debug: print('bcf =', bcf)
    if debug: print('bcdf =', bcdf)
    if debug: print('dG_dpsif =', dG_dpsif)
    if debug: print('del_Gf =', del_Gf)
    if debug: print('dG_ddel_psif =', dG_ddel_psif)
    if debug: print('d2G_dpsi2f =', d2G_dpsi2f)
    if debug: print('d2G_ddel_psi_dpsif =', d2G_ddel_psi_dpsif)
    if debug: print('d2G_dpsi_ddel_psif =', d2G_dpsi_ddel_psif)
    if debug: print('d2G_ddel_psi2f =', d2G_ddel_psi2f)
    if debug: print('x =', x)
    if debug: print('nhid =', nhid)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments.
    assert Gf
    assert len(bcf) > 0
    assert len(bcdf) == len(bcf)
    assert dG_dpsif
    assert len(del_Gf) == len(bcf)
    assert len(dG_ddel_psif) == len(bcf)
    assert len(d2G_ddel_psi_dpsif) == len(bcf)
    assert len(d2G_dpsi_ddel_psif) == len(bcf)
    assert len(d2G_ddel_psi2f) == len(bcf)
    assert len(x) > 0
    assert nhid > 0
    assert maxepochs > 0
    assert eta > 0

    #----------------------------------------------------------------------------

    # Determine the number of training points.
    ntrain = len(x)
    if debug: print('ntrain =', ntrain)

    #----------------------------------------------------------------------------

    # Create the network.

    # Create an array to hold the weights connecting the input nodes to the
    # hidden nodes. The weights are initialized with a uniform random
    # distribution.
    w = np.random.uniform(w_min, w_max, (2, nhid))
    if debug: print('w =', w)

    # Create an array to hold the biases for the hidden nodes. The
    # biases are initialized with a uniform random distribution.
    u = np.random.uniform(u_min, u_max, nhid)
    if debug: print('u =', u)

    # Create an array to hold the weights connecting the hidden nodes
    # to the output node. The weights are initialized with a uniform
    # random distribution.
    v = np.random.uniform(v_min, v_max, nhid)
    if debug: print('v =', v)

    # Change notation for convenience.
    n = ntrain
    if debug: print('n =', n)
    m = len(bcf)
    if debug: print('m =', m)
    H = nhid
    if debug: print('H =', H)

    #----------------------------------------------------------------------------

    # Run the network.
    for epoch in range(maxepochs):
        if debug: print('Starting epoch %d.' % epoch)

        # Compute the input, the sigmoid function and its derivatives,
        # for each hidden node.
        z = np.zeros((n, H))
        s = np.zeros((n, H))
        s1 = np.zeros((n, H))
        s2 = np.zeros((n, H))
        s3 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i][k] = u[k]
                for j in range(m):
                    z[i][k] += w[j][k] * x[i][j]
                    s[i][k] = sigma(z[i][k])
                    s1[i][k] = dsigma_dz(z[i][k])
                    s2[i][k] = d2sigma_dz2(z[i][k])
                    s3[i][k] = d3sigma_dz3(z[i][k])
        if debug: print('z =', z)
        if debug: print('s =', s)
        if debug: print('s1 =', s1)
        if debug: print('s2 =', s2)
        if debug: print('s3 =', s3)

        # Compute the network output and its derivatives, for each
        # training point.
        N = np.zeros(n)
        dN_dx = np.zeros((n, m))
        dN_dv = np.zeros((n, H))
        dN_du = np.zeros((n, H))
        dN_dw = np.zeros((n, m, H))
        d2N_dv2 = np.zeros((n, H))
        d2N_du2 = np.zeros((n, H))
        d2N_dw2 = np.zeros((n, m, H))
        d2N_dvdx = np.zeros((n, m, H))
        d2N_dudx = np.zeros((n, m, H))
        d2N_dwdx = np.zeros((n, m, H))
        d3N_dv2dx = np.zeros((n, m, H))
        d3N_du2dx = np.zeros((n, m, H))
        d3N_dw2dx = np.zeros((n, m, H))
        for i in range(n):
            for k in range(H):
                N[i] += v[k] * s[i][k]
                dN_dv[i][k] = s[i][k]
                dN_du[i][k] = v[k] * s1[i][k]
                d2N_dv2[i][k] = 0
                d2N_du2[i][k] = v[k] * s2[i][k]
                for j in range(m):
                    dN_dx[i][j] += v[k] * s1[i][k] * w[j][k]
                    dN_dw[i][j][k] = v[k] * s1[i][k] * x[i][j]
                    d2N_dw2[i][j][k] = v[k] * s2[i][k] * x[i][j]**2
                    d2N_dvdx[i][j][k] = s1[i][k] * w[j][k]
                    d2N_dudx[i][j][k] = v[k] * s2[i][k] * w[j][k]
                    d2N_dwdx[i][j][k] = (
                        v[k] * s1[i][k] + v[k] * s2[i][k] * w[j][k] * x[i][j]
                    )
                    d3N_dv2dx[i][j][k] = 0
                    d3N_du2dx[i][j][k] = v[k] * s3[i][k] * w[j][k]
                    d3N_dw2dx[i][j][k] = (
                        2 * v[k] * s2[i][k] * x[i][j] +
                        v[k] * s3[i][k] * w[j][k] * x[i][j]**2
                    )
        if debug: print('N =', N)
        if debug: print('dN_dx =', dN_dx)
        if debug: print('dN_dv =', dN_dv)
        if debug: print('dN_du =', dN_du)
        if debug: print('dN_dw =', dN_dw)
        if debug: print('d2N_dv2 =', d2N_dv2)
        if debug: print('d2N_du2 =', d2N_du2)
        if debug: print('d2N_dw2 =', d2N_dw2)
        if debug: print('d2N_dvdx =', d2N_dvdx)
        if debug: print('d2N_dudx =', d2N_dudx)
        if debug: print('d2N_dwdx =', d2N_dwdx)
        if debug: print('d3N_dv2dx =', d3N_dv2dx)
        if debug: print('d3N_du2dx =', d3N_du2dx)
        if debug: print('d3N_dw2dx =', d3N_dw2dx)

        #------------------------------------------------------------------------

        # Compute the value of the trial solution and its derivatives,
        # for each training point.
        A = np.zeros(n)
        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        psit = np.zeros(n)
        dpsit_dx = np.zeros((n, m))
        dpsit_dv = np.zeros((n, H))
        dpsit_du = np.zeros((n, H))
        dpsit_dw = np.zeros((n, m, H))
        d2psit_dv2 = np.zeros((n, H))
        d2psit_du2 = np.zeros((n, H))
        d2psit_dw2 = np.zeros((n, m, H))
        d2psit_dvdx = np.zeros((n, m, H))
        d2psit_dudx = np.zeros((n, m, H))
        d2psit_dwdx = np.zeros((n, m, H))
        d3psit_dv2dx = np.zeros((n, m, H))
        d3psit_du2dx = np.zeros((n, m, H))
        d3psit_dw2dx = np.zeros((n, m, H))
        for i in range(n):
            A[i] = Af(x[i], bcf)
            P[i] = Pf(x[i])
            psit[i] = psitf(x[i], N[i], bcf)
            for j in range(m):
                dA_dx[i][j] = del_Af[j](x[i], bcf, bcdf)
                dP_dx[i][j] = del_Pf[j](x[i])
                dpsit_dx[i][j] = (
                    dA_dx[i][j] + P[i] * dN_dx[i][j] + dP_dx[i][j] * N[i]
                )
            for k in range(H):
                dpsit_dv[i][k] = P[i] * dN_dv[i][k]
                dpsit_du[i][k] = P[i] * dN_du[i][k]
                d2psit_dv2[i][k] = P[i] * d2N_dv2[i][k]
                d2psit_du2[i][k] = P[i] * d2N_du2[i][k]
                for j in range(m):
                    dpsit_dw[i][j][k] = P[i] * dN_dw[i][j][k]
                    d2psit_dw2[i][j][k] = P[i] * d2N_dw2[i][j][k]
                    d2psit_dvdx[i][j][k] = (
                        P[i] * d2N_dvdx[i][j][k] + dP_dx[i][j] * dN_dv[i][k]
                    )
                    d2psit_dudx[i][j][k] = (
                        P[i] * d2N_dudx[i][j][k] + dP_dx[i][j] * dN_du[i][k]
                    )
                    d2psit_dwdx[i][j][k] = (
                        P[i] * d2N_dwdx[i][j][k] + dP_dx[i][j] * dN_dw[i][j][k]
                    )
                    d3psit_dv2dx[i][j][k] = (
                        P[i] * d3N_dv2dx[i][j][k] + dP_dx[i][j] * d2N_dv2[i][k]
                    )
                    d3psit_du2dx[i][j][k] = (
                        P[i] * d3N_du2dx[i][j][k] + dP_dx[i][j] * d2N_du2[i][k]
                    )
                    d3psit_dw2dx[i][j][k] = (
                        P[i] * d3N_dw2dx[i][j][k] +
                        dP_dx[i][j] * d2N_dw2[i][j][k]
                    )
        if debug: print('A =', A)
        if debug: print('dA_dx =', dA_dx)
        if debug: print('P =', P)
        if debug: print('dP_dx =', dP_dx)
        if debug: print('psit =', psit)
        if debug: print('dpsit_dx =', dpsit_dx)
        if debug: print('dpsit_dv =', dpsit_dv)
        if debug: print('dpsit_du =', dpsit_du)
        if debug: print('dpsit_dw =', dpsit_dw)
        if debug: print('d2psit_dv2 =', d2psit_dv2)
        if debug: print('d2psit_du2 =', d2psit_du2)
        if debug: print('d2psit_dw2 =', d2psit_dw2)
        if debug: print('d2psit_dvdx =', d2psit_dvdx)
        if debug: print('d2psit_dudx =', d2psit_dudx)
        if debug: print('d2psit_dwdx =', d2psit_dwdx)
        if debug: print('d3psit_dv2dx =', d3psit_dv2dx)
        if debug: print('d3psit_du2dx =', d3psit_du2dx)
        if debug: print('d3psit_dw2dx =', d3psit_dw2dx)

        # Compute the value of the original differential equation
        # for each training point, and its derivatives.
        G = np.zeros(n)
        dG_dx = np.zeros((n, m))
        dG_dpsit = np.zeros(n)
        dG_ddel_psit = np.zeros((n, m))
        dG_dv = np.zeros((n, H))
        dG_du = np.zeros((n, H))
        dG_dw = np.zeros((n, m, H))
        d2G_dpsit2 = np.zeros(n)
        d2G_ddel_psit_dpsit = np.zeros((n, m))
        d2G_dpsit_ddel_psit = np.zeros((n, m))
        d2G_ddel_psit2 = np.zeros((n, m))
        d2G_dvdpsit = np.zeros((n, H))
        d2G_dudpsit = np.zeros((n, H))
        d2G_dwdpsit = np.zeros((n, m, H))
        d2G_dvddel_psit = np.zeros((n, m, H))
        d2G_duddel_psit = np.zeros((n, m, H))
        d2G_dwddel_psit = np.zeros((n, m, H))
        d2G_dv2 = np.zeros((n, H))
        d2G_du2 = np.zeros((n, H))
        d2G_dw2 = np.zeros((n, m, H))
        for i in range(n):
            G[i] = Gf(x[i], psit[i], dpsit_dx[i])
            dG_dpsit[i] = dG_dpsif(x[i], psit[i], dpsit_dx[i])
            d2G_dpsit2[i] = d2G_dpsi2f(x[i], psit[i], dpsit_dx[i])
            for j in range(m):
                dG_dx[i][j] = del_Gf[j](x[i], psit[i], dpsit_dx[i])
                dG_ddel_psit[i][j] = dG_ddel_psif[j](x[i], psit[i], dpsit_dx[i])
                d2G_ddel_psit_dpsit[i][j] = (
                    dG_ddel_psif[j](x[i], psit[i], dpsit_dx[i])
                )
                d2G_dpsit_ddel_psit[i][j] = (
                    d2G_dpsi_ddel_psif[j](x[i], psit[i], dpsit_dx[i])
                )
                d2G_ddel_psit_dpsit[i][j] = (
                    d2G_ddel_psi_dpsif[j](x[i], psit[i], dpsit_dx[i])
                )
                d2G_ddel_psit2[i] = (
                    d2G_ddel_psi2f[j](x[i], psit[i], dpsit_dx[i])
                )
            for k in range(H):
                dG_dv[i][k] = dG_dpsit[i] * dpsit_dv[i][k]
                dG_du[i][k] = dG_dpsit[i] * dpsit_du[i][k]
                d2G_dvdpsit[i][k] = d2G_dpsit2[i] * dpsit_dv[i][k]
                d2G_dudpsit[i][k] = d2G_dpsit2[i] * dpsit_du[i][k]
                d2G_dv2[i][k] = (
                    dG_dpsit[i] * d2psit_dv2[i][k] +
                    d2G_dvdpsit[i][k] * dpsit_dv[i][k]**2
                )
                d2G_du2[i][k] = (
                    dG_dpsit[i] * d2psit_du2[i][k] +
                    d2G_dudpsit[i][k] * dpsit_du[i][k]**2
                )
                for j in range(m):
                    dG_dv[i][k] += dG_ddel_psit[i][j] * d2psit_dvdx[i][j][k]
                    dG_du[i][k] += dG_ddel_psit[i][j] * d2psit_dudx[i][j][k]
                    dG_dw[i][j][k] = (
                        dG_dpsit[i] * dpsit_dw[i][j][k] +
                        dG_ddel_psit[i][j] * d2psit_dwdx[i][j][k]
                    )
                    d2G_dvdpsit[i][k] += (
                        d2G_ddel_psit_dpsit[i][j] * d2psit_dvdx[i][j][k]
                    )
                    d2G_dudpsit[i][k] += (
                        d2G_ddel_psit_dpsit[i][j] * d2psit_dudx[i][j][k]
                    )
                    d2G_dwdpsit[i][j][k] = (
                        d2G_ddel_psit_dpsit[i][j] * d2psit_dwdx[i][j][k]
                    )
                    d2G_dvddel_psit[i][j][k] = (
                        d2G_dpsit_ddel_psit[i][j] * dpsit_dv[i][k] +
                        d2G_ddel_psit2[i][j] * d2psit_dvdx[i][j][k]
                    )
                    d2G_duddel_psit[i][j][k] = (
                        d2G_dpsit_ddel_psit[i][j] * dpsit_du[i][k] +
                        d2G_ddel_psit2[i][j] * d2psit_dudx[i][j][k]
                    )
                    d2G_dwddel_psit[i][j][k] = (
                        d2G_dpsit_ddel_psit[i][j] * dpsit_dw[i][j][k] +
                        d2G_ddel_psit2[i][j] * d2psit_dwdx[i][j][k]
                    )
                    d2G_dv2[i][k] += (
                        dG_ddel_psit[i][j] * d3psit_dv2dx[i][j][k] +
                        d2G_dvddel_psit[i][j][k] * d2psit_dvdx[i][j][k]
                    )
                    d2G_du2[i][k] += (
                        dG_ddel_psit[i][j] * d3psit_du2dx[i][j][k] +
                        d2G_duddel_psit[i][j][k] * d2psit_dudx[i][j][k]
                    )
                    d2G_dw2[i][j][k] = (
                        dG_dpsit[i] * d2psit_dw2[i][j][k] +
                        d2G_dwdpsit[i][j][k] * dpsit_dw[i][j][k]**2 +
                        dG_ddel_psit[i][j] * d3psit_du2dx[i][j][k] +
                        d2G_duddel_psit[i][j][k] * d2psit_dudx[i][j][k]
                    )
        if debug: print('G =', G)
        if debug: print('dG_dx =', dG_dx)
        if debug: print('dG_dpsit =', dG_dpsit)
        if debug: print('d2G_dpsit2 =', d2G_dpsit2)
        if debug: print('dG_ddel_psit =', dG_ddel_psit)
        if debug: print('d2G_ddel_psit_dpsit =', d2G_ddel_psit_dpsit)
        if debug: print('d2G_ddel_psit2 =', d2G_ddel_psit2)
        if debug: print('dG_dv =', dG_dv)
        if debug: print('dG_du =', dG_du)
        if debug: print('dG_dw =', dG_dw)
        if debug: print('d2G_dvdpsit =', d2G_dvdpsit)
        if debug: print('d2G_dudpsit =', d2G_dudpsit)
        if debug: print('d2G_dwdpsit =', d2G_dwdpsit)
        if debug: print('d2G_dvddel_psi2 =', d2G_dvddel_psit)
        if debug: print('d2G_dv2 =', d2G_dv2)

        # Compute the error function for this pass.
        E = 0
        for i in range(n):
            E += G[i]**2
        if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to the
        # network parameters.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros((m, H))
        d2E_dv2 = np.zeros(H)
        d2E_du2 = np.zeros(H)
        d2E_dw2 = np.zeros((m, H))
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2 * G[i] * dG_dv[i][k]
                dE_du[k] += 2 * G[i] * dG_du[i][k]
                d2E_dv2[k] += 2 * (G[i] * d2G_dv2[i][k] + dG_dv[i][k]**2)
                d2E_du2[k] += 2 * (G[i] * d2G_du2[i][k] + dG_du[i][k]**2)
            for j in range(m):
                for i in range(n):
                    dE_dw[j][k] += 2 * G[i] * dG_dw[i][j][k]
                    d2E_dw2[j][k] += 2 * (
                        G[i] * d2G_dw2[i][j][k] + dG_dw[i][j][k]**2
                    )
        if debug: print('dE_dv =', dE_dv)
        if debug: print('dE_du =', dE_du)
        if debug: print('dE_dw =', dE_dw)
        if debug: print('d2E_dv2 =', d2E_dv2)
        if debug: print('d2E_du2 =', d2E_du2)
        if debug: print('d2E_dw2 =', d2E_dw2)

        #------------------------------------------------------------------------

        # Update the weights and biases.
    
        # Compute the new values of the network parameters.
        v_new = np.zeros(H)
        u_new = np.zeros(H)
        w_new = np.zeros((m, H))
        for k in range(H):
            v_new[k] = v[k] - eta * dE_dv[k] / d2E_dv2[k]
            u_new[k] = u[k] - eta * dE_du[k] / d2E_du2[k]
            for j in range(m):
                w_new[j][k] = w[j][k] - eta * dE_dw[j][k] / d2E_dw2[j][k]
        if debug: print('v_new =', v_new)
        if debug: print('u_new =', u_new)
        if debug: print('w_new =', w_new)

        if verbose: print(epoch, E)

        # Save the new weights and biases.
        v = v_new
        u = u_new
        w = w_new

    # Return the final solution.
    if debug: print('Ending nnpde1().')
    return (psit, dpsit_dx)

#--------------------------------------------------------------------------------

# Begin main program.

if __name__ == '__main__':

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description = 'Solve a 2-variable, 1st-order PDE IVP with a neural net',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = 'Experiment with the settings to find what works.'
    )
    # print('parser =', parser)

    # Add command-line options.
    parser.add_argument('--debug', '-d',
                        action = 'store_true',
                        default = default_debug,
                        help = 'Produce debugging output')
    parser.add_argument('--eta', type = float,
                        default = default_eta,
                        help = 'Learning rate for parameter adjustment')
    parser.add_argument('--maxepochs', type = int,
                        default = default_maxepochs,
                        help = 'Maximum number of training epochs')
    parser.add_argument('--nhid', type = int,
                        default = default_nhid,
                        help = 'Number of hidden-layer nodes to use')
    parser.add_argument('--ntrain', type = int,
                        default = default_ntrain,
                        help = 'Number of evenly-spaced training points to use along each dimension')
    parser.add_argument('--pde', type = str,
                        default = default_pde,
                        help = 'Name of module containing PDE to solve')
    parser.add_argument('--seed', type = int,
                        default = default_seed,
                        help = 'Random number generator seed')
    parser.add_argument('--verbose', '-v',
                        action = 'store_true',
                        help = 'Produce verbose output')
    parser.add_argument('--version', action = 'version',
                        version = '%(prog)s 0.0')

    # Fetch and process the arguments from the command line.
    args = parser.parse_args()
    if args.debug: print('args =', args)

    # Extract the processed options.
    debug = args.debug
    eta = args.eta
    maxepochs = args.maxepochs
    nhid = args.nhid
    ntrain = args.ntrain
    pde = args.pde
    seed = args.seed
    verbose = args.verbose
    if debug: print('debug =', debug)
    if debug: print('eta =', eta)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('nhid =', nhid)
    if debug: print('ntrain =', ntrain)
    if debug: print('pde =', pde)
    if debug: print('seed =', seed)
    if debug: print('verbose =', verbose)

    # Perform basic sanity checks on the command-line options.
    assert eta > 0
    assert maxepochs > 0
    assert nhid > 0
    assert ntrain > 0
    assert pde
    assert seed >= 0

    #----------------------------------------------------------------------------

    # Initialize the random number generator to ensure repeatable results.
    # if verbose: print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the specified PDE module.
    if verbose:
        print('Importing PDE module %s.' % pde)
    pdemod = import_module(pde)
    assert pdemod.Gf
    assert len(pdemod.bcf) > 0
    assert len(pdemod.bcdf) == len(pdemod.bcf)
    assert pdemod.dG_dpsif
    assert pdemod.del_Gf
    assert pdemod.dG_ddel_psif
    assert pdemod.d2G_dpsi2f
    assert len(pdemod.d2G_ddel_psi_dpsif) == len(pdemod.bcf)
    assert len(pdemod.d2G_dpsi_ddel_psif) == len(pdemod.bcf)
    assert len(pdemod.d2G_ddel_psi2f) == len(pdemod.bcf)
    assert pdemod.psiaf
    assert len(pdemod.del_psiaf) == len(pdemod.bcf)

    # Create the array of evenly-spaced training points. Use the same
    # values of the training points for each dimension.
    if verbose: print('Computing training points along 2 dimensions.')
    dxy = 1 / (ntrain - 1)
    if debug: print('dxy =', dxy)
    xt = [i * dxy for i in range(ntrain)]
    if debug: print('xt =', xt)
    yt = xt
    if debug: print('yt =', yt)

    # Determine the number of training points.
    nxt = len(xt)
    if debug: print('nxt =', nxt)
    nyt = len(yt)
    if debug: print('nyt =', nyt)
    ntrain = len(xt) * len(yt)
    if debug: print('ntrain =', ntrain)

    # Create the list of training points.
    # ((x0,y0),(x1,y0),(x2,y0),...
    #  (x0,y1),(x1,y1),(x2,y1),...
    x = np.zeros((ntrain, 2))
    for j in range(nyt):
        for i in range(nxt):
            k = j * nxt + i
            if debug: print('k =', k)
            x[k][0] = xt[i]
            x[k][1] = yt[j]
    if debug: print('x =', x)

    #----------------------------------------------------------------------------

    # Compute the 1st-order PDE solution using the neural network.
    (psit, del_psi) = nnpde1(
        pdemod.Gf,             # 2-variable, 1st-order PDE IVP to solve
        pdemod.bcf,            # BC functions
        pdemod.bcdf,           # BC function derivatives
        pdemod.dG_dpsif,       # Partial of G wrt psi
        pdemod.del_Gf,         # Gradient of G
        pdemod.dG_ddel_psif,   # Partials of G wrt del psi
        pdemod.d2G_dpsi2f,     # 2nd partial wrt psi
        pdemod.d2G_ddel_psi_dpsif, # Cross-partials
        pdemod.d2G_dpsi_ddel_psif, # Cross-partials
        pdemod.d2G_ddel_psi2f, # 2nd partial wrt del psi
        x,                     # Training points as pairs
        nhid = nhid,           # Node count in hidden layer
        maxepochs = maxepochs, # Max training epochs
        eta = default_eta,     # Learning rate
        debug = debug,
        verbose = verbose
    )

    #----------------------------------------------------------------------------
    debug = True
    # Compute the analytical solution at the training points.
    psia = np.zeros(len(x))
    for i in range(len(x)):
        psia[i] = pdemod.psiaf(x[i])
    if debug: print('psia =', psia)

    # Compute the analytical derivative at the training points.
    del_psia = np.zeros((len(x), len(x[1])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            del_psia[i][j] = pdemod.del_psiaf[j](x[i])
    if debug: print('del_psia =', del_psia)

    # Compute the RMSE of the trial solution.
    psi_err = psit - psia
    if debug: print('psi_err =', psi_err)
    rmse_psi = sqrt(sum(psi_err**2) / len(x))
    if debug: print('rmse_psi =', rmse_psi)

    # Compute the MSE of the trial derivative.
    del_psi_err = del_psi - del_psia
    if debug: print('del_psi_err =', del_psi_err)
    rmse_del_psi = np.zeros(len(x[0]))
    e2sum = np.zeros(len(x[0]))
    for j in range(len(x[0])):
        for i in range(len(x)):
            e2sum[j] += del_psi_err[i][j]**2
        rmse_del_psi[j] = sqrt(e2sum[j] / len(x))
    if debug: print('rmse_del_psi =', rmse_del_psi)

    # Print the report.
    print('    x        y      psia     psit   dpsia_dx dpsia_dy dpsit_dx dpsit_dy')
    for i in range(len(psia)):
        print('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' %
              (x[i][0], x[i][1],
               psia[i], psit[i],
               del_psia[i][0], del_psi[i][0],
               del_psia[i][1], del_psi[i][1])
        )
    print('RMSE      %f          %f     %f' %
          (rmse_psi, rmse_del_psi[0], rmse_del_psi[1]))

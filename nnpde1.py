#!/usr/bin/env python

# Use a neural network to solve a 1st-order PDE IVP.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
import importlib
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
v_min = -1
v_max = 1
u_min = -1
u_max = 1
w_min = -1
w_max = 1

#********************************************************************************

# The range of the trial solution is assumed to be [0, 1].

# Define the boundary and coefficient functions for the trial solution.
# N.B. ASSUMES ONLY 2 DIMENSIONS!
def fA(xv, bc):
    (x, y) = xv
    (f0, g0) = bc
    return (1 - x) * f0(y) + (1 - y) * (g0(x) - (1 - x) * g0(0))

def dfA_dxv(xv, bc, bcd):
    (x, y) = xv
    (f0, g0) = bc
    (df0_dy, dg0_dx) = bcd
    return -f0(y) + (1 - y) * (dg0_dx(x) + g0(0))

def dfA_dyv(xv, bc, bcd):
    (x, y) = xv
    (f0, g0) = bc
    (df0_dy, dg0_dx) = bcd
    return (1 - x) * df0_dy(y) - g0(x) + (1 - x) * g0(0)

dfA_dx = (dfA_dxv, dfA_dyv)

def d2fA_dxv2(xv, bc, bcd, bcdd):
    (x, y) = xv
    (f0, g0) = bc
    (df0_dy, dg0_dx) = bcd
    (d2f0_dy2, d2g0_dx2) = bcdd
    return (1 - y) * d2g0_dx2(x)

def d2fA_dyv2(xv, bc, bcd, bcdd):
    (x, y) = xv
    (f0, g0) = bc
    (df0_dy, dg0_dx) = bcd
    (d2f0_dy2, d2g0_dx2) = bcdd
    return (1 - x) * d2f0_dy2(x)

d2fA_dx2 = (d2fA_dxv2, d2fA_dyv2)

def fP(x):
    P = 1
    for j in range(len(x)):
        P *= x[j] * (1 - x[j])
    return P

def dfP_dxv(xv):
    (x, y) = xv
    return (1 - 2 * x) * y * (1 - y)

def dfP_dyv(xv):
    (x, y) = xv
    return (1 - 2 * y) * x * (1 - x)

dfP_dx = (dfP_dxv, dfP_dyv)

def d2fP_dxv2(xv):
    (x, y) = xv
    return -2 * y * (1 - y)

def d2fP_dyv2(xv):
    (x, y) = xv
    return -2  * x * (1 - x)

d2fP_dx2 = (d2fP_dxv2, d2fP_dyv2)

# Define the trial solution and its derivatives.
def psitrial(A, P, N):
    return A + P * N

#********************************************************************************

# Function to solve a 1st-order PDE BVP using a single-hidden-layer
# feedforward neural network.
def nnpde1(x,
           fG,
           dfG_dx,
           dfG_dpsi,
           dfG_dpsi_dx,
           d2fG_dx2,
           d2fG_dpsi2,
           d2fG_dpsi_dx2,
           bc, bcd, bcdd,
           maxepochs = default_maxepochs, eta = default_eta, nhid = default_nhid,
           debug = default_debug, verbose = default_verbose):
    if debug: print('x =', x)
    if debug: print('fG =', fG)
    if debug: print('dfG_dx =', dfG_dx)
    if debug: print('dfG_dpsi =', dfG_dpsi)
    if debug: print('dfG_dpsi_dx =', dfG_dpsi_dx)
    if debug: print('d2fG_dx2 =', d2fG_dx2)
    if debug: print('d2fG_dpsi2 =', d2fG_dpsi2)
    if debug: print('d2fG_dpsi_dx2 =', d2fG_dpsi_dx2)
    if debug: print('bc =', bc)
    if debug: print('bcd =', bcd)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('nhid =', nhid)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments.
    assert len(x) > 0
    assert fG
    assert len(dfG_dx) > 0
    assert dfG_dpsi
    assert len(dfG_dpsi_dx) == len(dfG_dx)
    assert len(d2fG_dx2) == len(dfG_dx)
    assert d2fG_dpsi2
    assert len(d2fG_dpsi_dx2) == len(dfG_dx)
    assert len(bc) == len(dfG_dx)
    assert len(bcd) == len(dfG_dx)
    assert len(bcdd) == len(dfG_dx)
    assert maxepochs > 0
    assert eta > 0
    assert nhid > 0

    #----------------------------------------------------------------------------

    # Determine the number of training points.
    ntrain = len(x)
    if debug: print('ntrain =', ntrain)

    # Find the number of dimensions.
    ndim = len(dfG_dx)
    if debug: print('ndim =', ndim)
    # <HACK>
    assert ndim == 2
    # </HACK>

    #----------------------------------------------------------------------------

    # Create the network.

    # Create an array to hold the weights connecting the hidden nodes
    # to the output node. The weights are initialized with a uniform
    # random distribution.
    v = np.random.uniform(v_min, v_max, nhid)
    if debug: print('v =', v)

    # Create an array to hold the biases for the hidden nodes. The
    # biases are initialized with a uniform random distribution.
    u = np.random.uniform(u_min, u_max, nhid)
    if debug: print('u =', u)

    # Create an array to hold the weights connecting the input nodes to the
    # hidden nodes. The weights are initialized with a uniform random
    # distribution.
    w = np.random.uniform(w_min, w_max, (ndim, nhid))
    if debug: print('w =', w)

    #----------------------------------------------------------------------------

    # Run the network.
    for epoch in range(maxepochs):

        if debug: print('Starting epoch %d.' % epoch)

        # Compute the input, the sigmoid function and its derivatives,
        # for each hidden node.
        # z = np.zeros((ntrain, nhid))
        # s = np.zeros((ntrain, nhid))
        # s1 = np.zeros((ntrain, nhid))
        # s2 = np.zeros((ntrain, nhid))
        # s3 = np.zeros((ntrain, nhid))
        # for i in range(ntrain):
        #     for k in range(nhid):
        #         z[i][k] = u[k]
        #         for j in range(ndim):
        #             z[i][k] += w[j][k] * x[i][j]
        #             s[i][k] = sigma(z[i][k])
        #             s1[i][k] = dsigma_dz(z[i][k])
        #             s2[i][k] = d2sigma_dz2(z[i][k])
        #             s3[i][k] = d3sigma_dz3(z[i][k])
        # if debug: print('z =', z)
        # if debug: print('s =', s)
        # if debug: print('s1 =', s1)
        # if debug: print('s2 =', s2)
        # if debug: print('s3 =', s3)

        # Compute the network output and its derivatives, for each
        # training point.
        # N = np.zeros(ntrain)
        # dN_dx = np.zeros((ntrain, ndim))
        # d2N_dx2 = np.zeros((ntrain, ndim))
        # dN_dv = np.zeros((ntrain, nhid))
        # dN_du = np.zeros((ntrain, nhid))
        # dN_dw = np.zeros((ntrain, ndim, nhid))
        # d2N_dv2 = np.zeros((ntrain, nhid))
        # d2N_du2 = np.zeros((ntrain, nhid))
        # d2N_dw2 = np.zeros((ntrain, ndim, nhid))
        # d2N_dvdx = np.zeros((ntrain, ndim, nhid))
        # d2N_dudx = np.zeros((ntrain, ndim, nhid))
        # d2N_dwdx = np.zeros((ntrain, ndim, nhid))
        # d3N_dv2dx = np.zeros((ntrain, ndim, nhid))
        # d3N_du2dx = np.zeros((ntrain, ndim, nhid))
        # d3N_dw2dx = np.zeros((ntrain, ndim, nhid))
        # for i in range(ntrain):
        #     for k in range(nhid):
        #         N[i] += v[k] * s[i][k]
        #         dN_dv[i][k] = s[i][k]
        #         dN_du[i][k] = v[k] * s1[i][k]
        #         # d2N_dv2[i][k] = 0
        #         d2N_du2[i][k] = v[k] * s2[i][k]
        #         for j in range(ndim):
        #             dN_dx[i][j] += v[k] * s1[i][k] * w[j][k]
        #             d2N_dx2[i][j] += v[k] * s2[i][k] * w[j][k]**2
        #             dN_dw[i][j][k] = v[k] * s1[i][k] * x[i][j]
        #             d2N_dw2[i][j][k] = v[k] * s2[i][k] * x[i][j]**2
        #             d2N_dvdx[i][j][k] = s1[i][k] * w[j][k]
        #             d2N_dudx[i][j][k] = v[k] * s2[i][k] * w[j][k]
        #             d2N_dwdx[i][j][k] = (
        #                 v[k] * (s1[i][k] + s2[i][k] * w[j][k] * x[i][j])
        #             )
        #             # d3N_dv2dx[i][j][k] = 0
        #             d3N_du2dx[i][j][k] = v[k] * s3[i][k] * w[j][k]
        #             d3N_dw2dx[i][j][k] = (
        #                 2 * v[k] * s2[i][k] * x[i][j] +
        #                 v[k] * s3[i][k] * w[j][k] * x[i][j]**2
        #             )
        # if debug: print('N =', N)
        # if debug: print('dN_dx =', dN_dx)
        # if debug: print('d2N_dx2 =', d2N_dx2)
        # if debug: print('dN_dv =', dN_dv)
        # if debug: print('dN_du =', dN_du)
        # if debug: print('dN_dw =', dN_dw)
        # if debug: print('d2N_dv2 =', d2N_dv2)
        # if debug: print('d2N_du2 =', d2N_du2)
        # if debug: print('d2N_dw2 =', d2N_dw2)
        # if debug: print('d2N_dvdx =', d2N_dvdx)
        # if debug: print('d2N_dudx =', d2N_dudx)
        # if debug: print('d2N_dwdx =', d2N_dwdx)
        # if debug: print('d3N_dv2dx =', d3N_dv2dx)
        # if debug: print('d3N_du2dx =', d3N_du2dx)
        # if debug: print('d3N_dw2dx =', d3N_dw2dx)

        #------------------------------------------------------------------------

        # Compute the value of the trial solution and its derivatives,
        # for each training point.
        # A = np.zeros(ntrain)
        # dA_dx = np.zeros((ntrain, ndim))
        # d2A_dx2 = np.zeros((ntrain, ndim))
        # P = np.zeros(ntrain)
        # dP_dx = np.zeros((ntrain, ndim))
        # d2P_dx2 = np.zeros((ntrain, ndim))
        # psit = np.zeros(ntrain)
        # dpsit_dx = np.zeros((ntrain, ndim))
        # d2psit_dx2 = np.zeros((ntrain, ndim))
        # dpsit_dv = np.zeros((ntrain, nhid))
        # dpsit_du = np.zeros((ntrain, nhid))
        # dpsit_dw = np.zeros((ntrain, ndim, nhid))
        # d2psit_dv2 = np.zeros((ntrain, nhid))
        # d2psit_du2 = np.zeros((ntrain, nhid))
        # d2psit_dw2 = np.zeros((ntrain, ndim, nhid))
        # d2psit_dvdx = np.zeros((ntrain, ndim, nhid))
        # d2psit_dudx = np.zeros((ntrain, ndim, nhid))
        # d2psit_dwdx = np.zeros((ntrain, ndim, nhid))
        # d3psit_dv2dx = np.zeros((ntrain, ndim, nhid))
        # d3psit_du2dx = np.zeros((ntrain, ndim, nhid))
        # d3psit_dw2dx = np.zeros((ntrain, ndim, nhid))
        # for i in range(ntrain):
        #     A[i] = fA(x[i], bc)
        #     P[i] = fP(x[i])
        #     psit[i] = psitrial(A[i], P[i], N[i])
        #     for j in range(ndim):
        #         dA_dx[i][j] = dfA_dx[j](x[i], bc, bcd)
        #         d2A_dx2[i][j] = d2fA_dx2[j](x[i], bc, bcd, bcdd)
        #         dP_dx[i][j] = dfP_dx[j](x[i])
        #         d2P_dx2[i][j] = d2fP_dx2[j](x[i])
        #         dpsit_dx[i][j] = (
        #             dA_dx[i][j] + P[i] * dN_dx[i][j] + dP_dx[i][j] * N[i]
        #         )
        #         d2psit_dx2[i][j] = (
        #             d2A_dx2[i][j] + P[i] * d2N_dx2[i][j] +
        #             2 * dP_dx[i][j] * dN_dx[i][j] + d2P_dx2[i][j] * N[i]
        #         )
        #     for k in range(nhid):
        #         dpsit_dv[i][k] = P[i] * dN_dv[i][k]
        #         dpsit_du[i][k] = P[i] * dN_du[i][k]
        #         d2psit_dv2[i][k] = P[i] * d2N_dv2[i][k]
        #         d2psit_du2[i][k] = P[i] * d2N_du2[i][k]
        #         for j in range(ndim):
        #             dpsit_dw[i][j][k] = P[i] * dN_dw[i][j][k]
        #             d2psit_dw2[i][j][k] = P[i] * d2N_dw2[i][j][k]
        #             d2psit_dvdx[i][j][k] = (
        #                 P[i] * d2N_dvdx[i][j][k] + dP_dx[i][j] * dN_dv[i][k]
        #             )
        #             d2psit_dudx[i][j][k] = (
        #                 P[i] * d2N_dudx[i][j][k] + dP_dx[i][j] * dN_du[i][k]
        #             )
        #             d2psit_dwdx[i][j][k] = (
        #                 P[i] * d2N_dwdx[i][j][k] + dP_dx[i][j] * dN_dw[i][j][k]
        #             )
        #             d3psit_dv2dx[i][j][k] = (
        #                 P[i] * d3N_dv2dx[i][j][k] + dP_dx[i][j] * d2N_dv2[i][k]
        #             )
        #             d3psit_du2dx[i][j][k] = (
        #                 P[i] * d3N_du2dx[i][j][k] + dP_dx[i][j] * d2N_du2[i][k]
        #             )
        #             d3psit_dw2dx[i][j][k] = (
        #                 P[i] * d3N_dw2dx[i][j][k] +
        #                 dP_dx[i][j] * d2N_dw2[i][j][k]
        #             )
        # if debug: print('A =', A)
        # if debug: print('dA_dx =', dA_dx)
        # if debug: print('d2A_dx2 =', d2A_dx2)
        # if debug: print('P =', P)
        # if debug: print('dP_dx =', dP_dx)
        # if debug: print('d2P_dx2 =', d2P_dx2)
        # if debug: print('psit =', psit)
        # if debug: print('dpsit_dx =', dpsit_dx)
        # if debug: print('d2psit_dx2 =', d2psit_dx2)
        # if debug: print('dpsit_dv =', dpsit_dv)
        # if debug: print('dpsit_du =', dpsit_du)
        # if debug: print('dpsit_dw =', dpsit_dw)
        # if debug: print('d2psit_dv2 =', d2psit_dv2)
        # if debug: print('d2psit_du2 =', d2psit_du2)
        # if debug: print('d2psit_dw2 =', d2psit_dw2)
        # if debug: print('d2psit_dvdx =', d2psit_dvdx)
        # if debug: print('d2psit_dudx =', d2psit_dudx)
        # if debug: print('d2psit_dwdx =', d2psit_dwdx)
        # if debug: print('d3psit_dv2dx =', d3psit_dv2dx)
        # if debug: print('d3psit_du2dx =', d3psit_du2dx)
        # if debug: print('d3psit_dw2dx =', d3psit_dw2dx)
        # print('A', A[23], A[34], A[57])
        # print('dA_dx', dA_dx[23], dA_dx[34], dA_dx[57])
        # print('d2A_dx2', d2A_dx2[23], d2A_dx2[34], d2A_dx2[57])
        # print('N', N[23], N[34], N[57])
        # print('P', P[23], P[34], P[57])
        # print('dP_dx', dP_dx[23], dP_dx[34], dP_dx[57])
        # print('d2P_dx2', d2P_dx2[23], d2P_dx2[34], d2P_dx2[57])
        # print('psit', psit[23], psit[34], psit[57])
        # print('dpsit_dx', dpsit_dx[23], dpsit_dx[34], dpsit_dx[57])
        # print('d2psit_dx2', d2psit_dx2[23], d2psit_dx2[34], d2psit_dx2[57])
        # print('dpsit_dv', dpsit_dv[23], dpsit_dv[34], dpsit_dv[57])
        # print('dpsit_du', dpsit_du[23], dpsit_du[34], dpsit_du[57])
        # print('dpsit_dw', dpsit_dw[23][1], dpsit_dw[34][0], dpsit_dw[57][1])
        # print('d2psit_dv2', d2psit_dv2[23], d2psit_dv2[34], d2psit_dv2[57])
        # print('d2psit_du2', d2psit_du2[23], d2psit_du2[34], d2psit_du2[57])
        # print('d2psit_dw2', d2psit_dw2[23][1][3], d2psit_dw2[34][0][4],
        #       d2psit_dw2[57][1][7])
        # print('d2psit_dvdx', d2psit_dvdx[23][1][3], d2psit_dvdx[34][0][4],
        #       d2psit_dvdx[57][1][7])
        # print('d2psit_dudx', d2psit_dudx[23][1][3], d2psit_dudx[34][0][4],
        #       d2psit_dudx[57][1][7])
        # print('d2psit_dwdx', d2psit_dwdx[23][1][3], d2psit_dwdx[34][0][4],
        #       d2psit_dwdx[57][1][7])
        # print('d3psit_dv2dx', d3psit_dv2dx[23][1][3], d3psit_dv2dx[34][0][4],
        #       d3psit_dv2dx[57][1][7])
        # print('d3psit_du2dx', d3psit_du2dx[23][1][3], d3psit_du2dx[34][0][4],
        #       d3psit_du2dx[57][1][7])
        # print('d3psit_dw2dx', d3psit_dw2dx[23][1][3], d3psit_dw2dx[34][0][4],
        #       d3psit_dw2dx[57][1][7])

        # Compute the value of the original differential equation
        # for each training point, and its derivatives.
        # G = np.zeros(ntrain)
        # dG_dx = np.zeros((ntrain, ndim))
        # d2G_dx2 = np.zeros((ntrain, ndim))
        # dG_dpsit = np.zeros(ntrain)
        # d2G_dpsit2 = np.zeros(ntrain)
        # dG_dpsit_dx = np.zeros((ntrain, ndim))
        # d2G_dpsit_dx2 = np.zeros((ntrain, ndim))
        # dG_dv = np.zeros((ntrain, nhid))
        # dG_du = np.zeros((ntrain, nhid))
        # dG_dw = np.zeros((ntrain, ndim, nhid))
        # d2G_dv2 = np.zeros((ntrain, nhid))
        # d2G_du2 = np.zeros((ntrain, nhid))
        # d2G_dw2 = np.zeros((ntrain, ndim, nhid))
        # for i in range(ntrain):
        #     G[i] = fG(x[i], psit[i], dpsit_dx[i])
        #     dG_dpsit[i] = dfG_dpsi(x[i], psit[i], dpsit_dx[i])
        #     d2G_dpsit2[i] = d2fG_dpsi2(x[i], psit[i], dpsit_dx[i])
        #     for j in range(ndim):
        #         dG_dx[i][j] = dfG_dx[j](x[i], psit[i], dpsit_dx[i])
        #         d2G_dx2[i][j] = d2fG_dx2[j](x[i], psit[i], dpsit_dx[i])
        #         dG_dpsit_dx[i][j] = dfG_dpsi_dx[j](x[i], psit[i], dpsit_dx[i])
        #         d2G_dpsit_dx2[i][j] = d2fG_dpsi_dx2[j](x[i], psit[i],
        #                                                dpsit_dx[i])
        #     for k in range(nhid):
        #         dG_dv[i][k] = dG_dpsit[i] * dpsit_dv[i][k]
        #         dG_du[i][k] = dG_dpsit[i] * dpsit_du[i][k]
        #         d2G_dv2[i][k] = (
        #             dG_dpsit[i] * d2psit_dv2[i][k] +
        #             d2G_dpsit2[i] * dpsit_dv[i][k]**2
        #         )
        #         d2G_du2[i][k] = (
        #             dG_dpsit[i] * d2psit_du2[i][k] +
        #             d2G_dpsit2[i] * dpsit_du[i][k]**2
        #         )
        #         for j in range(ndim):
        #             dG_dv[i][k] += dG_dpsit_dx[i][j] * d2psit_dvdx[i][j][k]
        #             dG_du[i][k] += dG_dpsit_dx[i][j] * d2psit_dudx[i][j][k]
        #             dG_dw[i][j][k] = (
        #                 dG_dpsit[i] * dpsit_dw[i][j][k] +
        #                 dG_dpsit_dx[i][j] * d2psit_dwdx[i][j][k]
        #             )
        #             d2G_dv2[i][k] += (
        #                 dG_dpsit_dx[i][j] * d3psit_dv2dx[i][j][k] +
        #                 d2G_dpsit_dx2[i][j] * d2psit_dvdx[i][j][k]**2
        #             )
        #             d2G_du2[i][k] += (
        #                 dG_dpsit_dx[i][j] * d3psit_du2dx[i][j][k] +
        #                 d2G_dpsit_dx2[i][j] * d2psit_dudx[i][j][k]**2
        #             )
        #             d2G_dw2[i][j][k] = (
        #                 dG_dpsit[i] * d2psit_dw2[i][j][k] +
        #                 d2G_dpsit2[i] * dpsit_dw[i][j][k]**2 +
        #                 dG_dpsit_dx[i][j] * d3psit_dw2dx[i][j][k] +
        #                 d2G_dpsit_dx2[i][j] * d2psit_dwdx[i][j][k]**2
        #             )
        # if debug: print('G =', G)
        # if debug: print('dG_dx =', dG_dx)
        # if debug: print('d2G_dx2 =', d2G_dx2)
        # if debug: print('dG_dpsit =', dG_dpsit)
        # if debug: print('d2G_dpsit2 =', d2G_dpsit2)
        # if debug: print('dG_dpsit_dx =', dG_dpsit_dx)
        # if debug: print('d2G_dpsit_dx2 =', d2G_dpsit_dx2)
        # if debug: print('dG_dv =', dG_dv)
        # if debug: print('dG_du =', dG_du)
        # if debug: print('dG_dw =', dG_dw)
        # if debug: print('d2G_dv2 =', d2G_dv2)
        # if debug: print('d2G_du2 =', d2G_du2)
        # if debug: print('d2G_dw2 =', d2G_dw2)
        # print('G', G[23], G[34], G[57])
        # print('dG_dx', dG_dx[23], dG_dx[34], dG_dx[57])
        # print('d2G_dx2', d2G_dx2[23], d2G_dx2[34], d2G_dx2[57])
        # print('dG_dpsit', dG_dpsit[23], dG_dpsit[34], dG_dpsit[57])
        # print('d2G_dpsit2', d2G_dpsit2[23], d2G_dpsit2[34], d2G_dpsit2[57])
        # print('dG_dpsit_dx', dG_dpsit_dx[23], dG_dpsit_dx[34], dG_dpsit_dx[57])
        # print('d2G_dpsit_dx2', d2G_dpsit_dx2[23], d2G_dpsit_dx2[34],
        #       d2G_dpsit_dx2[57])
        # print('dG_dv', dG_dv[23][3], dG_dv[34][4], dG_dv[57][7])
        # print('dG_du', dG_du[23][3], dG_du[34][4], dG_du[57][7])
        # print('dG_dw', dG_dw[23][1][3], dG_dw[34][0][4], dG_dw[57][1][7])
        # print('d2G_dv2', d2G_dv2[23][3], d2G_dv2[34][4], d2G_dv2[57][7])
        # print('d2G_du2', d2G_du2[23][3], d2G_du2[34][4], d2G_du2[57][7])
        # print('d2G_dw2', d2G_dw2[23][1][3], d2G_dw2[34][0][4], d2G_dw2[57][1][7])

        # Compute the error function for this pass.
        # E = 0
        # for i in range(ntrain):
        #     E += G[i]**2
        # if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to the
        # network parameters.
        # dE_dv = np.zeros(nhid)
        # dE_du = np.zeros(nhid)
        # dE_dw = np.zeros((ndim, nhid))
        # d2E_dv2 = np.zeros(nhid)
        # d2E_du2 = np.zeros(nhid)
        # d2E_dw2 = np.zeros((ndim, nhid))
        # for k in range(nhid):
        #     for i in range(ntrain):
        #         dE_dv[k] += 2 * G[i] * dG_dv[i][k]
        #         dE_du[k] += 2 * G[i] * dG_du[i][k]
        #         d2E_dv2[k] += 2 * (G[i] * d2G_dv2[i][k] + dG_dv[i][k]**2)
        #         d2E_du2[k] += 2 * (G[i] * d2G_du2[i][k] + dG_du[i][k]**2)
        #     for j in range(ndim):
        #         for i in range(ntrain):
        #             dE_dw[j][k] += 2 * G[i] * dG_dw[i][j][k]
        #             d2E_dw2[j][k] += 2 * (
        #                 G[i] * d2G_dw2[i][j][k] + dG_dw[i][j][k]**2
        #             )
        # if debug: print('dE_dv =', dE_dv)
        # if debug: print('dE_du =', dE_du)
        # if debug: print('dE_dw =', dE_dw)
        # if debug: print('d2E_dv2 =', d2E_dv2)
        # if debug: print('d2E_du2 =', d2E_du2)
        # if debug: print('d2E_dw2 =', d2E_dw2)
        # print('dE_dv', dE_dv[3], dE_dv[4], dE_dv[7])
        # print('dE_du', dE_du[3], dE_du[4], dE_du[7])
        # print('dE_dw', dE_dw[1][3], dE_dw[0][4], dE_dw[1][7])
        # print('d2E_dv2', d2E_dv2[3], d2E_dv2[4], d2E_dv2[7])
        # print('d2E_du2', d2E_du2[3], d2E_du2[4], d2E_du2[7])
        # print('d2E_dw2', d2E_dw2[1][3], d2E_dw2[0][4], d2E_dw2[1][7])

        #------------------------------------------------------------------------

        # Update the weights and biases.
    
        # Compute the new values of the network parameters.
        # v_new = np.zeros(nhid)
        # u_new = np.zeros(nhid)
        # w_new = np.zeros((ndim, nhid))
        # for k in range(nhid):
        #     v_new[k] = v[k] - eta * dE_dv[k] / d2E_dv2[k]
        #     u_new[k] = u[k] - eta * dE_du[k] / d2E_du2[k]
        #     for j in range(ndim):
        #         w_new[j][k] = w[j][k] - eta * dE_dw[j][k] / d2E_dw2[j][k]
        # if debug: print('v_new =', v_new)
        # if debug: print('u_new =', u_new)
        # if debug: print('w_new =', w_new)

        # if verbose: print(epoch, E)

        # Save the new weights and biases.
        # v = v_new
        # u = u_new
        # w = w_new

    # Return the final solution.
    # return (psit, dpsit_dx)
    return None, None

#--------------------------------------------------------------------------------

# Begin main program.

if __name__ == '__main__':

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description = 'Solve a 1st-order PDE IVP with a neural net',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = 'Experiment with the settings to find what works.'
    )
    # print('parser =', parser)

    # Add command-line options.
    parser.add_argument('--debug', '-d', action = 'store_true',
                        help = 'Produce debugging output')
    parser.add_argument('--eta', type = float, default = default_eta,
                        help = 'Learning rate for parameter adjustment')
    parser.add_argument('--maxepochs', type = int, default = default_maxepochs,
                        help = 'Maximum number of training epochs')
    parser.add_argument('--nhid', type = int, default = default_nhid,
                        help = 'Number of hidden-layer nodes to use')
    parser.add_argument('--ntrain', type = int, default = default_ntrain,
                        help = 'Number of evenly-spaced training points to use along each dimension')
    parser.add_argument('--pde', type = str, default = default_pde,
                        help = 'Name of module containing PDE to solve')
    parser.add_argument('--seed', type = int, default = default_seed,
                        help = 'Random number generator seed')
    parser.add_argument('--verbose', '-v', action = 'store_true',
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

    # Perform basic sanity checks on the command-line options.
    assert eta > 0
    assert maxepochs > 0
    assert nhid > 0
    assert ntrain > 0
    assert pde
    assert seed >= 0

    #----------------------------------------------------------------------------

    # Initialize the random number generator to ensure repeatable results.
    if verbose: print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the specified ODE module.
    pdemod = importlib.import_module(pde)
    assert pdemod.psia
    assert len(pdemod.dpsia_dx) > 0
    assert len(pdemod.d2psia_dx2) == len(pdemod.dpsia_dx)
    assert pdemod.fG
    assert len(pdemod.dfG_dx) == len(pdemod.dpsia_dx)
    assert pdemod.dfG_dpsi
    assert len(pdemod.dfG_dpsi_dx) == len(pdemod.dpsia_dx)
    assert len(pdemod.d2fG_dx2) == len(pdemod.dpsia_dx)
    assert pdemod.d2fG_dpsi2
    assert len(pdemod.d2fG_dpsi_dx2) == len(pdemod.dpsia_dx)
    assert len(pdemod.bc) == len(pdemod.dpsia_dx)
    assert len(pdemod.bcd) == len(pdemod.dpsia_dx)
    assert len(pdemod.bcdd) == len(pdemod.dpsia_dx)

    # Determine the number of dimensions.
    ndim = len(pdemod.dpsia_dx)
    if debug: print('ndim =', ndim)
    # <HACK>
    assert ndim == 2
    # </HACK>

    # Create the array of evenly-spaced training points. Use the same
    # values of the training points for each dimension.
    # N.B ASSUMES ndim=2!
    if verbose: print('Computing training points.')
    xtt = np.linspace(pdemod.xmin, pdemod.xmax, ntrain)
    if debug: print('xtt =', xtt)
    xt = np.zeros((ntrain**ndim, ndim))
    for i in range(ntrain):
        for ii in range(ntrain):
            xt[i * ntrain + ii][0] = xtt[i]
            xt[i * ntrain + ii][1] = xtt[ii]
    if debug: print('xt =', xt)

    #----------------------------------------------------------------------------

    # Compute the 1st-order PDE solution using the neural network.
    (psit, dpsit_dx) = nnpde1(xt,
                              pdemod.fG,
                              pdemod.dfG_dx,
                              pdemod.dfG_dpsi,
                              pdemod.dfG_dpsi_dx,
                              pdemod.d2fG_dx2,
                              pdemod.d2fG_dpsi2,
                              pdemod.d2fG_dpsi_dx2,
                              pdemod.bc, pdemod.bcd, pdemod.bcdd,
                              maxepochs = maxepochs, eta = eta, nhid = nhid,
                              debug = debug, verbose = verbose)

    #----------------------------------------------------------------------------

    # Compute the analytical solution at the training points.
    psia = np.zeros(ntrain * ntrain)
    for i in range(ntrain * ntrain):
        psia[i] = pdemod.psia(xt[i])
    if debug: print('psia =', psia)

    # Compute the analytical derivative at the training points.
    dpsia_dx = np.zeros((ntrain * ntrain, ndim))
    for i in range(ntrain * ntrain):
        for j in range(ndim):
            dpsia_dx[i][j] = pdemod.dpsia_dx[j](xt[i])
    if debug: print('dpsia_dx =', dpsia_dx)

    # Compute the MSE of the trial solution.
    psi_err = psit - psia
    if debug: print('psi_err =', psi_err)
    mse_psi = sqrt(sum(psi_err**2) / ntrain)
    if debug: print('mse_psi =', mse_psi)

    # Compute the MSE of the trial derivative.
    # dpsi_dx_err = dpsit_dx - dpsia_dx
    # if debug: print('dpsi_dx_err =', dpsi_dx_err)
    # mse_dpsi_dx = np.zeros(ndim)
    # for j in range(ndim):
    #     mse_dpsi_dx[j] = sqrt(sum((dpsit_dx[:][j] - dpsia_dx[:][j])**2) /
    #                           ntrain**2)
    # if debug: print('mse_dpsi_dx =', mse_dpsi_dx)

    # Print the report.
    print('    x        y      psia     psit   dpsia_dx dpsia_dy dpsit_dx dpsit_dy')
    for i in range(ntrain**ndim):
        print('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' %
              (xt[i][0], xt[i][1], psia[i], psit[i],
               dpsia_dx[i][0], dpsia_dx[i][1], dpsit_dx[i][0], dpsit_dx[i][1]))
    # print('MSE      %f          %f     %f' %
    #       (mse_psi, mse_dpsi_dx[0], mse_dpsi_dx[1]))

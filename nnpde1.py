#!/usr/bin/env python

# Use a neural network to solve a 1st-order PDE IVP.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
import importlib
# from math import sqrt
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

# Define the trial solution and its derivatives.
def psitrial(x, N, bc):
    return fA(x, bc) + fP(x) * N

# def dpsitrial_dxv(xv, N, dN_dx, bc, bcd):
#     (x, y) = xv
#     (f0, g0) = bc
#     (df0_dx, dg0_dy) = bcd
#     return (
#         dA_dx[0](xv, bc, bcd) + fP(xv) * dN_dx + dP_dx[0](xv) * N
#     )

# def dpsitrial_dyv(xv, N, dN_dx, bc, bcd):
#     return 0

# dpsitrial_dx = (dpsitrial_dxv, dpsitrial_dyv)

#********************************************************************************

# Function to solve a 1st-order PDE BVP using a single-hidden-layer
# feedforward neural network.
def nnpde1(x, G, dG_dx, dG_dy_dx, d2G_dx2, d2G_dy_dx2, bc, bcd,
           maxepochs = default_maxepochs, eta = default_eta, nhid = default_nhid,
           debug = default_debug, verbose = default_verbose):
    if debug: print('x =', x)
    if debug: print('G =', G)
    if debug: print('dG_dx =', dG_dx)
    if debug: print('dG_dy_dx =', dG_dy_dx)
    if debug: print('d2G_dx2 =', d2G_dx2)
    if debug: print('d2G_dy_dx2 =', d2G_dy_dx2)
    if debug: print('bc =', bc)
    if debug: print('bcd =', bcd)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('nhid =', nhid)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments.
    assert len(x) > 0
    assert G
    assert len(dG_dx) > 0
    assert len(dG_dy_dx) == len(dG_dx)
    assert len(d2G_dx2) == len(dG_dx)
    assert len(d2G_dy_dx2) == len(dG_dx)
    assert len(bc) == len(dG_dx)
    assert len(bcd) == len(dG_dx)
    assert maxepochs > 0
    assert eta > 0
    assert nhid > 0

    #----------------------------------------------------------------------------

    # Determine the number of training points.
    ntrain = len(x)
    if debug: print('ntrain =', ntrain)

    # Find the number of dimensions.
    ndim = len(dG_dx)
    if debug: print('ndim =', ndim)

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

    # Create an array to hold the weights connecting the input node to the
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
        z = np.zeros((ntrain, nhid))
        print(z)
        s = np.zeros((ntrain, nhid))
        s1 = np.zeros((ntrain, nhid))
        s2 = np.zeros((ntrain, nhid))
        s3 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for k in range(nhid):
                z[i][k] = u[k]
                for j in range(ndim):
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
        N = np.zeros(ntrain)
        dN_dx = np.zeros((ntrain, ndim))
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
        for i in range(ntrain):
            for k in range(nhid):
                N[i] += v[k] * s[i][k]
                for j in range(ndim):
                    dN_dx[i][j] += v[k] * w[j][k] * s1[i][k]
        #             d2N_dx2[i][j] += v[k] * w[j][k]**2 * s2[i][k]
        #             dN_dw[i][j][k] = v[k] * s1[i][k] * x[i][j]
        #             d2N_dw2[i][j][k] = v[k] * s2[i][k] * x[i][j]**2
        #             d2N_dvdx[i][j][k] = w[j][k] * s1[i][k]
        #             d2N_dudx[i][j][k] = v[k] * w[j][k] * s2[i][k]
        #             d2N_dwdx[i][j][k] = (
        #                 v[k] * w[j][k] * s2[i][k] * x[i][j] + v[k] * s1[i][k]
        #             )
        #             d3N_dv2dx[i][j][k] = 0
        #             d3N_du2dx[i][j][k] = v[k] * w[j][k] * s3[i][k]
        #             d3N_dw2dx[i][j] = (
        #                 v[k] * (w[j][k] * s3[i][k] * x[i][j] + s2[i][k]) +
        #                 v[k] * s2[i][k] * x[i][j]
        #             )
        #         dN_dv[i][k] = s[i][k]
        #         dN_du[i][k] = v[k] * s1[i][k]
        #         d2N_dv2[i][k] = 0
        #         d2N_du2[i][k] = v[k] * s2[i][k]
        if debug: print('N =', N)
        if debug: print('dN_dx =', dN_dx)
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

        # Compute boundary value constants.
        # (f0, g0) = bc

        # Compute the value of the trial solution and its derivatives,
        # for each training point.
        A = np.zeros(ntrain)
        dA_dx = np.zeros((ntrain, ndim))
        P = np.ones(ntrain)
        dP_dx = np.zeros((ntrain, ndim))
        psit = np.zeros(ntrain)
        dpsit_dx = np.zeros((ntrain, ndim))
        # dyt_dv = np.zeros((ntrain, nhid))
        # dyt_du = np.zeros((ntrain, nhid))
        # dyt_dw = np.zeros((ntrain, nhid))
        # d2yt_dv2 = np.zeros((ntrain, nhid))
        # d2yt_du2 = np.zeros((ntrain, nhid))
        # d2yt_dw2 = np.zeros((ntrain, nhid))
        # d2yt_dvdx = np.zeros((ntrain, nhid))
        # d2yt_dudx = np.zeros((ntrain, nhid))
        # d2yt_dwdx = np.zeros((ntrain, nhid))
        # d3yt_dv2dx = np.zeros((ntrain, nhid))
        # d3yt_du2dx = np.zeros((ntrain, nhid))
        # d3yt_dw2dx = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            A[i] = fA(x[i], bc)
            P[i] = fP(x[i])
            psit[i] = psitrial(x[i], N[i], bc)
            for j in range(ndim):
                print(i, j)
                dA_dx[i][j] = dfA_dx[j](x[i], bc, bcd)
                dP_dx[i][j] = dfP_dx[j](x[i])
                dpsit_dx[i][j] = (
                    dA_dx[i][j] + P[i] * dN_dx[i][j] + dP_dx[i][j] * N[i]
                )
                # dyt_dv[i][j] = x[i] * dN_dv[i][j]
        #         dyt_du[i][j] = x[i] * dN_du[i][j]
        #         dyt_dw[i][j] = x[i] * dN_dw[i][j]
        #         d2yt_dv2[i][j] = x[i] * d2N_dv2[i][j]
        #         d2yt_du2[i][j] = x[i] * d2N_du2[i][j]
        #         d2yt_dw2[i][j] = x[i] * d2N_dw2[i][j]
        #         d2yt_dvdx[i][j] = x[i] * d2N_dvdx[i][j] + dN_dv[i][j]
        #         d2yt_dudx[i][j] = x[i] * d2N_dudx[i][j] + dN_du[i][j]
        #         d2yt_dwdx[i][j] = x[i] * d2N_dwdx[i][j] + dN_dw[i][j]
        #         d3yt_dv2dx[i][j] = x[i] * d3N_dv2dx[i][j] + d2N_dv2[i][j]
        #         d3yt_du2dx[i][j] = x[i] * d3N_du2dx[i][j] + d2N_du2[i][j]
        #         d3yt_dw2dx[i][j] = x[i] * d3N_dw2dx[i][j] + d2N_dw2[i][j]
        if debug: print('A =', A)
        if debug: print('P =', P)
        if debug: print('psit =', psit)
        if debug: print('dA_dx =', dA_dx)
        if debug: print('dP_dx =', dP_dx)
        if debug: print('dpsit_dx =', dpsit_dx)
        # if debug: print('dyt_dx =', dyt_dx)
        # if debug: print('dyt_dv =', dyt_dv)
        # if debug: print('dyt_du =', dyt_du)
        # if debug: print('dyt_dw =', dyt_dw)
        # if debug: print('d2yt_dv2 =', d2yt_dv2)
        # if debug: print('d2yt_du2 =', d2yt_du2)
        # if debug: print('d2yt_dw2 =', d2yt_dw2)
        # if debug: print('d2yt_dvdx =', d2yt_dvdx)
        # if debug: print('d2yt_dudx =', d2yt_dudx)
        # if debug: print('d2yt_dwdx =', d2yt_dwdx)
        # if debug: print('d3yt_dv2dx =', d3yt_dv2dx)
        # if debug: print('d3yt_du2dx =', d3yt_du2dx)
        # if debug: print('d3yt_dw2dx =', d3yt_dw2dx)

        # Compute the value of the original 1st derivative function
        # for each training point, and its derivatives.
        # f = np.zeros(ntrain)
        # df_dyt = np.zeros(ntrain)
        # d2f_dyt2 = np.zeros(ntrain)
        # df_dv = np.zeros((ntrain, nhid))
        # df_du = np.zeros((ntrain, nhid))
        # df_dw = np.zeros((ntrain, nhid))
        # d2f_dv2 = np.zeros((ntrain, nhid))
        # d2f_du2 = np.zeros((ntrain, nhid))
        # d2f_dw2 = np.zeros((ntrain, nhid))
        # for i in range(ntrain):
        #     f[i] = F(x[i], yt[i])
        #     df_dyt[i] = dF_dy(x[i], yt[i])
        #     d2f_dyt2[i] = d2F_dy2(x[i], yt[i])
        #     for j in range(nhid):
        #         df_dv[i][j] = df_dyt[i] * dyt_dv[i][j]
        #         df_du[i][j] = df_dyt[i] * dyt_du[i][j]
        #         df_dw[i][j] = df_dyt[i] * dyt_dw[i][j]
        #         d2f_dv2[i][j] = (
        #             df_dyt[i] * d2yt_dv2[i][j] + d2f_dyt2[i] * dyt_dv[i][j]**2
        #         )
        #         d2f_du2[i][j] = (
        #             df_dyt[i] * d2yt_du2[i][j] + d2f_dyt2[i] * dyt_du[i][j]**2
        #         )
        #         d2f_dw2[i][j] = (
        #             df_dyt[i] * d2yt_dw2[i][j] + d2f_dyt2[i] * dyt_dw[i][j]**2
        #         )
        # if debug: print('f =', f)
        # if debug: print('df_dyt =', df_dyt)
        # if debug: print('d2f_dyt2 =', d2f_dyt2)
        # if debug: print('df_dv =', df_dv)
        # if debug: print('df_du =', df_du)
        # if debug: print('df_dw =', df_dw)
        # if debug: print('d2f_dv2 =', d2f_dv2)
        # if debug: print('d2f_du2 =', d2f_du2)
        # if debug: print('d2f_dw2 =', d2f_dw2)

        # Compute the error function for this pass.
        E = 0
        for i in range(ntrain):
            E += G(x[i], dpsit_dx[i])**2
        if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to the
        # network parameters.
        # dE_dv = np.zeros(nhid)
        # dE_du = np.zeros(nhid)
        # dE_dw = np.zeros(nhid)
        # d2E_dv2 = np.zeros(nhid)
        # d2E_du2 = np.zeros(nhid)
        # d2E_dw2 = np.zeros(nhid)
        # for j in range(nhid):
        #     for i in range(ntrain):
        #         dE_dv[j] += (
        #             2 * (dyt_dx[i] - f[i]) * (d2yt_dvdx[i][j] - df_dv[i][j])
        #         )
        #         dE_du[j] += (
        #             2 * (dyt_dx[i] - f[i]) * (d2yt_dudx[i][j] - df_du[i][j])
        #         )
        #         dE_dw[j] += (
        #             2 * (dyt_dx[i] - f[i]) * (d2yt_dwdx[i][j] - df_dw[i][j])
        #         )
        #         d2E_dv2[j] += 2 * (
        #             (dyt_dx[i] - f[i]) * (d3yt_dv2dx[i][j] - d2f_dv2[i][j])
        #             + (d2yt_dvdx[i][j] - df_dv[i][j])**2
        #         )
        #         d2E_du2[j] += 2 * (
        #             (dyt_dx[i] - f[i]) * (d3yt_du2dx[i][j] - d2f_du2[i][j])
        #             + (d2yt_dudx[i][j] - df_du[i][j])**2
        #         )
        #         d2E_dw2[j] += 2 * (
        #             (dyt_dx[i] - f[i]) * (d3yt_dw2dx[i][j] - d2f_dw2[i][j])
        #             + (d2yt_dwdx[i][j] - df_dw[i][j])**2
        #         )
        # if debug: print('dE_dv =', dE_dv)
        # if debug: print('dE_du =', dE_du)
        # if debug: print('dE_dw =', dE_dw)
        # if debug: print('d2E_dv2 =', d2E_dv2)
        # if debug: print('d2E_du2 =', d2E_du2)
        # if debug: print('d2E_dw2 =', d2E_dw2)

        #------------------------------------------------------------------------

        # Update the weights and biases.
    
        # Compute the new values of the network parameters.
        # v_new = np.zeros(nhid)
        # u_new = np.zeros(nhid)
        # w_new = np.zeros(nhid)
        # for j in range(nhid):
        #     v_new[j] = v[j] - eta * dE_dv[j] / d2E_dv2[j]
        #     u_new[j] = u[j] - eta * dE_du[j] / d2E_du2[j]
        #     w_new[j] = w[j] - eta * dE_dw[j] / d2E_dw2[j]
        # if debug: print('v_new =', v_new)
        # if debug: print('u_new =', u_new)
        # if debug: print('w_new =', w_new)

        # if verbose: print(epoch, E)

        # Save the new weights and biases.
        # v = v_new
        # u = u_new
        # w = w_new

    # Return the final solution.
    # return (yt, dyt_dx)
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
                        help = 'Number of evenly-spaced training points to use')
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
    assert pdemod.ya
    assert len(pdemod.dya_dx) > 0
    assert len(pdemod.d2ya_dx2) == len(pdemod.dya_dx)
    assert pdemod.G
    assert len(pdemod.dG_dx) == len(pdemod.dya_dx)
    assert len(pdemod.dG_dy_dx) == len(pdemod.dya_dx)
    assert len(pdemod.d2G_dx2) == len(pdemod.dya_dx)
    assert len(pdemod.d2G_dy_dx2) == len(pdemod.dya_dx)
    assert len(pdemod.bc) == len(pdemod.dya_dx)
    assert len(pdemod.bcd) == len(pdemod.dya_dx)

    # Determine the number of dimensions.
    ndim = len(pdemod.dya_dx)
    if debug: print('ndim =', ndim)

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
    (yt, dyt_dx) = nnpde1(xt, pdemod.G, pdemod.dG_dx, pdemod.dG_dy_dx,
                          pdemod.d2G_dx2, pdemod.d2G_dy_dx2, pdemod.bc,
                          pdemod.bcd,
                          maxepochs = maxepochs, eta = eta, nhid = nhid,
                          debug = debug, verbose = verbose)

    #----------------------------------------------------------------------------

    # Compute the analytical solution at the training points.
    # ya = np.zeros(ntrain)
    # for i in range(ntrain):
    #     ya[i] = odemod.ya(xt[i])
    # if debug: print('ya =', ya)

    # Compute the analytical derivative at the training points.
    # dya_dx = np.zeros(ntrain)
    # for i in range(ntrain):
    #     dya_dx[i] = odemod.dya_dx(xt[i])
    # if debug: print('dya_dx =', dya_dx)

    # Compute the MSE of the trial solution.
    # y_err = yt - ya
    # if debug: print('y_err =', y_err)
    # mse_y = sqrt(sum((yt - ya)**2) / ntrain)
    # if debug: print('mse_y =', mse_y)

    # Compute the MSE of the trial derivative.
    # dy_dx_err = dyt_dx - dya_dx
    # if debug: print('dy_dx_err =', dy_dx_err)
    # mse_dy_dx = sqrt(sum((dyt_dx - dya_dx)**2) / ntrain)
    # if debug: print('mse_dy_dx =', mse_dy_dx)

    # Print the report.
    # print('    xt       yt       ya      dyt_dx    dya_dx')
    # for i in range(ntrain):
    #     print('%f %f %f %f %f' % (xt[i], yt[i], ya[i], dyt_dx[i], dya_dx[i]))
    # print('MSE      %f          %f' % (mse_y, mse_dy_dx))

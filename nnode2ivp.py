#!/usr/bin/env python

# Use a neural network to solve a 2nd-order ODE IVP, with Neumann BC,
# i.e. the value of the solution and its derivative are specified at
# x=0, in solution region [0,1]. Note that any 2nd-order IVP can be
# mapped to a corresponding IVP with initial values at 0, so this is
# the only solution form needed.

# The general form of such equations is:

# G(x, y, dy_dx, dy_dx2) = 0

# Notation notes:

# 0. Notation is developed to mirror my derivations and notes.

# 1. Names that end in 'f' are usually functions, or containers of functions.

# 2. Underscores separate the numerator and denominator in a name
# which represents a derivative.

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
default_ode = 'ode01ivp'
default_seed = 0
default_verbose = False

# Default ranges for weights and biases
v_min = -1
v_max = 1
w_min = -1
w_max = 1
u_min = -1
u_max = 1

#********************************************************************************

# Define the trial solution for a 2nd-order ODE IVP.
def ytrial(A, Ap, x, N):
    return A + x * Ap + N * x**2

# Define the 1st trial derivative.
def dytrial_dx(A, Ap, x, N, dN_dx):
    return Ap + x**2*dN_dx + 2*x*N

# Define the 2nd trial derivative.
def d2ytrial_dx2(A, Ap, x, N, dN_dx, d2N_dx2):
    return x**2*d2N_dx2 + 4*x*dN_dx + 2*N

#********************************************************************************

# Function to solve a 2nd-order ODE IVP using a single-hidden-layer
# feedforward neural network with a single input node and a single
# output node.

def nnode2ivp(
        Gf,             # 1st-order ODE IVP to solve
        ic,             # IC for ODE
        ic1,            # 1st derivative IC for ODE
        dG_dyf,         # Partial of G(x,y,dy/dx) wrt y
        dG_dydxf,       # Partial of G(x,y,dy/dx) wrt dy/dx
        dG_d2ydx2f,     # Partial of G(x,y,dy/dx) wrt d2y/dx2
        x,                             # x-values for training points
        nhid = default_nhid,           # Node count in hidden layer
        maxepochs = default_maxepochs, # Max training epochs
        eta = default_eta,             # Learning rate
        debug = default_debug,
        verbose = default_verbose
):
    if debug: print('Gf =', Gf)
    if debug: print('ic =', ic)
    if debug: print('ic1 =', ic1)
    if debug: print('dG_dyf =', dG_dyf)
    if debug: print('dG_dydxf =', dG_dydxf)
    if debug: print('dG_d2ydx2f =', dG_d2ydx2f)
    if debug: print('x =', x)
    if debug: print('nhid =', nhid)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments.
    assert Gf
    assert ic != None
    assert ic1 != None
    assert dG_dyf
    assert dG_dydxf
    assert dG_d2ydx2f
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

    # Create an array to hold the weights connecting the input node to the
    # hidden nodes. The weights are initialized with a uniform random
    # distribution.
    w = np.random.uniform(w_min, w_max, nhid)
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
    A = ic
    if debug: print('A =', A)
    Ap = ic1
    if debug: print('Ap =', Ap)
    n = ntrain
    if debug: print('n =', n)
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
                z[i][k] = w[k] * x[i] + u[k]
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
        dN_dx = np.zeros(n)
        dN_dv = np.zeros((n, H))
        dN_du = np.zeros((n, H))
        dN_dw = np.zeros((n, H))
        d2N_dvdx = np.zeros((n, H))
        d2N_dudx = np.zeros((n, H))
        d2N_dwdx = np.zeros((n, H))
        d2N_dx2 = np.zeros(n)
        d3N_dvdx2 = np.zeros((n, H))
        d3N_dudx2 = np.zeros((n, H))
        d3N_dwdx2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                N[i] += v[k] * s[i][k]
                dN_dx[i] += v[k] * s1[i][k] * w[k]
                dN_dv[i][k] = s[i][k]
                dN_du[i][k] = v[k] * s1[i][k]
                dN_dw[i][k] = v[k] * s1[i][k] * x[i]
                d2N_dvdx[i][k] = s1[i][k] * w[k]
                d2N_dudx[i][k] = v[k] * s2[i][k] * w[k]
                d2N_dwdx[i][k] = v[k] * (s1[i][k] + s2[i][k] * w[k] * x[i])
                d2N_dx2[i] += v[k] * s2[i][k] * w[k]**2
                d3N_dvdx2[i][k] = s2[i][k] * w[k]**2
                d3N_dudx2[i][k] = v[k] * s3[i][k] * w[k]**2
                d3N_dwdx2[i][k] = (
                    v[k] * (2 * s2[i][k] * w[k] + s3[i][k] * w[k]**2 * x[i])
                )
        if debug: print('N =', N)
        if debug: print('dN_dx =', dN_dx)
        if debug: print('dN_dv =', dN_dv)
        if debug: print('dN_du =', dN_du)
        if debug: print('dN_dw =', dN_dw)
        if debug: print('d2N_dvdx =', d2N_dvdx)
        if debug: print('d2N_dudx =', d2N_dudx)
        if debug: print('d2N_dwdx =', d2N_dwdx)
        if debug: print('d2N_dx2 =', d2N_dx2)
        if debug: print('d3N_dvdx2 =', d3N_dvdx2)
        if debug: print('d3N_dudx2 =', d3N_dudx2)
        if debug: print('d3N_dwdx2 =', d3N_dwdx2)

        #------------------------------------------------------------------------

        # Compute the value of the trial solution and its derivatives
        # for each training point.
        yt = np.zeros(n)
        dyt_dx = np.zeros(n)
        d2yt_dx2 = np.zeros(n)
        dyt_dv = np.zeros((n, H))
        dyt_du = np.zeros((n, H))
        dyt_dw = np.zeros((n, H))
        d2yt_dvdx = np.zeros((n, H))
        d2yt_dudx = np.zeros((n, H))
        d2yt_dwdx = np.zeros((n, H))
        d3yt_dvdx2 = np.zeros((n, H))
        d3yt_dudx2 = np.zeros((n, H))
        d3yt_dwdx2 = np.zeros((n, H))
        for i in range(n):
            yt[i] = ytrial(A, Ap, x[i], N[i])
            dyt_dx[i] = dytrial_dx(A, Ap, x[i], N[i], dN_dx[i])
            d2yt_dx2[i] = d2ytrial_dx2(A, Ap, x[i], N[i], dN_dx[i], d2N_dx2[i])
            for k in range(H):
                dyt_dv[i][k] = x[i]**2*dN_dv[i][k]
                dyt_du[i][k] = x[i]**2*dN_du[i][k]
                dyt_dw[i][k] = x[i]**2*dN_dw[i][k]
                d2yt_dvdx[i][k] = x[i]**2*d2N_dvdx[i][k] + 2*x[i]*dN_dv[i][k]
                d2yt_dudx[i][k] = x[i]**2*d2N_dudx[i][k] + 2*x[i]*dN_du[i][k]
                d2yt_dwdx[i][k] = x[i]**2*d2N_dwdx[i][k] + 2*x[i]*dN_dw[i][k]
                d3yt_dvdx2[i][k] = (
                    x[i]**2*d3N_dvdx2[i][k] + 4*x[i]*d2N_dvdx[i][k]
                    + 2*dN_dv[i][k]
                )
                d3yt_dudx2[i][k] = (
                    x[i]**2*d3N_dudx2[i][k] + 4*x[i]*d2N_dudx[i][k]
                    + 2*dN_du[i][k]
                )
                d3yt_dwdx2[i][k] = (
                    x[i]**2*d3N_dwdx2[i][k] + 4*x[i]*d2N_dwdx[i][k]
                    + 2*dN_dw[i][k]
                )
        if debug: print('yt =', yt)
        if debug: print('dyt_dx =', dyt_dx)
        if debug: print('d2yt_dx2 =', d2yt_dx2)
        if debug: print('dyt_dv =', dyt_dv)
        if debug: print('dyt_du =', dyt_du)
        if debug: print('dyt_dw =', dyt_dw)
        if debug: print('d2yt_dvdx =', d2yt_dvdx)
        if debug: print('d2yt_dudx =', d2yt_dudx)
        if debug: print('d2yt_dwdx =', d2yt_dwdx)
        if debug: print('d3yt_dvdx2 =', d3yt_dvdx2)
        if debug: print('d3yt_dudx2 =', d3yt_dudx2)
        if debug: print('d3yt_dwdx2 =', d3yt_dwdx2)

                # Compute the value of the original differential equation for
        # each training point, and its derivatives.
        G = np.zeros(n)
        dG_dyt = np.zeros(n)
        dG_dytdx = np.zeros(n)
        dG_d2ytdx2 = np.zeros(n)
        dG_dv = np.zeros((n, H))
        dG_du = np.zeros((n, H))
        dG_dw = np.zeros((n, H))
        for i in range(n):
            G[i] = Gf(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            dG_dyt[i] = dG_dyf(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            dG_dytdx[i] = dG_dydxf(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            dG_d2ytdx2[i] = dG_d2ydx2f(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            for k in range(H):
                dG_dv[i][k] = (
                    dG_dyt[i]*dyt_dv[i][k] + dG_dytdx[i]*d2yt_dvdx[i][k]
                    + dG_d2ytdx2[i]*d3yt_dvdx2[i][k]
                )
                dG_du[i][k] = (
                    dG_dyt[i]*dyt_du[i][k] + dG_dytdx[i]*d2yt_dudx[i][k]
                    + dG_d2ytdx2[i]*d3yt_dudx2[i][k]
                )
                dG_dw[i][k] = (
                    dG_dyt[i]*dyt_dw[i][k] + dG_dytdx[i]*d2yt_dwdx[i][k]
                    + dG_d2ytdx2[i]*d3yt_dwdx2[i][k]
                )
        if debug: print('G =', G)
        if debug: print('dG_dyt =', dG_dyt)
        if debug: print('dG_dytdx =', dG_dytdx)
        if debug: print('dG_d2ytdx2 =', dG_d2ytdx2)
        if debug: print('dG_dv =', dG_dv)
        if debug: print('dG_du =', dG_du)
        if debug: print('dG_dw =', dG_dw)

        # Compute the error function for this pass.
        E = 0
        for i in range(n):
            E += G[i]**2
        if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to
        # the network parameters.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros(H)
        for j in range(H):
            for i in range(n):
                dE_dv[k] += 2 * G[i] * dG_dv[i][k]
                dE_du[k] += 2 * G[i] * dG_du[i][k]
                dE_dw[k] += 2 * G[i] * dG_dw[i][k]
        if debug: print('dE_dv =', dE_dv)
        if debug: print('dE_du =', dE_du)
        if debug: print('dE_dw =', dE_dw)

        #------------------------------------------------------------------------

        # Update the weights and biases.
    
        # Compute the new values of the network parameters.
        v_new = np.zeros(H)
        u_new = np.zeros(H)
        w_new = np.zeros(H)
        for j in range(H):
            v_new[j] = v[j] - eta * dE_dv[j]
            u_new[j] = u[j] - eta * dE_du[j]
            w_new[j] = w[j] - eta * dE_dw[j]
        if debug: print('v_new =', v_new)
        if debug: print('u_new =', u_new)
        if debug: print('w_new =', w_new)

        if verbose: print(epoch, E)

        # Save the new weights and biases.
        v = v_new
        u = u_new
        w = w_new

    # Return the final solution.
    return (yt, dyt_dx, d2yt_dx2)

#--------------------------------------------------------------------------------

# Begin main program.

if __name__ == '__main__':

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description =
        'Solve a 2nd-order ODE IVP with Neumann BC with a neural net',
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
                        help = 'Number of evenly-spaced training points to use')
    parser.add_argument('--ode', type = str,
                        default = default_ode,
                        help = 'Name of module containing ODE to solve')
    parser.add_argument('--seed', type = int,
                        default = default_seed,
                        help = 'Random number generator seed')
    parser.add_argument('--verbose', '-v',
                        action = 'store_true',
                        default = default_verbose,
                        help = 'Produce verbose output')
    parser.add_argument('--version',
                        action = 'version',
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
    ode = args.ode
    seed = args.seed
    verbose = args.verbose
    if debug: print('debug =', debug)
    if debug: print('eta =', eta)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('nhid =', nhid)
    if debug: print('ntrain =', ntrain)
    if debug: print('ode =', ode)
    if debug: print('seed =', seed)
    if debug: print('verbose =', verbose)

    # Perform basic sanity checks on the command-line options.
    assert eta > 0
    assert maxepochs > 0
    assert nhid > 0
    assert ntrain > 0
    assert ode
    assert seed >= 0

    #----------------------------------------------------------------------------

    # Initialize the random number generator to ensure repeatable results.
    if verbose: print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the specified ODE module.
    if verbose:
        print('Importing ODE module %s.' % ode)
    odemod = importlib.import_module(ode)
    assert odemod.Gf
    assert odemod.ic != None
    assert odemod.ic1 != None
    assert odemod.dG_dyf
    assert odemod.dG_dydxf
    assert odemod.dG_d2ydx2f
    assert odemod.yaf
    assert odemod.dya_dxf
    assert odemod.d2ya_dx2f

    # Create the array of evenly-spaced training points.
    if verbose: print('Computing training points in domain[0,1].')
    dx = 1 / (ntrain - 1)
    if debug: print('dx =', dx)
    xt = [i * dx for i in range(ntrain)]
    if debug: print('xt =', xt)

    #----------------------------------------------------------------------------

    # Compute the 2nd-order ODE solution using the neural network.
    (yt, dyt_dx, d2yt_dx2) = nnode2ivp(
        odemod.Gf,             # 1st-order ODE IVP to solve
        odemod.ic,             # IC for ODE
        odemod.ic1,            # 1st derivative IC for ODE
        odemod.dG_dyf,         # Partial of G(x,y,dy/dx) wrt y
        odemod.dG_dydxf,       # Partial of G(x,y,dy/dx) wrt dy/dx
        odemod.dG_d2ydx2f,     # Partial of G(x,y,dy/dx) wrt d2y/dx2
        xt,                    # x-values for training points
        nhid = nhid,           # Node count in hidden layer
        maxepochs = maxepochs, # Max training epochs
        eta = eta,             # Learning rate
        debug = debug,
        verbose = verbose
    )

    #----------------------------------------------------------------------------

    # Compute the analytical solution at the training points.
    ya = np.zeros(ntrain)
    for i in range(ntrain):
        ya[i] = odemod.yaf(xt[i])
    if debug: print('ya =', ya)

    # Compute the 1st analytical derivative at the training points.
    dya_dx = np.zeros(ntrain)
    for i in range(ntrain):
        dya_dx[i] = odemod.dya_dxf(xt[i])
    if debug: print('dya_dx =', dya_dx)

    # Compute the 2nd analytical derivative at the training points.
    d2ya_dx2 = np.zeros(ntrain)
    for i in range(ntrain):
        d2ya_dx2[i] = odemod.d2ya_dx2f(xt[i])
    if debug: print('d2ya_dx2 =', d2ya_dx2)

    # Compute the RMS error of the trial solution.
    y_err = yt - ya
    if debug: print('y_err =', y_err)
    rmse_y = sqrt(sum(y_err**2) / ntrain)
    if debug: print('rmse_y =', rmse_y)

    # Compute the RMS error of the 1st trial derivative.
    dy_dx_err = dyt_dx - dya_dx
    if debug: print('dy_dx_err =', dy_dx_err)
    rmse_dy_dx = sqrt(sum(dy_dx_err**2) / ntrain)
    if debug: print('rmse_dy_dx =', rmse_dy_dx)

    # Compute the RMS error of the 2nd trial derivative.
    d2y_dx2_err = d2yt_dx2 - d2ya_dx2
    if debug: print('d2y_dx2_err =', d2y_dx2_err)
    rmse_d2y_dx2 = sqrt(sum(d2y_dx2_err**2) / ntrain)
    if debug: print('rmse_d2y_dx2 =', rmse_d2y_dx2)

    # Print the report.
    print('    xt       yt       ya     dyt_dx   dya_dx  d2yt_dx2  d2ya_dx2')
    for i in range(ntrain):
        print('%f %f %f %f %f %f %f' %
              (xt[i], yt[i], ya[i],
               dyt_dx[i], dya_dx[i],
               d2yt_dx2[i], d2ya_dx2[i]))
    print('RMSE     %f          %f          %f' %
          (rmse_y, rmse_dy_dx, rmse_d2y_dx2))

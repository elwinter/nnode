#!/usr/bin/env python

# Use a neural network to solve a 2nd-order ODE BVP, with 2 Dirichlet
# BC. Note that any 2nd-order ODE BVP with Dirichlet BC can be mapped
# to a corresponding BVP of this type, so this is the only solution
# form needed.

# The general form of such equations is:

# G(x, y, dy_dx, d2y_dx2) = 0

# Notation notes:

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
default_clamp = False
default_debug = False
default_eta = 0.01
default_maxepochs = 1000
default_nhid = 10
default_ntest = 10
default_ntrain = 10
default_ode = 'ode01bvp'
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

# The domain of the trial solution is assumed to be [0, 1].

# Define the trial solution for a 2nd-order ODE BVP.
def ytf(A, B, x, N):
    return A*(1 - x) + B*x + x*(1 - x)*N

# Define the 1st trial derivative.
def dyt_dxf(A, B, x, N, dN_dx):
    return -A + B + x*(1 - x)*dN_dx + (1 - 2*x)*N

# Define the 2nd trial derivative.
def d2yt_dx2f(A, B, x, N, dN_dx, d2N_dx2):
    return x*(1 - x)*d2N_dx2 + 2*(1 - 2*x)*dN_dx - 2*N

#********************************************************************************

# Function to solve a 2nd-order ODE BVP using a single-hidden-layer
# feedforward neural network.

def nnode2bvp(
        Gf,                            # 2nd-order ODE BVP to solve
        bc0,                           # BC at x=0
        bc1,                           # BC at x=1
        dG_dyf,                        # Partial of G(x,y,dy/dx) wrt y
        dG_dydxf,                      # Partial of G(x,y,dy/dx) wrt dy/dx
        dG_d2ydx2f,                    # Partial of G(x,y,dy/dx) wrt d2y/dx2
        x,                             # x-values for training points
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
        rmseout = default_rmseout,     # Output file for ODE RMS error
        debug = default_debug,
        verbose = default_verbose
):
    if debug: print('Gf =', Gf)
    if debug: print('bc0 =', bc0)
    if debug: print('bc1 =', bc1)
    if debug: print('dG_dyf =', dG_dyf)
    if debug: print('dG_dydxf =', dG_dydxf)
    if debug: print('dG_d2ydx2f =', dG_d2ydx2f)
    if debug: print('x =', x)
    if debug: print('nhid =', nhid)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('clamp =', clamp)
    if debug: print('randomize =', randomize)
    if debug: print('rmseout =', rmseout)
    if debug: print('vmin =', vmin)
    if debug: print('vmax =', vmax)
    if debug: print('wmin =', wmin)
    if debug: print('wmax =', wmax)
    if debug: print('umin =', umin)
    if debug: print('umax =', umax)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments.
    assert Gf
    assert bc0 != None
    assert bc1 != None
    assert dG_dyf
    assert dG_dydxf
    assert dG_d2ydx2f
    assert len(x) > 0
    assert nhid > 0
    assert maxepochs > 0
    assert eta > 0
    assert rmseout
    assert vmin < vmax
    assert wmin < wmax
    assert umin < umax

    #----------------------------------------------------------------------------

    # Determine the number of training points.
    n = len(x)
    if debug: print('n =', n)

    # Change notation for convenience.
    A = bc0
    if debug: print('A =', A)
    B = bc1
    if debug: print('B =', B)
    H = nhid
    if debug: print('H =', H)

    #----------------------------------------------------------------------------

    # Create the network.

    # Create an array to hold the weights connecting the input node to the
    # hidden nodes. The weights are initialized with a uniform random
    # distribution.
    w = np.random.uniform(wmin, wmax, H)
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
    w_history = np.zeros((maxepochs, H))
    u_history = np.zeros((maxepochs, H))
    v_history = np.zeros((maxepochs, H))

    #----------------------------------------------------------------------------
    # Vectorize the functions used by the network.
    sigma_v = np.vectorize(sigma)
    dsigma_dz_v = np.vectorize(dsigma_dz)
    d2sigma_dz2_v = np.vectorize(d2sigma_dz2)
    d3sigma_dz3_v = np.vectorize(d3sigma_dz3)

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

        # Compute the input, the sigmoid function and its derivatives,
        # for each hidden node.
        z = np.outer(x, w) + u
        s = sigma_v(z)
        s1 = dsigma_dz_v(z)
        s2 = d2sigma_dz2_v(z)
        s3 = d3sigma_dz3_v(z)
        if debug: print('z =', z)
        if debug: print('s =', s)
        if debug: print('s1 =', s1)
        if debug: print('s2 =', s2)
        if debug: print('s3 =', s3)

        # Compute the network output and its derivatives, for each
        # training point.
        N = s.dot(v)
        dN_dx = s1.dot(v*w)
        dN_dv = s
        dN_du = s1*v
        dN_dw = s1*np.outer(x, v)
        d2N_dvdx = s1*w
        d2N_dudx = v*s2*w
        d2N_dwdx = v*(s1 + s2*np.outer(x, w))
        d2N_dx2 = s2.dot(v*w**2)
        d3N_dvdx2 = s2*w**2
        d3N_dudx2 = s3*v*w**2
        d3N_dwdx2 = v*(2*s2*w + s3*np.outer(x, w**2))
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
            yt[i] = ytf(A, B, x[i], N[i])
            dyt_dx[i] = dyt_dxf(A, B, x[i], N[i], dN_dx[i])
            d2yt_dx2[i] = d2yt_dx2f(A, B, x[i], N[i], dN_dx[i], d2N_dx2[i])
            for k in range(H):
                dyt_dv[i,k] = x[i]*(1 - x[i])*dN_dv[i,k]
                dyt_du[i,k] = x[i]*(1 - x[i])*dN_du[i,k]
                dyt_dw[i,k] = x[i]*(1 - x[i])*dN_dw[i,k]
                d2yt_dvdx[i,k] = (
                    x[i]*(1 - x[i])*d2N_dvdx[i,k] + (1 - 2*x[i])*dN_dv[i,k]
                )
                d2yt_dudx[i,k] = (
                    x[i]*(1 - x[i])*d2N_dudx[i,k] + (1 - 2*x[i])*dN_du[i,k]
                )
                d2yt_dwdx[i,k] = (
                    x[i]*(1 - x[i])*d2N_dwdx[i,k] + (1 - 2*x[i])*dN_dw[i,k]
                )
                d3yt_dvdx2[i,k] = (
                    x[i]*(1 - x[i])*d3N_dvdx2[i,k]
                    + 2*(1 - 2*x[i])*d2N_dvdx[i,k]
                    - 2*dN_dv[i,k]
                )
                d3yt_dudx2[i,k] = (
                    x[i]*(1 - x[i])*d3N_dudx2[i,k]
                    + 2*(1 - 2*x[i])*d2N_dudx[i,k]
                    - 2*dN_du[i,k]
                )
                d3yt_dwdx2[i,k] = (
                    x[i]*(1 - x[i])*d3N_dwdx2[i,k]
                    + 2*(1 - 2*x[i])*d2N_dwdx[i,k]
                    - 2*dN_dw[i,k]
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
        dG_dv = np.zeros((n, H))
        dG_du = np.zeros((n, H))
        dG_dw = np.zeros((n, H))
        dG_d2ytdx2 = np.zeros(n)
        for i in range(n):
            G[i] = Gf(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            dG_dyt[i] = dG_dyf(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            dG_dytdx[i] = dG_dydxf(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            dG_d2ytdx2[i] = dG_d2ydx2f(x[i], yt[i], dyt_dx[i], d2yt_dx2[i])
            for k in range(H):
                dG_dv[i,k] = (
                    dG_dyt[i]*dyt_dv[i,k] + dG_dytdx[i]*d2yt_dvdx[i,k]
                    + dG_d2ytdx2[i]*d3yt_dvdx2[i,k]
                )
                dG_du[i,k] = (
                    dG_dyt[i]*dyt_du[i,k] + dG_dytdx[i]*d2yt_dudx[i,k]
                    + dG_d2ytdx2[i]*d3yt_dudx2[i,k]
                )
                dG_dw[i,k] = (
                    dG_dyt[i]*dyt_dw[i,k] + dG_dytdx[i]*d2yt_dwdx[i,k]
                    + dG_d2ytdx2[i]*d3yt_dwdx2[i,k]
                )
        if debug: print('G =', G)
        if debug: print('dG_dyt =', dG_dyt)
        if debug: print('dG_dytdx =', dG_dytdx)
        if debug: print('dG_dv =', dG_dv)
        if debug: print('dG_du =', dG_du)
        if debug: print('dG_dw =', dG_dw)
        if debug: print('dG_d2ytdx2 =', dG_d2ytdx2)

        # Compute the error function for this epoch.
        E = sum(G**2)
        if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to
        # the network parameters.
        dE_dv = np.zeros(H)
        dE_du = np.zeros(H)
        dE_dw = np.zeros(H)
        for k in range(H):
            for i in range(n):
                dE_dv[k] += 2*G[i]*dG_dv[i,k]
                dE_du[k] += 2*G[i]*dG_du[i,k]
                dE_dw[k] += 2*G[i]*dG_dw[i,k]
        if debug: print('dE_dv =', dE_dv)
        if debug: print('dE_du =', dE_du)
        if debug: print('dE_dw =', dE_dw)

        #------------------------------------------------------------------------

        # Compute the new values of the network parameters.
        v_new = np.zeros(H)
        u_new = np.zeros(H)
        w_new = np.zeros(H)
        for k in range(H):
            v_new[k] = v[k] - eta*dE_dv[k]
            u_new[k] = u[k] - eta*dE_du[k]
            w_new[k] = w[k] - eta*dE_dw[k]
        if debug: print('v_new =', v_new)
        if debug: print('u_new =', u_new)
        if debug: print('w_new =', w_new)

        # Clamp the values at the limits.
        if clamp:
            w_new[w_new < wmin] = wmin
            w_new[w_new > wmax] = wmax
            u_new[u_new < umin] = umin
            u_new[u_new > umax] = umax
            v_new[v_new < vmin] = vmin
            v_new[v_new > vmax] = vmax

        # Record the current RMSE.
        rmse = sqrt(E/n)
        rmse_history[epoch] = rmse
        if verbose: print(epoch, rmse)

        # Save the new weights and biases.
        v = v_new
        u = u_new
        w = w_new

    # Save the error and parameter history.
    np.savetxt(rmseout, rmse_history)
    np.savetxt('w.dat', w_history)
    np.savetxt('v.dat', v_history)
    np.savetxt('u.dat', u_history)

    # Return the final solution.
    return (yt, dyt_dx, d2yt_dx2, v, u, w)

#--------------------------------------------------------------------------------

# Run the network using the specified parameters.

def run_net(v, w, u, x):

    # Compute the input to and output from each hidden node.
    z = w*x + u
    s = np.vectorize(sigma)(z)

    # Compute the network output.
    N = np.dot(v, s)

    return N

#--------------------------------------------------------------------------------

# Run the derivative network using the specified parameters.

def run_derivative_net(v, w, u, x):

    # Compute the input to and output from each hidden node.
    z = w*x + u
    s = np.vectorize(sigma)(z)

    # Compute the network output.
    dN_dx = s.dot(v*w)

    return dN_dx

#--------------------------------------------------------------------------------

# Run the 2nd derivative network using the specified parameters.

# def run_derivative2_net(v, w, u, x):

#     DIE HERE
    
#     # Compute the input to and output from each hidden node.
#     z = w*x + u
#     s2 = np.vectorize(dsigma_dz)(z)

#     # Compute the network output.
#     dN_dx = s.dot(v*w)

#     return dN_dx

#--------------------------------------------------------------------------------

def create_argument_parser():

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description = 'Solve a 2nd-order ODE BVP with a neural net',
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
    parser.add_argument('--maxepochs', type = int,
                        default = default_maxepochs,
                        help = 'Maximum number of training epochs')
    parser.add_argument('--nhid', type = int,
                        default = default_nhid,
                        help = 'Number of hidden-layer nodes to use')
    parser.add_argument('--ntest', type = int,
                        default = default_ntest,
                        help = 'Number of evenly-spaced test points to use')
    parser.add_argument('--ntrain', type = int,
                        default = default_ntrain,
                        help = 'Number of evenly-spaced training points to use')
    parser.add_argument('--ode', type = str,
                        default = default_ode,
                        help = 'Name of module containing ODE to solve')
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

    # Create the argument parser.
    parser = create_argument_parser()
    assert parser

    # Fetch and process the arguments from the command line.
    args = parser.parse_args()
    if args.debug: print('args =', args)

    # Extract the processed options.
    clamp = args.clamp
    debug = args.debug
    eta = args.eta
    maxepochs = args.maxepochs
    nhid = args.nhid
    ntest = args.ntest
    ntrain = args.ntrain
    ode = args.ode
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
    if debug: print('maxepochs =', maxepochs)
    if debug: print('nhid =', nhid)
    if debug: print('ntest =', ntest)
    if debug: print('ntrain =', ntrain)
    if debug: print('ode =', ode)
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
    assert maxepochs > 0
    assert nhid > 0
    assert ntest > 0
    assert ntrain > 0
    assert ode
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

    # Import the specified ODE module.
    if verbose:
        print('Importing ODE module %s.' % ode)
    odemod = importlib.import_module(ode)
    assert odemod.Gf           # Function for the ODE as a whole
    assert odemod.bc0 != None  # Boundary condition y(0)
    assert odemod.bc1 != None  # Boundary condition y(1)
    assert odemod.dG_dyf       # Function for derivative of G wrt y
    assert odemod.dG_dydxf     # Function for derivative of G wrt dy/dx
    assert odemod.dG_d2ydx2f   # Function for derivative of G wrt d2y/dx2
    assert odemod.yaf          # Function for analytical solution ya
    assert odemod.dya_dxf      # Function for analytical derivative dya/dx
    assert odemod.d2ya_dx2f    # Function for analytical derivative d2ya/dx2

    # Create the array of evenly-spaced training points.
    if verbose:
        print('Computing training points in domain [0,1].')
    x_train = np.linspace(0, 1, ntrain)
    if debug: print('x_train =', x_train)

    #----------------------------------------------------------------------------

    # Compute the 2nd-order ODE solution using the neural network.
    (yt_train, dyt_dx_train, d2yt_dx2_train, v, u, w) = nnode2bvp(
        odemod.Gf,             # 2nd-order ODE BVP to solve
        odemod.bc0,            # BC at x=0
        odemod.bc1,            # BC at x=1
        odemod.dG_dyf,         # Partial of G(x,y,dy/dx,d2y_dx2) wrt y
        odemod.dG_dydxf,       # Partial of G(x,y,dy/dx,d2y_dx2) wrt dy/dx
        odemod.dG_d2ydx2f,     # Partial of G(x,y,dy/dx,d2y_dx2) wrt d2y/dx2
        x_train,               # x-values for training points
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
        rmseout = rmseout,     # Output file for ODE RMS error
        debug = debug,
        verbose = verbose
    )

    #----------------------------------------------------------------------------

    # debug = True

    # Compute the analytical solution at the training points.
    ya_train = np.vectorize(odemod.yaf)(x_train)
    if debug: print('ya_train =', ya_train)

    # Compute the RMS error of the solution at the training points.
    rmse_y_train = sqrt(np.sum((yt_train - ya_train)**2)/ntrain)
    if debug: print('rmse_y_train =', rmse_y_train)
    print(rmse_y_train)

    # Compute the analytical 1st derivative at the training points.
    # dya_dx_train = np.vectorize(odemod.dya_dxf)(x_train)
    # if debug: print('dya_dx_train =', dya_dx_train)

    # Compute the RMS error of the 1st derivative at the training points.
    # rmse_dy_dx_train = sqrt(np.sum((dyt_dx_train - dya_dx_train)**2)/ntrain)
    # if debug: print('rmse_dy_dx_train =', rmse_dy_dx_train)

    # Compute the analytical 2nd derivative at the training points.
    # d2ya_dx2_train = np.vectorize(odemod.d2ya_dx2f)(x_train)
    # if debug: print('d2ya_dx2_train =', d2ya_dx2_train)

    # Compute the RMS error of the 2nd derivative at the training points.
    # rmse_d2y_dx2_train = sqrt(np.sum((d2yt_dx2_train - d2ya_dx2_train)**2)
    #                           /ntrain)
    # if debug: print('rmse_d2y_dx2_train =', rmse_d2y_dx2_train)

    # Save the trained and analytical values at the training points.
    # np.savetxt(trainout,
    #            list(zip(x_train, yt_train, ya_train,
    #                     dyt_dx_train, dya_dx_train,
    #                     d2yt_dx2_train, d2ya_dx2_train)),
    #            header = 'x_train yt_train ya_train dyt_dx_train dya_dx_train d2yt_dx2_train d2ya_dx2_train',
    #            fmt = '%.6E'
    # )

    #----------------------------------------------------------------------------

    # Compute the array of test points.
    # x_test = np.linspace(0, 1, ntest)
    # if debug: print('x_test =', x_test)

    # Compute the trained and analytical solution at the test points.
    # yt_test = np.zeros(ntest)
    # ya_test = np.zeros(ntest)
    # dyt_dx_test = np.zeros(ntest)
    # dya_dx_test = np.zeros(ntest)
    # A = odemod.bc0
    # B = odemod.bc1
    # for i, x in enumerate(x_test):
    #     N = run_net(v, w, u, x)
    #     dN_dx = run_derivative_net(v, w, u, x)
    #     # yt_test[i] = ytf(A, x, N)
        # ya_test[i] = odemod.yaf(x)
        # dyt_dx_test[i] = dyt_dxf(x, N, dN_dx)
        # dya_dx_test[i] = odemod.dya_dxf(x)
    # if debug: print('yt_test =', yt_test)
    # if debug: print('ya_test =', ya_test)
    # if debug: print('dyt_dx_test =', dyt_dx_test)
    # if debug: print('dya_dx_test =', dya_dx_test)

    # Save the trained and analytical values at the test points.
    # np.savetxt(testout,
    #            list(zip(x_test, yt_test, ya_test, dyt_dx_test, dya_dx_test)),
    #            header = 'x_test yt_test ya_test dyt_dx_test dya_dx_test',
    #            fmt = '%.6E'
    # )

    # Compute the RMS error of the solution at the test points.
    # rmse_y_test = sqrt(np.sum((yt_test - ya_test)**2)/ntest)
    # if debug: print('rmse_y_test =', rmse_y_test)

    # Compute the RMS error of the derivative at the test points.
    # rmse_dy_dx_test = sqrt(np.sum((dyt_dx_test - dya_dx_test)**2)/ntest)
    # if debug: print('rmse_dy_dx_test =', rmse_dy_dx_test)

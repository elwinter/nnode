#!/usr/bin/env python

# Use a neural network to solve a 1st-order ODE IVP. Note that any
# 1st-order BVP can be mapped to a corresponding IVP with initial
# value at 0, so this is the only solution form needed.

# The general form of such equations is:

# G(x, y, dy_dx) = 0

# Notation notes:

# 1. Names that end in 'f' are usually functions, or containers of functions.

# 2. Underscores separate the numerator and denominator in a name
# which represents a derivative.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
from importlib import import_module
from math import sqrt
import numpy as np
from sigma import sigma, dsigma_dz, d2sigma_dz2

#********************************************************************************

# Default values for program parameters
default_clamp = False
default_debug = False
default_eta = 0.01
default_maxepochs = 1000
default_nhid = 10
default_ntest = 10
default_ntrain = 10
default_ode = 'ode00'
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

# Define the trial solution for a 1st-order ODE IVP.
def ytf(A, x, N):
    return A + x*N

# Define the 1st trial derivative.
def dyt_dxf(x, N, dN_dx):
    return x*dN_dx + N

#********************************************************************************

# Function to solve a 1st-order ODE IVP using a single-hidden-layer
# feedforward neural network.

def nnode1(
        Gf,                            # 1st-order ODE IVP to solve
        ic,                            # IC for ODE
        dG_dyf,                        # Partial of G(x,y,dy/dx) wrt x
        dG_dydxf,                      # Partial of G(x,y,dy/dx) wrt x
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
    if debug: print('ic =', ic)
    if debug: print('dG_dyf =', dG_dyf)
    if debug: print('dG_dydxf =', dG_dydxf)
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
    assert ic != None
    assert dG_dyf
    assert dG_dydxf
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
    A = ic
    if debug: print('A =', A)
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
    ytf_v = np.vectorize(ytf)
    dyt_dxf_v = np.vectorize(dyt_dxf)
    Gf_v = np.vectorize(Gf)
    dG_dyf_v = np.vectorize(dG_dyf)
    dG_dydxf_v = np.vectorize(dG_dydxf)

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
        if debug: print('z =', z)
        if debug: print('s =', s)
        if debug: print('s1 =', s1)
        if debug: print('s2 =', s2)

        # Compute the network output and its derivatives, for each
        # training point.
        N = s.dot(v)
        dN_dx = s.dot(v*w)
        dN_dv = s
        dN_du = s1*v
        dN_dw = s1*np.outer(x,v)
        d2N_dvdx = s1*w
        d2N_dudx = v*s2*w
        d2N_dwdx = v*(s1 + s2*np.outer(x,w))
        if debug: print('N =', N)
        if debug: print('dN_dx =', dN_dx)
        if debug: print('dN_dv =', dN_dv)
        if debug: print('dN_du =', dN_du)
        if debug: print('dN_dw =', dN_dw)
        if debug: print('d2N_dvdx =', d2N_dvdx)
        if debug: print('d2N_dudx =', d2N_dudx)
        if debug: print('d2N_dwdx =', d2N_dwdx)

        #------------------------------------------------------------------------

        # Compute the value of the trial solution and its derivatives,
        # for each training point.
        yt = ytf_v(A, x, N)
        dyt_dx = dyt_dxf_v(x, N, dN_dx)
        dyt_dv = np.broadcast_to(x, (H, n)).T*dN_dv
        dyt_du = np.broadcast_to(x, (H, n)).T*dN_du
        dyt_dw = np.broadcast_to(x, (H, n)).T*dN_dw
        d2yt_dvdx = np.broadcast_to(x, (H, n)).T*d2N_dvdx + dN_dv
        d2yt_dudx = np.broadcast_to(x, (H, n)).T*d2N_dudx + dN_du
        d2yt_dwdx = np.broadcast_to(x, (H, n)).T*d2N_dwdx + dN_dw
        if debug: print('yt =', yt)
        if debug: print('dyt_dx =', dyt_dx)
        if debug: print('dyt_dv =', dyt_dv)
        if debug: print('dyt_du =', dyt_du)
        if debug: print('dyt_dw =', dyt_dw)
        if debug: print('d2yt_dvdx =', d2yt_dvdx)
        if debug: print('d2yt_dudx =', d2yt_dudx)
        if debug: print('d2yt_dwdx =', d2yt_dwdx)

        # Compute the value of the original differential equation for
        # each training point, and its derivatives.
        G = Gf_v(x, yt, dyt_dx)
        dG_dyt = dG_dyf_v(x, yt, dyt_dx)
        dG_dytdx = dG_dydxf_v(x, yt, dyt_dx)
        dG_dv = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dv + \
                np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dvdx
        dG_du = np.broadcast_to(dG_dyt, (H, n)).T*dyt_du + \
                np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dudx
        dG_dw = np.broadcast_to(dG_dyt, (H, n)).T*dyt_dw + \
                np.broadcast_to(dG_dytdx, (H, n)).T*d2yt_dwdx
        if debug: print('G =', G)
        if debug: print('dG_dyt =', dG_dyt)
        if debug: print('dG_dytdx =', dG_dytdx)
        if debug: print('dG_dv =', dG_dv)
        if debug: print('dG_du =', dG_du)
        if debug: print('dG_dw =', dG_dw)

        # Compute the error function for this epoch.
        E = np.sum(G**2)
        if debug: print('E =', E)

        # Compute the partial derivatives of the error with respect to the
        # network parameters.
        dE_dv = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dv, axis = 0)
        dE_du = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_du, axis = 0)
        dE_dw = 2*np.sum(np.broadcast_to(G, (H, n)).T*dG_dw, axis = 0)
        if debug: print('dE_dv =', dE_dv)
        if debug: print('dE_du =', dE_du)
        if debug: print('dE_dw =', dE_dw)

        #------------------------------------------------------------------------
   
        # Compute the new values of the network parameters.
        v_new = v - eta*dE_dv
        u_new = u - eta*dE_du
        w_new = w - eta*dE_dw
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

    # Return the final solution, and the network parameters.
    return (yt, dyt_dx, v, w, u)

#--------------------------------------------------------------------------------

# Run the network using the specified parameters.

def run(v, w, u, x):

    # Compute the input to and output from each hidden node.
    z = w*x + u
    s = np.vectorize(sigma)(z)

    # Compute the network output.
    N = np.dot(v, s)

    return N

#--------------------------------------------------------------------------------

def create_argument_parser():

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description = 'Solve a 1st-order ODE IVP with a neural net',
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
    assert vmin < vmax
    assert wmin < wmax
    assert umin < umax

    #----------------------------------------------------------------------------

    # Initialize the random number generator to ensure repeatable results.
    if verbose: print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the specified ODE module.
    if verbose:
        print('Importing ODE module %s.' % ode)
    odemod = import_module(ode)
    assert odemod.Gf
    assert odemod.ic != None
    assert odemod.dG_dyf
    assert odemod.dG_dydxf
    assert odemod.yaf
    assert odemod.dya_dxf

    # Create the array of evenly-spaced training points.
    if verbose: print('Computing training points in domain [0,1].')
    xt = np.linspace(0, 1, ntrain)
    if debug: print('xt =', xt)

    #----------------------------------------------------------------------------

    # Compute the 1st-order ODE solution using the neural network.
    (yt, dyt_dx, v, w, u) = nnode1(
        odemod.Gf,             # 1st-order ODE IVP to solve
        odemod.ic,             # IC for ODE
        odemod.dG_dyf,         # Partial of G(x,y,dy/dx) wrt y
        odemod.dG_dydxf,       # Partial of G(x,y,dy/dx) wrt dy/dx
        xt,                    # x-values for training points
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

    # Compute the analytical solution at the training points.
    ya = np.vectorize(odemod.yaf)(xt)
    if debug: print('ya =', ya)

    # Compute the analytical derivative at the training points.
    dya_dx = np.vectorize(odemod.dya_dxf)(xt)
    if debug: print('dya_dx =', dya_dx)

    # Compute the RMS error of the trial solution.
    y_err = yt - ya
    if debug: print('y_err =', y_err)
    rmse_y = sqrt(np.sum(y_err**2)/ntrain)
    if debug: print('rmse_y =', rmse_y)

    # Compute the RMS error of the trial derivative.
    dy_dx_err = dyt_dx - dya_dx
    if debug: print('dy_dx_err =', dy_dx_err)
    rmse_dy_dx = sqrt(np.sum(dy_dx_err**2)/ntrain)
    if debug: print('rmse_dy_dx =', rmse_dy_dx)

    # Print the report.
    # print('    xt       yt       ya      dyt_dx    dya_dx')
    # for i in range(ntrain):
    #     print('%f %f %f %f %f' % (xt[i], yt[i], ya[i], dyt_dx[i], dya_dx[i]))
    # print('RMSE     %f          %f' % (rmse_y, rmse_dy_dx))
    print(rmse_y)

    # Save the trained and analytical values at the training points.
    np.savetxt(trainout, list(zip(xt, yt, ya)))

    # Compute the value of the analytical and trained solution at the
    # test points.
    xtest = np.linspace(0, 1, ntest)
    ytest = np.zeros(ntest)
    yatest = np.zeros(ntest)
    A = odemod.ic
    for i, x in enumerate(xtest):
        N = run(v, w, u, x)
        ytest[i] = ytf(A, x, N)
        yatest[i] = odemod.yaf(x)

    # Save the trained and analytical values at the test points.
    np.savetxt(testout, list(zip(xtest, ytest, yatest)))

    # Compute the RMS error of the solution at the test points.
    ytest_err = ytest - yatest
    if debug: print('ytest_err =', ytest_err)
    rmse_ytest = sqrt(np.sum(ytest_err**2)/ntest)
    if debug: print('rmse_ytest =', rmse_ytest)
    print(rmse_ytest)

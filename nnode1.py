#!/usr/bin/env python

# Use a neural network to solve a 1st-order ODE IVP. Note that any
# 1st-order BVP can be mapped to a corresponding IVP with initial
# value at 0, so this is the only solution form needed.

# The general form of such equations is:

# G(x,y,dy/dx) = 0

# Notation notes:

# 1. Names that end in 'f' are usually functions, or containers of functions.

# 2. Underscores separate the numerator and denominator in a name
# which represents a derivative.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
from math import sqrt
import numpy as np

from ode1ivp import ODE1IVP
from nnode1ivp import NNODE1IVP

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
    assert umin < umax
    assert vmin < vmax
    assert wmin < wmax

    #----------------------------------------------------------------------------

    # Initialize the random number generator to ensure repeatable results.
    if verbose:
        print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the ODE module.
    if verbose:
        print('Importing ODE module %s.' % ode)
    ode1ivp = ODE1IVP(ode)

    # Create the array of evenly-spaced training points.
    if verbose:
        print('Computing training points in domain [0,1].')
    x_train = np.linspace(0, 1, ntrain)
    if debug: print('x_train =', x_train)

    #----------------------------------------------------------------------------

    # Create and train the network.
    net = NNODE1IVP(ode1ivp)
    net.train(x_train, nhid = nhid, maxepochs = maxepochs, eta = eta,
              clamp = clamp, randomize = randomize,
              wmin = wmin, wmax = wmax, umin = umin, umax = umax,
              vmin = vmin, vmax = vmax,
              rmseout = rmseout, debug = debug, verbose = True)
    if debug: print('net =', net)

    #----------------------------------------------------------------------------

    # Compute the trained solution at the training points.
    yt_train = np.vectorize(net.run)(x_train)
    if debug: print('yt_train =', yt_train)

    # Compute the analytical solution at the training points.
    ya_train = np.vectorize(ode1ivp.yaf)(x_train)
    if debug: print('ya_train =', ya_train)

    # Compute the RMS error of the solution at the training points.
    rmse_y_train = sqrt(np.sum((yt_train - ya_train)**2)/ntrain)
    if debug: print('rmse_y_train =', rmse_y_train)

    # Compute the trained 1st derivative at the training points.
    dyt_dx_train = np.vectorize(net.run_derivative)(x_train)
    if debug: print('dyt_dx_train =', dyt_dx_train)

    # Compute the analytical 1st derivative at the training points.
    dya_dx_train = np.vectorize(ode1ivp.dya_dxf)(x_train)
    if debug: print('dya_dx_train =', dya_dx_train)

    # Compute the RMS error of the 1st derivative at the training points.
    rmse_dy_dx_train = sqrt(np.sum((dyt_dx_train - dya_dx_train)**2)/ntrain)
    if debug: print('rmse_dy_dx_train =', rmse_dy_dx_train)

    # Save the trained and analytical values at the training points.
    np.savetxt(trainout,
               list(zip(x_train, yt_train, ya_train, dyt_dx_train,
                        dya_dx_train)),
               header = 'x_train yt_train ya_train dyt_dx_train dya_dx_train',
               fmt = '%.6E'
    )

    #----------------------------------------------------------------------------

    # Compute the array of test points.
    x_test = np.linspace(0, 1, ntest)
    if debug: print('x_test =', x_test)

    # Compute the trained and analytical solution at the test points.
    yt_test = np.vectorize(net.run)(x_test)
    if debug: print('yt_test =', yt_test)
    ya_test = np.vectorize(ode1ivp.yaf)(x_test)
    if debug: print('ya_test =', ya_test)

    # Compute the final RMS error of the solution at the test points.
    rmse_y_test = sqrt(np.sum((yt_test - ya_test)**2)/ntest)
    if debug: print('rmse_y_test =', rmse_y_test)

    # Compute the trained and analytical 1st derivative at the test points.
    dyt_dx_test = np.vectorize(net.run_derivative)(x_test)
    if debug: print('dyt_dx_test =', dyt_dx_test)
    dya_dx_test = np.vectorize(ode1ivp.dya_dxf)(x_test)
    if debug: print('dya_dx_test =', dya_dx_test)

    # Compute the final RMS error of the derivative at the test points.
    rmse_dy_dx_test = sqrt(np.sum((dyt_dx_test - dya_dx_test)**2)/ntest)
    if debug: print('rmse_dy_dx_test =', rmse_dy_dx_test)

    # Save the trained and analytical values at the test points.
    np.savetxt(testout,
               list(zip(x_test, yt_test, ya_test, dyt_dx_test, dya_dx_test)),
               header = 'x_test yt_test ya_test dyt_dx_test dya_dx_test',
               fmt = '%.6E'
    )

    #----------------------------------------------------------------------------

    # Print the report.
    print('rmse_y_train ', rmse_y_train)
    print('rmse_y_test ', rmse_y_test)
    print('rmse_dy_dx_train ', rmse_dy_dx_train)
    print('rmse_dy_dx_test ', rmse_dy_dx_test)

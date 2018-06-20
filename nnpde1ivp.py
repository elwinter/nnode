#!/usr/bin/env python

# Use a neural network to solve a 2-variable, 1st-order PDE IVP. Note
# that any 2-variable 1st-order PDE BVP can be mapped to a
# corresponding IVP with initial values at 0, so this is the only
# solution form needed.

# The general form of such equations is:

# G(x[], Y, delY[]) = 0

# Notation notes:

# 1. Names that end in 'f' are usually functions, or containers of functions.

# 2. Underscores separate the numerator and denominator in a name
# which represents a derivative.

# 3. Names beginning with 'del' are gradients of another function, in
# the form of function lists.

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
default_ntrain = 10
default_pde = 'pde00'
default_randomize = False
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

# The domain of the trial solution is assumed to be [[0, 1], [0, 1]].

# Define the coefficient functions for the trial solution, and their derivatives.
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

# Define the trial solution.
def Ytf(xy, N, bcf):
    A = Af(xy, bcf)
    P = Pf(xy)
    Yt = A + P*N
    return Yt

#********************************************************************************

# Function to solve a 2-variable, 1st-order PDE IVP using a single-hidden-layer
# feedforward neural network.

def nnpde1ivp(
        Gf,                            # 2-variable, 1st-order PDE IVP to solve
        dG_dYf,                        # Partial of G wrt Y
        dG_ddelYf,                     # Partials of G wrt del Y
        bcf,                           # BC functions
        bcdf,                          # BC function derivatives
        x,                             # Training points as pairs
        nhid = default_nhid,           # Node count in hidden layer
        maxepochs = default_maxepochs, # Max training epochs
        eta = default_eta,             # Learning rate
        clamp = default_clamp,         # Turn on/off parameter clamping
        randomize = default_randomize, # Randomize training sample order
        debug = default_debug,
        verbose = default_verbose
):
    if debug: print('Gf =', Gf)
    if debug: print('dG_dYf =', dG_dYf)
    if debug: print('dG_ddelYf =', dG_ddelYf)
    if debug: print('bcf =', bcf)
    if debug: print('bcdf =', bcdf)
    if debug: print('x =', x)
    if debug: print('nhid =', nhid)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('eta =', eta)
    if debug: print('clamp =', clamp)
    if debug: print('randomize =', randomize)
    if debug: print('debug =', debug)
    if debug: print('verbose =', verbose)

    # Sanity-check arguments.
    assert Gf
    assert dG_dYf
    assert len(dG_ddelYf) == 2
    assert len(bcf) == 2
    assert len(bcdf) == 2
    assert len(x) > 0
    assert nhid > 0
    assert maxepochs > 0
    assert eta > 0

    #----------------------------------------------------------------------------

    # Determine the number of training points.
    n = len(x)
    if debug: print('n =', n)

    # Change notation for convenience.
    m = len(bcf)
    if debug: print('m =', m)  # Will always be 2 in this code.
    H = nhid
    if debug: print('H =', H)

    #----------------------------------------------------------------------------

    # Create the network.

    # Create an array to hold the weights connecting the input nodes to the
    # hidden nodes. The weights are initialized with a uniform random
    # distribution.
    w = np.random.uniform(w_min, w_max, (2, H))
    if debug: print('w =', w)

    # Create an array to hold the biases for the hidden nodes. The
    # biases are initialized with a uniform random distribution.
    u = np.random.uniform(u_min, u_max, H)
    if debug: print('u =', u)

    # Create an array to hold the weights connecting the hidden nodes
    # to the output node. The weights are initialized with a uniform
    # random distribution.
    v = np.random.uniform(v_min, v_max, H)
    if debug: print('v =', v)

    # Create arrays to hold parameter history.
    w_history = np.zeros((maxepochs, 2, H))
    u_history = np.zeros((maxepochs, H))
    v_history = np.zeros((maxepochs, H))

    #----------------------------------------------------------------------------

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
        z = np.zeros((n, H))
        s = np.zeros((n, H))
        s1 = np.zeros((n, H))
        s2 = np.zeros((n, H))
        for i in range(n):
            for k in range(H):
                z[i,k] = u[k]
                for j in range(m):
                    z[i,k] += w[j,k]*x[i,j]
                s[i,k] = sigma(z[i,k])
                s1[i,k] = dsigma_dz(z[i,k])
                s2[i,k] = d2sigma_dz2(z[i,k])
        if debug: print('z =', z)
        if debug: print('s =', s)
        if debug: print('s1 =', s1)
        if debug: print('s2 =', s2)

        # Compute the network output and its derivatives, for each
        # training point.
        N = np.zeros(n)
        dN_dx = np.zeros((n, m))
        dN_dv = np.zeros((n, H))
        dN_du = np.zeros((n, H))
        dN_dw = np.zeros((n, m, H))
        d2N_dvdx = np.zeros((n, m, H))
        d2N_dudx = np.zeros((n, m, H))
        d2N_dwdx = np.zeros((n, m, H))
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
                    d2N_dwdx[i,j,k] = (
                        v[k]*s1[i,k] + v[k]*s2[i,k]*w[j,k]*x[i,j]
                    )
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
        dA_dx = np.zeros((n, m))
        P = np.zeros(n)
        dP_dx = np.zeros((n, m))
        Yt = np.zeros(n)
        dYt_dx = np.zeros((n, m))
        dYt_dv = np.zeros((n, H))
        dYt_du = np.zeros((n, H))
        dYt_dw = np.zeros((n, m, H))
        d2Yt_dvdx = np.zeros((n, m, H))
        d2Yt_dudx = np.zeros((n, m, H))
        d2Yt_dwdx = np.zeros((n, m, H))
        for i in range(n):
            P[i] = Pf(x[i])
            Yt[i] = Ytf(x[i], N[i], bcf)
            for j in range(m):
                dA_dx[i,j] = delAf[j](x[i], bcf, bcdf)
                dP_dx[i,j] = delPf[j](x[i])
                dYt_dx[i,j] = (
                    dA_dx[i,j] + P[i]*dN_dx[i,j] + dP_dx[i,j]*N[i]
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
                    d2Yt_dwdx[i,j,k] = (
                        P[i]*d2N_dwdx[i,j,k] + dP_dx[i,j]*dN_dw[i,j,k]
                    )
        if debug: print('dA_dx =', dA_dx)
        if debug: print('P =', P)
        if debug: print('dP_dx =', dP_dx)
        if debug: print('Yt =', Yt)
        if debug: print('dYt_dx =', dYt_dx)
        if debug: print('dYt_dv =', dYt_dv)
        if debug: print('dYt_du =', dYt_du)
        if debug: print('dYt_dw =', dYt_dw)
        if debug: print('d2Yt_dvdx =', d2Yt_dvdx)
        if debug: print('d2Yt_dudx =', d2Yt_dudx)
        if debug: print('d2Yt_dwdx =', d2Yt_dwdx)

        # Compute the value of the original differential equation
        # for each training point, and its derivatives.
        G = np.zeros(n)
        dG_dYt = np.zeros(n)
        dG_ddelYt = np.zeros((n, m))
        dG_dv = np.zeros((n, H))
        dG_du = np.zeros((n, H))
        dG_dw = np.zeros((n, m, H))
        for i in range(n):
            G[i] = Gf(x[i], Yt[i], dYt_dx[i])
            dG_dYt[i] = dG_dYf(x[i], Yt[i], dYt_dx[i])
            for j in range(m):
                dG_ddelYt[i,j] = dG_ddelYf[j](x[i], Yt[i], dYt_dx[i])
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

        # Compute the new values of the network parameters.
        v_new = np.zeros(H)
        u_new = np.zeros(H)
        w_new = np.zeros((m, H))
        for k in range(H):
            v_new[k] = v[k] - eta*dE_dv[k]
            u_new[k] = u[k] - eta*dE_du[k]
            for j in range(m):
                w_new[j,k] = w[j,k] - eta*dE_dw[j,k]
        if debug: print('v_new =', v_new)
        if debug: print('u_new =', u_new)
        if debug: print('w_new =', w_new)

        # Clamp the values at +/-1.
        # Clamp the values at +/-1.
        if clamp:
            w_new[w_new < w_min] = w_min
            w_new[w_new > w_max] = w_max
            u_new[u_new < u_min] = u_min
            u_new[u_new > u_max] = u_max
            v_new[v_new < v_min] = v_min
            v_new[v_new > v_max] = v_max

        if verbose: print(epoch, sqrt(E/n))

        # Save the new weights and biases.
        v = v_new
        u = u_new
        w = w_new

    # Save the parameter history.
    with open('w0.dat', 'w') as f:
        for epoch in range(maxepochs):
            for k in range(H):
                f.write('%f ' % w_history[epoch][0][k])
            f.write('\n')
    with open('w1.dat', 'w') as f:
        for epoch in range(maxepochs):
            for k in range(H):
                f.write('%f ' % w_history[epoch][1][k])
            f.write('\n')
    with open('u.dat', 'w') as f:
        for epoch in range(maxepochs):
            for k in range(H):
                f.write('%f ' % u_history[epoch][k])
            f.write('\n')
    with open('v.dat', 'w') as f:
        for epoch in range(maxepochs):
            for k in range(H):
                f.write('%f ' % v_history[epoch][k])
            f.write('\n')

    # Return the final solution.
    return (Yt, dYt_dx)

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
    parser.add_argument('--clamp', '-c',
                        action = 'store_true',
                        default = default_clamp,
                        help = 'Clamp parameter values at +/- 1.')
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
    parser.add_argument('--randomize', '-r',
                        action = 'store_true',
                        default = default_randomize,
                        help = 'Randomize training sample order')
    parser.add_argument('--seed', type = int,
                        default = default_seed,
                        help = 'Random number generator seed')
    parser.add_argument('--verbose', '-v',
                        action = 'store_true',
                        default = default_verbose,
                        help = 'Produce verbose output')
    parser.add_argument('--version', action = 'version',
                        version = '%(prog)s 0.0')

    # Fetch and process the arguments from the command line.
    args = parser.parse_args()
    if args.debug: print('args =', args)

    # Extract the processed options.
    clamp = args.clamp
    debug = args.debug
    eta = args.eta
    maxepochs = args.maxepochs
    nhid = args.nhid
    ntrain = args.ntrain
    pde = args.pde
    randomize = args.randomize
    seed = args.seed
    verbose = args.verbose
    if debug: print('clamp =', clamp)
    if debug: print('debug =', debug)
    if debug: print('eta =', eta)
    if debug: print('maxepochs =', maxepochs)
    if debug: print('nhid =', nhid)
    if debug: print('ntrain =', ntrain)
    if debug: print('pde =', pde)
    if debug: print('randomize =', randomize)
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
    if verbose: print('Seeding random number generator with value %d.' % seed)
    np.random.seed(seed)

    # Import the specified PDE module.
    if verbose:
        print('Importing PDE module %s.' % pde)
    pdemod = import_module(pde)
    assert pdemod.Gf
    assert len(pdemod.bcf) > 0
    assert len(pdemod.bcdf) == len(pdemod.bcf)
    assert pdemod.dG_dYf
    assert pdemod.dG_ddelYf
    assert pdemod.Yaf
    assert len(pdemod.delYaf) == len(pdemod.bcf)

    # Create the array of evenly-spaced training points. Use the same
    # values of the training points for each dimension.
    if verbose: print('Computing training points in [[0,1],[0,1]].')
    xt = np.linspace(0, 1, ntrain)
    if debug: print('xt =', xt)
    yt = xt
    if debug: print('yt =', yt)

    # Determine the number of training points.
    nxt = len(xt)
    if debug: print('nxt =', nxt)
    nyt = len(yt)
    if debug: print('nyt =', nyt)
    ntrain = len(xt)*len(yt)
    if debug: print('ntrain =', ntrain)

    # Create the list of training points.
    # ((x0,y0),(x1,y0),(x2,y0),...
    #  (x0,y1),(x1,y1),(x2,y1),...
    x = np.zeros((ntrain, 2))
    for j in range(nyt):
        for i in range(nxt):
            k = j*nxt + i
            x[k,0] = xt[i]
            x[k,1] = yt[j]
    if debug: print('x =', x)

    #----------------------------------------------------------------------------

    # Compute the 1st-order PDE solution using the neural network.
    (Yt, delYt) = nnpde1ivp(
        pdemod.Gf,             # 2-variable, 1st-order PDE IVP to solve
        pdemod.dG_dYf,         # Partial of G wrt Y
        pdemod.dG_ddelYf,      # Partials of G wrt del Y
        pdemod.bcf,            # BC functions
        pdemod.bcdf,           # BC function derivatives
        x,                     # Training points as pairs
        nhid = nhid,           # Node count in hidden layer
        maxepochs = maxepochs, # Max training epochs
        eta = eta,             # Learning rate
        clamp = clamp,         # Turn on/off parameter clamping
        randomize = randomize, # Randomize training sample order
        debug = debug,
        verbose = verbose
    )

    #----------------------------------------------------------------------------

    # Compute the analytical solution at the training points.
    Ya = np.zeros(len(x))
    for i in range(len(x)):
        Ya[i] = pdemod.Yaf(x[i])
    if debug: print('Ya =', Ya)

    # Compute the analytical derivative at the training points.
    delYa = np.zeros((len(x), len(x[1])))
    for i in range(len(x)):
        for j in range(len(x[0])):
            delYa[i,j] = pdemod.delYaf[j](x[i])
    if debug: print('delYa =', delYa)

    # Compute the RMSE of the trial solution.
    Y_err = Yt - Ya
    if debug: print('Y_err =', Y_err)
    rmse_Y = sqrt(sum(Y_err**2) / len(x))
    if debug: print('rmse_Y =', rmse_Y)

    # Compute the MSE of the trial derivative.
    delY_err = delYt - delYa
    if debug: print('delY_err =', delY_err)
    rmse_delY = np.zeros(len(x[0]))
    e2sum = np.zeros(len(x[0]))
    for j in range(len(x[0])):
        for i in range(len(x)):
            e2sum[j] += delY_err[i,j]**2
        rmse_delY[j] = sqrt(e2sum[j] / len(x))
    if debug: print('rmse_delY =', rmse_delY)

    # Print the report.
    # print('    x        y      Ya     Yt   dYa_dx dYt_dx dYa_dy dYt_dy')
    # for i in range(len(Ya)):
    #     print('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' %
    #           (x[i,0], x[i,1],
    #            Ya[i], Yt[i],
    #            delYa[i,0], delYt[i,0],
    #            delYa[i,1], delYt[i,1]
    #           )
    #     )
    # print('RMSE              %f          %f          %f' %
    #       (rmse_Y, rmse_delY[0], rmse_delY[1]))
    print(rmse_Y)

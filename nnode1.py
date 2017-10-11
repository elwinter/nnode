#!/usr/bin/env python

# Use a neural network to solve a 1st-order ODE.

#********************************************************************************

# Import external modules, using standard shorthand.

import argparse
import importlib
from math import sqrt
import numpy as np

from sigma import sigma, dsigma_dz, d2sigma_dz2, d3sigma_dz3

#********************************************************************************

# Function to solve a 1st-order ODE using a single-hidden-layer
# feedforward neural network.

def nnode1(debug = False, eta = 0.01, maxepochs = 1000, nhid = 10, ntrain = 10,
           ode = 'ode00', seed = 0, verbose = False, xmin = 0, xmax = 1):
    # print('debug =', debug)
    # print('eta =', eta)
    # print('maxepochs =', maxepochs)
    # print('nhid =', nhid)
    # print('ode =', ode)
    # print('seed =', seed)
    # print('verbose =', verbose)
    # print('xmin =', xmin)
    # print('xmax =', xmax)

    #----------------------------------------------------------------------------

    # Import the specified ODE module, and map its functions to local
    # names for convenience.
    if verbose:
        print("Importing 1st-order ODE module '%s'." % ode)
    odemod = importlib.import_module(ode)
    b = odemod.b
    yanal = odemod.yanal
    F = odemod.F
    dF_dy = odemod.dF_dy
    d2F_dy2 = odemod.d2F_dy2
    ytrial = odemod.ytrial
    dytrial_dx = odemod.dytrial_dx

    #----------------------------------------------------------------------------

    # Create the network.

    # Initialize the random number generator to ensure repeatable results.
    if verbose: print('Seeding random number generator.')
    np.random.seed(seed)

    # Create the array of training inputs.
    dx = (xmax - xmin) / ntrain
    x = np.arange(xmin, xmax , dx) + dx
    if debug: print('x =', x)

    # Compute the analytical solution at the training points.
    ya = np.zeros(ntrain)
    for i in range(ntrain):
        ya[i] = yanal(x[i])
    if debug: print('ya =', ya)

    # Compute the analytical derivative at the training points.
    dya_dx = np.zeros(ntrain)
    for i in range(ntrain):
        dya_dx[i] = F(x[i], ya[i])
    if debug: print('dya_dx =', dya_dx)

    # Create an array to hold the weights connecting the hidden nodes to
    # the output node. The weights are initialized by using a uniform
    # random distribution between -1 and 1.
    v = np.random.uniform(-1, 1, nhid)
    if debug: print('v =', v)

    # Create an array to hold the biases for the hidden nodes. The biases
    # are initialized by using a uniform random distribution between -1
    # and 1.
    u = np.random.uniform(-1, 1, nhid)
    if debug: print('u =', u)

    # Create an array to hold the weights connecting the input node to the
    # hidden nodes. The weights are initialized by using a uniform random
    # distribution between -1 and 1.
    w = np.random.uniform(-1, 1, nhid)
    if debug: print('w =', w)

    #----------------------------------------------------------------------------

    # Run the network.
    for epoch in range(maxepochs):

        if debug: print('Starting epoch %d.' % epoch)

        # Compute the net input into each hidden node for each
        # training point.
        z = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                z[i][j] = w[j] * x[i] + u[j]
        if debug: print('z =', z)

        # Compute the output of each hidden node using the sigmoid
        # transfer function, for each training point.
        s = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                s[i][j] = sigma(z[i][j])
        if debug: print('s =', s)

        # Compute the net input to the output node, for each training
        # point (my equation (6)).
        z_out = np.zeros(ntrain)
        for i in range(ntrain):
            for j in range(nhid):
                z_out[i] += v[j] * s[i][j]
        if debug: print('z_out =', z_out)

        # Compute the output from the output node.
        N = z_out
        if debug: print('N =', N)

        #------------------------------------------------------------------------

        # Compute the trial solution.

        # Compute the value of the trial solution for each training point.
        yt = np.zeros(ntrain)
        for i in range(ntrain):
            yt[i] = ytrial(x[i], N[i])
        if debug: print('yt =', yt)

        #------------------------------------------------------------------------

        # Compute the error function.

        # Evaluate the first derivative of the sigmoid function, for each
        # training point at each hidden node.
        s1 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                s1[i][j] = dsigma_dz(z[i][j])
        if debug: print('s1 =', s1)

        # Compute the gradient of the network output with respect to the
        # training points.
        Ng = np.zeros(ntrain)
        for i in range(ntrain):
            for j in range(nhid):
                Ng[i] += v[j] * w[j] * s1[i][j]
        if debug: print('Ng =', Ng)

        # Compute the value of the derivative of the trial function
        # dyt/dx for each training point.
        dyt_dx = np.zeros(ntrain)
        for i in range(ntrain):
            dyt_dx[i] = dytrial_dx(x[i], N[i], Ng[i])
        if debug: print('dyt_dx =', dyt_dx)

        # Compute the value of the original derivative function for each
        # training point.
        f = np.zeros(ntrain)
        for i in range(ntrain):
            f[i] = F(x[i], yt[i])
        if debug: print('f =', f)

        # Compute the error function for this pass.
        E = 0
        for i in range(ntrain):
            E += (dyt_dx[i] - f[i])**2
        if debug: print('E =', E)

        #------------------------------------------------------------------------

        # Compute the derivatives needed to update the network weights
        # and biases.

        # Evaluate the second derivative of the sigmoid function, for each
        # training point at each hidden node.
        s2 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                s2[i][j] = d2sigma_dz2(z[i][j])
        if debug: print('s2 =', s2)

        # Evaluate the 3rd derivative of the sigmoid function, for each
        # training point at each hidden node.
        s3 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                s3[i][j] = d3sigma_dz3(z[i][j])
        if debug: print('s3 =', s3)

        # 1st partials of the network output wrt network parameters
        dN_dv = np.zeros((ntrain, nhid))
        dN_du = np.zeros((ntrain, nhid))
        dN_dw = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                dN_dv[i][j] = s[i][j]
                dN_du[i][j] = v[j] * s1[i][j]
                dN_dw[i][j] = x[i] * v[j] * s1[i][j]
        if debug: print('dN_dv =', dN_dv)
        if debug: print('dN_du =', dN_du)
        if debug: print('dN_dw =', dN_dw)

        # 2nd partials of the network output wrt network parameters
        d2N_dv2 = np.zeros((ntrain, nhid))
        d2N_du2 = np.zeros((ntrain, nhid))
        d2N_dw2 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                d2N_dv2[i][j] = 0
                d2N_du2[i][j] = v[j] * s2[i][j]
                d2N_dw2[i][j] = x[i]**2 * v[j] * s2[i][j]
        if debug: print('d2N_dv2 =', d2N_dv2)
        if debug: print('d2N_du2 =', d2N_du2)
        if debug: print('d2N_dw2 =', d2N_dw2)

        # 1st partials of the network gradient wrt network parameters
        dNg_dv = np.zeros((ntrain, nhid))
        dNg_du = np.zeros((ntrain, nhid))
        dNg_dw = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                dNg_dv[i][j] = w[j] * s1[i][j]
                dNg_du[i][j] = v[j] * w[j] * s2[i][j]
                dNg_dw[i][j] = x[i] * v[j] * w[j] * s2[i][j] + v[j] * s1[i][j]
        if debug: print('dNg_dv =', dNg_dv)
        if debug: print('dNg_du =', dNg_du)
        if debug: print('dNg_dw =', dNg_dw)

        # 2nd partials of the network gradient wrt network parameters
        d2Ng_dv2 = np.zeros((ntrain, nhid))
        d2Ng_du2 = np.zeros((ntrain, nhid))
        d2Ng_dw2 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                d2Ng_dv2[i][j] = 0
                d2Ng_du2[i][j] = v[j] * w[j] * s3[i][j]
                d2Ng_dw2[i][j] = (
                    x[i] * v[j] * (s2[i][j] + w[j] * x[i] * s3[i][j])
                    + x[i] * v[j] * s2[i][j]
                )
        if debug: print('d2Ng_dv2 =', d2Ng_dv2)
        if debug: print('d2Ng_du2 =', d2Ng_du2)
        if debug: print('d2Ng_dw2 =', d2Ng_dw2)

        # 1st partials of yt wrt network parameters
        dyt_dv = np.zeros((ntrain, nhid))
        dyt_du = np.zeros((ntrain, nhid))
        dyt_dw = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                dyt_dv[i][j] = x[i] * dN_dv[i][j]
                dyt_du[i][j] = x[i] * dN_du[i][j]
                dyt_dw[i][j] = x[i] * dN_dw[i][j]
        if debug: print('dyt_dv =', dyt_dv)
        if debug: print('dyt_du =', dyt_du)
        if debug: print('dyt_dw =', dyt_dw)

        # 1st y-partial of the derivative function
        df_dyt = np.zeros(ntrain)
        for i in range(ntrain):
            df_dyt[i] = dF_dy(x[i], yt[i])
        if debug: print('df_dyt =', df_dyt)

        # 2nd y-partial of the derivative function
        d2f_dyt2 = np.zeros(ntrain)
        for i in range(ntrain):
            d2f_dyt2[i] = d2F_dy2(x[i], yt[i])
        if debug: print('d2f_dyt2 =', d2f_dyt2)

        # Partials of df/dy wrt the network parameters
        d2f_dvdyt = np.zeros((ntrain, nhid))
        d2f_dudyt = np.zeros((ntrain, nhid))
        d2f_dwdyt = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                d2f_dvdyt[i][j] = d2f_dyt2[i] * dyt_dv[i][j]
                d2f_dudyt[i][j] = d2f_dyt2[i] * dyt_du[i][j]
                d2f_dwdyt[i][j] = d2f_dyt2[i] * dyt_dw[i][j]
        if debug: print('d2f_dvdyt =', d2f_dvdyt)
        if debug: print('d2f_dudyt =', d2f_dudyt)
        if debug: print('d2f_dwdyt =', d2f_dwdyt)

        #------------------------------------------------------------------------

        # 1st partials of f wrt network parameters
        df_dv = np.zeros((ntrain, nhid))
        df_du = np.zeros((ntrain, nhid))
        df_dw = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                df_dv[i][j] = (x[i] - b) * df_dyt[i] * dN_dv[i][j]
                df_du[i][j] = (x[i] - b) * df_dyt[i] * dN_du[i][j]
                df_dw[i][j] = (x[i] - b) * df_dyt[i] * dN_dw[i][j]
        if debug: print('df_dv =', df_dv)
        if debug: print('df_du =', df_du)
        if debug: print('df_dw =', df_dw)

        # 2nd partials of f wrt network parameters
        d2f_dv2 = np.zeros((ntrain, nhid))
        d2f_du2 = np.zeros((ntrain, nhid))
        d2f_dw2 = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                d2f_dv2[i][j] = (x[i] - b) * (
                    df_dyt[i] * d2N_dv2[i][j] + d2f_dvdyt[i][j] * dN_dv[i][j]
                )
                d2f_du2[i][j] = (x[i] - b) * (
                    df_dyt[i] * d2N_du2[i][j] + d2f_dudyt[i][j] * dN_du[i][j]
                )
                d2f_dw2[i][j] = (x[i] - b) * (
                    df_dyt[i] * d2N_dw2[i][j] + d2f_dwdyt[i][j] * dN_dw[i][j]
                )
        if debug: print('d2f_dv2 =', d2f_dv2)
        if debug: print('d2f_du2 =', d2f_du2)
        if debug: print('d2f_dw2 =', d2f_dw2)

        # 1st partials of dyt/dx wrt network parameters
        d2yt_dvdx = np.zeros((ntrain, nhid))
        d2yt_dudx = np.zeros((ntrain, nhid))
        d2yt_dwdx = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                d2yt_dvdx[i][j] = (x[i] - b) * dNg_dv[i][j] + dN_dv[i][j]
                d2yt_dudx[i][j] = (x[i] - b) * dNg_du[i][j] + dN_du[i][j]
                d2yt_dwdx[i][j] = (x[i] - b) * dNg_dw[i][j] + dN_dw[i][j]
        if debug: print('d2yt_dvdx =', d2yt_dvdx)
        if debug: print('d2yt_dudx =', d2yt_dudx)
        if debug: print('d2yt_dwdx =', d2yt_dwdx)

        # 2nd partials of dyt/dx wrt network parameters
        d3yt_dv2dx = np.zeros((ntrain, nhid))
        d3yt_du2dx = np.zeros((ntrain, nhid))
        d3yt_dw2dx = np.zeros((ntrain, nhid))
        for i in range(ntrain):
            for j in range(nhid):
                d3yt_dv2dx[i][j] = (x[i] - b) * d2Ng_dv2[i][j] + d2N_dv2[i][j]
                d3yt_du2dx[i][j] = (x[i] - b) * d2Ng_du2[i][j] + d2N_du2[i][j]
                d3yt_dw2dx[i][j] = (x[i] - b) * d2Ng_dw2[i][j] + d2N_dw2[i][j]
        if debug: print('d3yt_dv2dx =', d3yt_dv2dx)
        if debug: print('d3yt_du2dx =', d3yt_du2dx)
        if debug: print('d3yt_dw2dx =', d3yt_dw2dx)

        #------------------------------------------------------------------------

        # Compute the partial derivatives of the error with respect to the
        # network parameters.
        dE_dv = np.zeros(nhid)
        dE_du = np.zeros(nhid)
        dE_dw = np.zeros(nhid)
        for j in range(nhid):
            for i in range(ntrain):
                dE_dv[j] += (
                    2 * (dyt_dx[i] - f[i]) * (d2yt_dvdx[i][j] - df_dv[i][j])
                )
                dE_du[j] += (
                    2 * (dyt_dx[i] - f[i]) * (d2yt_dudx[i][j] - df_du[i][j])
                )
                dE_dw[j] += (
                    2 * (dyt_dx[i] - f[i]) * (d2yt_dwdx[i][j] - df_dw[i][j])
                )
        if debug: print('dE_dv =', dE_dv)
        if debug: print('dE_du =', dE_du)
        if debug: print('dE_dw =', dE_dw)

        # Compute the 2nd partial derivatives of the error with respect to the
        # network parameters.
        d2E_dv2 = np.zeros(nhid)
        d2E_du2 = np.zeros(nhid)
        d2E_dw2 = np.zeros(nhid)
        for j in range(nhid):
            for i in range(ntrain):
                d2E_dv2[j] += 2 * (
                    (dyt_dx[i] - f[i]) * (d3yt_dv2dx[i][j] - d2f_dv2[i][j])
                    + (d2yt_dvdx[i][j] - df_dv[i][j])**2
                )
                d2E_du2[j] += 2 * (
                    (dyt_dx[i] - f[i]) * (d3yt_du2dx[i][j] - d2f_du2[i][j])
                    + (d2yt_dudx[i][j] - df_du[i][j])**2
                )
                d2E_dw2[j] += 2 * (
                    (dyt_dx[i] - f[i]) * (d3yt_dw2dx[i][j] - d2f_dw2[i][j])
                    + (d2yt_dwdx[i][j] - df_dw[i][j])**2
                )
        if debug: print('d2E_dv2 =', d2E_dv2)
        if debug: print('d2E_du2 =', d2E_du2)
        if debug: print('d2E_dw2 =', d2E_dw2)

        #------------------------------------------------------------------------

        # Update the weights and biases.
    
        # Compute the new values of the network parameters.
        v_new = np.zeros(nhid)
        u_new = np.zeros(nhid)
        w_new = np.zeros(nhid)
        for j in range(nhid):
            v_new[j] = v[j] - eta * dE_dv[j] / d2E_dv2[j]
            u_new[j] = u[j] - eta * dE_du[j] / d2E_du2[j]
            w_new[j] = w[j] - eta * dE_dw[j] / d2E_dw2[j]
        if debug: print('v_new =', v_new)
        if debug: print('u_new =', u_new)
        if debug: print('w_new =', w_new)

        if verbose: print(epoch, E)

        # Save the new weights and biases.
        v = v_new
        u = u_new
        w = w_new

    # Compute the MSE of the trial solution.
    y_err = yt - ya
    if debug: print('y_err =', y_err)
    mse_y = sqrt(sum((yt - ya)**2) / ntrain)
    if debug: print('mse_y =', mse_y)

    # Compute the MSE of the trial derivative.
    dy_dx_err = dyt_dx - dya_dx
    if debug: print('dy_dx_err =', dy_dx_err)
    mse_dy_dx = sqrt(sum((dyt_dx - dya_dx)**2) / ntrain)
    if debug: print('mse_dy_dx =', mse_dy_dx)

    # Return the final solution.
    return (x, yt, ya, dyt_dx, dya_dx, mse_y, mse_dy_dx)

#--------------------------------------------------------------------------------

# Begin main program.

if __name__ == '__main__':

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        description = 'Solve a ODE with a neural net',
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = 'Experiment with the settings to find what works.'
    )
    # print('parser =', parser)

    # Add command-line options.
    parser.add_argument('--debug', '-d', action = 'store_true',
                        help = 'Produce debugging output')
    parser.add_argument('--eta', type = float, default = 0.01,
                        help = 'Learning rate for parameter adjustment')
    parser.add_argument('--maxepochs', type = int, default = 1000,
                        help = 'Maximum number of training epochs')
    parser.add_argument('--nhid', type = int, default = 10,
                        help = 'Number of hidden-layer nodes to use')
    parser.add_argument('--ntrain', type = int, default = 10,
                        help = 'Number of evenly-spaced training points to use')
    parser.add_argument('--ode', type = str, default = 'ode00',
                        help = 'Name of module containing ODE to solve')
    parser.add_argument('--seed', type = int, default = 0,
                        help = 'Random number generator seed')
    parser.add_argument('--verbose', '-v', action = 'store_true',
                        help = 'Produce verbose output')
    parser.add_argument('--version', action = 'version',
                        version = '%(prog)s 0.0')
    parser.add_argument('--xmin', type = float, default = 0,
                        help = 'Minimum training x-value')
    parser.add_argument('--xmax', type = float, default = 1,
                        help = 'Maximum training x-value')

    # Fetch and process the arguments from the command line.
    args = parser.parse_args()
    if args.debug:
        print('args =', args)

    # Extract the processed options.
    debug = args.debug
    eta = args.eta
    maxepochs = args.maxepochs
    nhid = args.nhid
    ntrain = args.ntrain
    ode = args.ode
    seed = args.seed
    verbose = args.verbose
    xmin = args.xmin
    xmax = args.xmax

    #----------------------------------------------------------------------------

    # Compute the 1st-order ODE solution using the neural network.
    (x, yt, ya, dyt_dx, dya_dx, mse_y, mse_dy_dx) = nnode1(
        debug = debug,
        eta = eta,
        maxepochs = maxepochs,
        nhid = nhid,
        ntrain = ntrain,
        ode = ode,
        seed = seed,
        verbose = verbose,
        xmin = xmin,
        xmax = xmax
    )
    
    #----------------------------------------------------------------------------

    # Print the report.
    print('   yt       ya      dyt_dx    dya_dx')
    for i in range(ntrain):
        print('%f %f %f %f' % (yt[i], ya[i], dyt_dx[i], dya_dx[i]))
    print('MSE      %f            %f' % (mse_y, mse_dy_dx))

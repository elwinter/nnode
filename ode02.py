#!/usr/bin/env python

# ode01.py - solve second ODE in Lagaris

import math as m
import numpy as np

#********************************************************************************

# Adjust these parameters and functions for the differential equation
# being solved.

# Number of epochs to run (no convergence check)
n_epochs = 2000

# Create the array of training inputs.
n = 20
x_min = 0
x_max = 2
dx = (x_max - x_min) / n
x = np.arange(x_min, x_max , dx) + dx
# print('x =', x)

# Number of hidden nodes
H = 20

# Learning rate
eta = 0.01

# Define the analytical solution.
def yanal(x):
    return m.exp(-x / 5) * m.sin(x)

# Define the original differential equation:
# dy/dx + x*y = x  ->  dy/dx = x*(1 - y) = F(x,y)
def F(x, y):
    return (
        m.exp(-x / 5) * m.cos(x) - y / 5
    )

# Define the y-partial derivative of the differential equation.
def dF_dy(x, y):
    return -1 / 5

# Define the 2nd y-partial derivative of the differential equation.
def d2F_dy2(x, y):
    return 0

# Define the trial solution for this differential equation.
# Ref: Lagaris eq. (12), my equation (2)
def ytrial(x, N):
    A = 0
    return A + x * N

# Define the first trial derivative.
def dytrial_dx(x, N, Ng):
    return x * Ng + N

#********************************************************************************

# Define the sigmoid transfer function and its derivatives.
def sigma(z):
    return 1 / (1 + m.exp(-z))

def dsigma_dz(z):
    return m.exp(-z) / (1 + m.exp(-z))**2

def d2sigma_dz2(z):
    return (
        2 * m.exp(-2 * z) / (1 + m.exp(-z))**3 - m.exp(-z) / (1 + m.exp(-z))**2
    )

def d3sigma_dz3(z):
    return (
        6 * m.exp(-3 * z) / (1 + m.exp(-z))**4
        - 6 * m.exp(-2 * z) / (1 + m.exp(-z))**3
        + m.exp(-z) / (1 + m.exp(-z))**2
    )

#********************************************************************************

# Create the network.

# Initialize the random number generator to ensure repeatable results.
np.random.seed(0)

# Compute the analytical solution at the training points.
ya = np.zeros(n)
for i in range(n):
    ya[i] = yanal(x[i])
# print('ya =', ya)

# Create an array to hold the weights connecting the hidden nodes to
# the output node. The weights are initialized by using a uniform
# random distribution between -1 and 1.
v = np.random.uniform(-1, 1, H)
# print('v =', v)

# Create an array to hold the biases for the hidden nodes. The biases
# are initialized by using a uniform random distribution between -1
# and 1.
u = np.random.uniform(-1, 1, H)
# print('u =', u)

# Create an array to hold the weights connecting the input node to the
# hidden nodes. The weights are initialized by using a uniform random
# distribution between -1 and 1.
w = np.random.uniform(-1, 1, H)
# print('w =', w)

#--------------------------------------------------------------------------------

# Run the network.
for epoch in range(n_epochs):

    # print('Epoch', epoch)

    #----------------------------------------------------------------------------
    
    # Make a pass forward through the network.
    
    # Compute the net input into each hidden node for each training
    # point (my equation (3)).
    z = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            z[i][j] = w[j] * x[i] + u[j]
    # print('z =', z)

    # Compute the output of each hidden node using the sigmoid
    # transfer function, for each training point (my equation (5)).
    s = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            s[i][j] = sigma(z[i][j])
    # print('s =', s)

    # Compute the net input to the output node, for each training
    # point (my equation (6)). NOTE: Summation over j.
    z_out = np.zeros(n)
    for i in range(n):
        for j in range(H):
            z_out[i] += v[j] * s[i][j]
    # print('z_out =', z_out)

    # Compute the output from the output node.
    N = z_out
    # print('N =', N)

    #----------------------------------------------------------------------------

    # Compute the trial solution.

    # Compute the value of the trial solution for each training point.
    yt = np.zeros(n)
    for i in range(n):
        yt[i] = ytrial(x[i], N[i])
    # print('yt =', yt)

    #----------------------------------------------------------------------------
    
    # Compute the error function for this pass.

    # Evaluate the first derivative of the sigmoid function, for each
    # training point at each hidden node (my equation (27)).
    s1 = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            s1[i][j] = dsigma_dz(z[i][j])
    # print('s1 =', s1)
    
    # Compute the gradient of the network output with respect to the
    # training points (my equation (9)). NOTE summation over j.
    Ng = np.zeros(n)
    for i in range(n):
        for j in range(H):
            Ng[i] += v[j] * w[j] * s1[i][j]
    # print('Ng =', Ng)

    # Compute the value of the derivative of the trial function dyt/dx
    # for each training point (my equation (8)).
    dyt_dx = np.zeros(n)
    for i in range(n):
        dyt_dx[i] = dytrial_dx(x[i], N[i], Ng[i])
    # print('dyt_dx =', dyt_dx)

    # Compute the value of the original derivative function for each
    # training point.
    f = np.zeros(n)
    for i in range(n):
        f[i] = F(x[i], yt[i])
    # print('f =', f)

    # Compute the error function for this pass (my equation (7)).
    E = 0
    for i in range(n):
        E += (dyt_dx[i] - f[i])**2
    # print('E =', E)

    #----------------------------------------------------------------------------

    # Compute the derivatives needed to update the network weights and biases.

    # Evaluate the second derivative of the sigmoid function, for each
    # training point at each hidden node (28).
    s2 = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            s2[i][j] = d2sigma_dz2(z[i][j])
    # print('s2 =', s2)

    # Evaluate the 3rd derivative of the sigmoid function, for each
    # training point at each hidden node (29).
    s3 = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            s3[i][j] = d3sigma_dz3(z[i][j])
    # print('s3 =', s3)

    # 1st partials of the network output wrt network parameters (20,21,22).
    dN_dv = np.zeros((n, H))
    dN_du = np.zeros((n, H))
    dN_dw = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            dN_dv[i][j] = s[i][j]
            dN_du[i][j] = v[j] * s1[i][j]
            dN_dw[i][j] = x[i] * v[j] * s1[i][j]
    # print('dN_dv =', dN_dv)
    # print('dN_du =', dN_du)
    # print('dN_dw =', dN_dw)

    # 2nd partials of the network output wrt network parameters (31,32,33).
    d2N_dv2 = np.zeros((n, H))
    d2N_du2 = np.zeros((n, H))
    d2N_dw2 = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            d2N_dv2[i][j] = 0
            d2N_du2[i][j] = v[j] * s2[i][j]
            d2N_dw2[i][j] = x[i]**2 * v[j] * s2[i][j]
    # print('d2N_dv2 =', d2N_dv2)
    # print('d2N_du2 =', d2N_du2)
    # print('d2N_dw2 =', d2N_dw2)

    # 1st partials of the network gradient wrt network paarameters (24,25,26).
    dNg_dv = np.zeros((n, H))
    dNg_du = np.zeros((n, H))
    dNg_dw = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            dNg_dv[i][j] = w[j] * s1[i][j]
            dNg_du[i][j] = v[j] * w[j] * s2[i][j]
            dNg_dw[i][j] = x[i] * v[j] * w[j] * s2[i][j] + v[j] * s1[i][j]
    # print('dNg_dv =', dNg_dv)
    # print('dNg_du =', dNg_du)
    # print('dNg_dw =', dNg_dw)

    # 2nd partials of the network gradient wrt network parameters (35,36,37).
    d2Ng_dv2 = np.zeros((n, H))
    d2Ng_du2 = np.zeros((n, H))
    d2Ng_dw2 = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            d2Ng_dv2[i][j] = 0
            d2Ng_du2[i][j] = v[j] * w[j] * s3[i][j]
            d2Ng_dw2[i][j] = (
                x[i] * v[j] * (s2[i][j] + w[j] * x[i] * s3[i][j])
                + x[i] * v[j] * s2[i][j]
            )
    # print('d2Ng_dv2 =', d2Ng_dv2)
    # print('d2Ng_du2 =', d2Ng_du2)
    # print('d2Ng_dw2 =', d2Ng_dw2)

    # 1st partials of yt wrt network parameters (2).
    dyt_dv = np.zeros((n, H))
    dyt_du = np.zeros((n, H))
    dyt_dw = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            dyt_dv[i][j] = x[i] * dN_dv[i][j]
            dyt_du[i][j] = x[i] * dN_du[i][j]
            dyt_dw[i][j] = x[i] * dN_dw[i][j]
    # print('dyt_dv =', dyt_dv)
    # print('dyt_du =', dyt_du)
    # print('dyt_dw =', dyt_dw)

    # 1st y-partial of the derivative function (1).
    df_dyt = np.zeros(n)
    for i in range(n):
        df_dyt[i] = dF_dy(x[i], yt[i])
    # print('df_dyt =', df_dyt)

    # 2nd y-partial of the derivative function (1).
    d2f_dyt2 = np.zeros(n)
    for i in range(n):
        d2f_dyt2[i] = d2F_dy2(x[i], yt[i])
    # print('d2f_dyt2 =', d2f_dyt2)

    # Partials of df/dy wrt the network parameters (17d).
    d2f_dvdyt = np.zeros((n, H))
    d2f_dudyt = np.zeros((n, H))
    d2f_dwdyt = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            d2f_dvdyt[i][j] = d2f_dyt2[i] * dyt_dv[i][j]
            d2f_dudyt[i][j] = d2f_dyt2[i] * dyt_du[i][j]
            d2f_dwdyt[i][j] = d2f_dyt2[i] * dyt_dw[i][j]
    # print('d2f_dvdyt =', d2f_dvdyt)
    # print('d2f_dudyt =', d2f_dudyt)
    # print('d2f_dwdyt =', d2f_dwdyt)

    #----------------------------------------------------------------------------

    # 1st partials of f wrt network parameters (14)
    df_dv = np.zeros((n, H))
    df_du = np.zeros((n, H))
    df_dw = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            df_dv[i][j] = x[i] * df_dyt[i] * dN_dv[i][j]
            df_du[i][j] = x[i] * df_dyt[i] * dN_du[i][j]
            df_dw[i][j] = x[i] * df_dyt[i] * dN_dw[i][j]
    # print('df_dv =', df_dv)
    # print('df_du =', df_du)
    # print('df_dw =', df_dw)

    # 2nd partials of f wrt network parameters (17)
    d2f_dv2 = np.zeros((n, H))
    d2f_du2 = np.zeros((n, H))
    d2f_dw2 = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            d2f_dv2[i][j] = x[i] * (
                df_dyt[i] * d2N_dv2[i][j] + d2f_dvdyt[i][j] * dN_dv[i][j]
            )
            d2f_du2[i][j] = x[i] * (
                df_dyt[i] * d2N_du2[i][j] + d2f_dudyt[i][j] * dN_du[i][j]
            )
            d2f_dw2[i][j] = x[i] * (
                df_dyt[i] * d2N_dw2[i][j] + d2f_dwdyt[i][j] * dN_dw[i][j]
            )
    # print('d2f_dv2 =', d2f_dv2)
    # print('d2f_du2 =', d2f_du2)
    # print('d2f_dw2 =', d2f_dw2)

    # 1st partials of dyt/dx wrt network parameters (13abc).
    d2yt_dvdx = np.zeros((n, H))
    d2yt_dudx = np.zeros((n, H))
    d2yt_dwdx = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            d2yt_dvdx[i][j] = x[i] * dNg_dv[i][j] + dN_dv[i][j]
            d2yt_dudx[i][j] = x[i] * dNg_du[i][j] + dN_du[i][j]
            d2yt_dwdx[i][j] = x[i] * dNg_dw[i][j] + dN_dw[i][j]
    # print('d2yt_dvdx =', d2yt_dvdx)
    # print('d2yt_dudx =', d2yt_dudx)
    # print('d2yt_dwdx =', d2yt_dwdx)

    # 2nd partials of dyt/dx wrt network parameters (16abc).
    d3yt_dv2dx = np.zeros((n, H))
    d3yt_du2dx = np.zeros((n, H))
    d3yt_dw2dx = np.zeros((n, H))
    for i in range(n):
        for j in range(H):
            d3yt_dv2dx[i][j] = x[i] * d2Ng_dv2[i][j] + d2N_dv2[i][j]
            d3yt_du2dx[i][j] = x[i] * d2Ng_du2[i][j] + d2N_du2[i][j]
            d3yt_dw2dx[i][j] = x[i] * d2Ng_dw2[i][j] + d2N_dw2[i][j]
    # print('d3yt_dv2dx =', d3yt_dv2dx)
    # print('d3yt_du2dx =', d3yt_du2dx)
    # print('d3yt_dw2dx =', d3yt_dw2dx)

    #----------------------------------------------------------------------------

    # Compute the partial derivatives of the error with respect to the
    # network parameters (12).
    dE_dv = np.zeros(H)
    dE_du = np.zeros(H)
    dE_dw = np.zeros(H)
    for j in range(H):
        for i in range(n):
            dE_dv[j] += 2 * (dyt_dx[i] - f[i]) * (d2yt_dvdx[i][j] - df_dv[i][j])
            dE_du[j] += 2 * (dyt_dx[i] - f[i]) * (d2yt_dudx[i][j] - df_du[i][j])
            dE_dw[j] += 2 * (dyt_dx[i] - f[i]) * (d2yt_dwdx[i][j] - df_dw[i][j])
    # print('dE_dv =', dE_dv)
    # print('dE_du =', dE_du)
    # print('dE_dw =', dE_dw)

    # Compute the 2nd partial derivatives of the error with respect to the
    # network parameters (18).
    d2E_dv2 = np.zeros(H)
    d2E_du2 = np.zeros(H)
    d2E_dw2 = np.zeros(H)
    for j in range(H):
        for i in range(n):
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
    # print('d2E_dv2 =', d2E_dv2)
    # print('d2E_du2 =', d2E_du2)
    # print('d2E_dw2 =', d2E_dw2)

    #----------------------------------------------------------------------------

    # Update the weights and biases.
    
    # Compute the new values of the network parameters (11).
    v_new = np.zeros(H)
    u_new = np.zeros(H)
    w_new = np.zeros(H)
    for j in range(H):
        v_new[j] = v[j] - eta * dE_dv[j] / d2E_dv2[j]
        u_new[j] = u[j] - eta * dE_du[j] / d2E_du2[j]
        w_new[j] = w[j] - eta * dE_dw[j] / d2E_dw2[j]
    # print('v_new =', v_new)
    # print('u_new =', u_new)
    # print('w_new =', w_new)

    print(epoch, E)

    # Save the new weights and biases.
    v = v_new
    u = u_new
    w = w_new

#---

# Print the final estimated and analytical values.
print('Final yt =', yt)
print('Final ya =', ya)

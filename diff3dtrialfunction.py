################################################################################
"""
Diff3DTrialFunction - Class implementing the trial function for 3-D diffusion
problems

The trial function takes the form:

Yt(x,y,z,t) = A(x,y,z,t) + P(x,y,z,t)N(x,y,z,t,p)

where:

A(x, y, z, t) = boundary condition function that reduces to BC at boundaries
P(x, y, z, t) = network coefficient function that vanishes at boundaries
N(x, y, z, t, p) = scalar output of neural network with parameter vector p

Example:
    Create a default Diff3DTrialFunction object
        Yt_obj = Diff3DTrialFunction(bcf, delbcf, del2bcf)

    Compute the value of the trial function at a given point
        Yt = Yt_obj.Ytf([x, y. z, t], N)

    Compute the value of the boundary condition function at a given point
        A = Yt_obj.Af([x, y, z, t])

Notes:
    Variables that end in 'f' are usually functions or arrays of functions.

Attributes:
    bcf - 4x2 array of BC functions at (x,y,z,t)=0|1
    delbcf - 4x2x4 array of BC gradient functions at (x,y,z,t)=0|1
    del2bcf - 4x2x4 array of BC Laplacian component functions at (x,y,z,t)=0|1

Methods:
    Af([x, y, z, t]) - Compute boundary condition function at [x, y, z, t]

    delAf([x, y, z, t]) - Compute boundary condition function gradient at
        [x, y, z, t]

    del2Af([x, y, z, t]) - Compute boundary condition function Laplacian
        components at [x, y, z, t]

    Pf([x, y, z, t]) - Compute network coefficient function at [x, y, z, t]

    delPf([x, y, z, t]) - Compute network coefficient function gradient at
        [x, y, z, t]

    del2Pf([x, y, z, t]) - Compute network coefficient function Laplacian
        components at [x, y, z, t]

    Ytf([x, y, z, t], N) - Compute trial function at [x, y, z, t] with network
        output N

    delYtf([x, y, z, t], N, delN) - Compute trial function gradient at [x, y, z, t]
        with network output N and network output gradient delN.

    del2Ytf([x, y, z, t], N, delN, del2N) - Compute trial function Laplacian
        components at [x, y, z, t] with network output N, network output gradient
        delN, and network output Laplacian components del2N

Todo:

"""


import numpy as np


class Diff3DTrialFunction():
    """Trial function for 3D diffusion problems."""


    # Public methods

    def __init__(self, bcf, delbcf, del2bcf):
        """Constructor"""
        self.bcf = bcf
        self.delbcf = delbcf
        self.del2bcf = del2bcf

    def Af(self, xyzt):
        """Boundary condition function"""
        (x, y, z, t) = xyzt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f), (Y0f, Y1f)) = self.bcf
        A = (1 - x)*f0f([0, y, z, t]) + x*f1f([1, y, z, t]) \
            + (1 - y)*(g0f([x, 0, z, t]) - ((1 - x)*g0f([0, 0, z, t])
                       + x*g0f([1, 0, z, t]))) \
            + y*(g1f([x, 1, z, t]) - ((1 - x)*g1f([0, 1, z, t])
                 + x*g1f([1, 1, z, t]))) \
            + (1 - z)*(h0f([x, y, 0, t]) - ((1 - x)*h0f([0, y, 0, t]) + x*h0f([1, y, 0, t])
                                            + y*(h0f([x, 1, 0, t]) - ((1 - x)*h0f([0, 1, 0, t]) + x*h0f([1, 1, 0, t]))))) \
            + z*(h1f([x, y, 1, t]) -
                 ((1 - x)*h1f([0, y, 1, t]) + x*h1f([1, y, 1, t])
                  + (1 - y)*(h1f([x, 0, 1, t]) - ((1 - x)*h1f([0, 0, 1, t]) + x*h1f([1, 0, 1, t])))
                  + y*(h1f([x, 1, 1, t]) - ((1 - x)*h1f([0, 1, 1, t]) + x*h1f([1, 1, 1, t]))))) \
            + (1-t)* \
               (Y0f([x, y, z, 0]) -
                ((1 - x)*Y0f([0, y, z, 0]) + x*Y0f([1, y, z, 0]) +
                 (1 - y)*(Y0f([x, 0, z, 0]) - ((1 - x)*Y0f([0, 0, z, 0]) + x*Y0f([1, 0, z, 0]))) +
                 y*(Y0f([z, 1, z, 0]) - ((1 - x)*Y0f([0, 1, z, 0]) + x*Y0f([1, 1, z, 0]))) +
                 (1 - z)*
                 (Y0f([x, y, 0, 0]) - ((1 - x)*Y0f([0, y, 0, 0]) + x*Y0f([1, y, 0, 0]) +
                                       (1 - y)*(Y0f([x, 0, 0, 0]) - ((1 - x)*Y0f([0, 0, 0, 0]) + x*Y0f([1, 0, 0, 0]))) +
                                       y*(Y0f([x, 1, 0, 0]) - ((1 - x)*Y0f([0, 1, 0, 0]) + x*Y0f([1, 1, 0, 0]))))) +
                 z*(Y0f([x, y, 1, 0]) - ((1 - x)*Y0f([0, y, 1, 0]) + x*Y0f([1, y, 1, 0]) +
                                         (1 - y)*(Y0f([x, 0, 1, 0]) - ((1 - x)*Y0f([0, 0, 1, 0]) + x*Y0f([1, 0, 1, 0]))) +
                                         y*(Y0f([x, 1, 1, 0]) - ((1 - x)*Y0f([0, 1, 1, 0]) + x*Y0f([1, 1, 1, 0])))))))
        return A

    def delAf(self, xyzt):
        """Gradient of boundary condition function"""
        (x, y, z, t) = xyzt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dzf, df0_dtf), (df1_dxf, df1_dyf, df1_dzf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf)),
         ((dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf), (dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf))
         ) = self.delbcf

        dA_dx = -f0f([0, y, z, t]) + f1f([1, y, z, t]) + \
            (1 - y)*(f0f([0, 0, z, t]) - f1f([1, 0, z, t]) + dg0_dxf([x, 0, z, t])) + \
            y*(f0f([0, 1, z, t]) - f1f([1, 1, z, t]) + dg1_dxf([x, 1, z, t])) + \
            (1 - z)*(f0f([0, y, 0, t]) - f1f([1, y, 0, t]) -
                     (1 - y)*(h0f([0, 0, 0, t]) - h0f([1, 0, 0, t]) + dg0_dxf([x, 0, 0, t])) -
                     y*(h0f([0, 1, 0, t]) - h0f([1, 1, 0, t]) + dg1_dxf([x, 1, 0, t])) + dh0_dxf([x, y, 0, t])) + \
            z*(f0f([0, y, 1, t]) - f1f([1, y, 1, t]) -
               (1 - y)*(h1f([0, 0, 1, t]) - h1f([1, 0, 1, t]) + dg0_dxf([x, 0, 1, t])) -
               y*(h1f([0, 1, 1, t]) - h1f([1, 1, 1, t]) + dg1_dxf([x, 1, 1, t])) + dh1_dxf([x, y, 1, t])) + \
            (1 - t)*(f0f([0, y, z, 0]) - f1f([1, y, z, 0]) -
                     (1 - y)*(f0f([0, 0, z, 0]) - f1f([1, 0, z, 0]) + dg0_dxf([x, 0, z, 0])) -
                     y*(f0f([0, 1, z, 0]) - f1f([1, 1, z, 0]) + dg1_dxf([x, 1, z, 0])) -
                     (1 - z)*(f0f([0, y, 0, 0]) - f1f([1, y, 0, 0]) -
                              (1 - y)*(f0f([0, 0, 0, 0]) - f1f([1, 0, 0, 0]) + dg0_dxf([x, 0, 0, 0])) -
                              + y*(f0f([0, 1, 0, 0]) - f1f([1, 1, 0, 0]) + dg1_dxf([x, 1, 0, 0])) + dh0_dxf([x, y, 0, 0])) -
                     z*(f0f([0, y, 1, 0]) - f1f([1, y, 1, 0]) -
                        (1 - y)*(f0f([0, 0, 1, 0]) - f1f([1, 0, 1, 0]) + dg0_dxf([x, 0, 1, 0])) -
                        y*(f0f([0, 1, 1, 0]) - f1f([1, 1, 1, 0]) + dg1_dxf([x, 1, 1, 0])) +
                        dh1_dxf([x, y, 1, 0])) + dY0_dxf([x, y, z, 0]))
        dA_dy = (1 - x)*f0f([0, 0, z, t]) \
                - (1 - x)*f0f([0, 1, z, t]) \
                + x*f1f([1, 0, z, t]) - x*f1f([1, 1, z, t]) - g0f([x, 0, z, t]) + g1f([x, 1, z, t]) + \
                (1 - x)*df0_dyf([0, y, z, t]) + x*df1_dyf([1, y, z, t]) + \
                (1 - z)*(g0f([x, 0, 0, t]) - g1f([x, 1, 0, t]) - (1 - x)*h0f([0, 0, 0, t]) +
                         (1 - x)*h0f([0, 1, 0, t]) - x*h0f([1, 0, 0, t]) + x*h0f([1, 1, 0, t]) -
                         (1 - x)*df0_dyf([0, y, 0, t]) - x*df1_dyf([1, y, 0, t]) + dh0_dyf([x, y, 0, t])) + \
                z*(g0f([x, 0, 1, t]) - g1f([x, 1, 1, t]) - (1 - x)*h1f([0, 0, 1, t]) +
                   (1 - x)*h1f([0, 1, 1, t]) - x*h1f([1, 0, 1, t]) + x*h1f([1, 1, 1, t]) -
                   (1 - x)*df0_dyf([0, y, 1, t]) - x*df1_dyf([1, y, 1, t]) + dh1_dyf([x, y, 1, t])) + \
                (1 - t)*(-(1 - x)*f0f([0, 0, z, 0]) + (1 - x)*f0f([0, 1, z, 0]) -
                         x*f1f([1, 0, z, 0]) + x*f1f([1, 1, z, 0]) + g0f([x, 0, z, 0]) - g1f([x, 1, z, 0]) -
                         (1 - x)*df0_dyf([0, y, z, 0]) - x*df1_dyf([1, y, z, 0]) -
                         (1 - z)*(-(1 - x)*f0f([0, 0, 0, 0]) + (1 - x)*f0f([0, 1, 0, 0]) - x*f1f([1, 0, 0, 0]) +
                                  x*f1f([1, 1, 0, 0]) + g0f([x, 0, 0, 0]) - g1f([x, 1, 0, 0]) - (1 - x)*df0_dyf([0, y, 0, 0]) -
                                  x*df1_dyf([1, y, 0, 0]) + dh0_dyf([x, y, 0, 0])) -
                         z*(-(1 - x)*f0f([0, 0, 1, 0]) + (1 - x)*f0f([0, 1, 1, 0]) - x*f1f([1, 0, 1, 0]) +
                            x*f1f([1, 1, 1, 0]) + g0f([x, 0, 1, 0]) - g1f([x, 1, 1, 0]) -
                            (1 - x)*df0_dyf([0, y, 1, 0]) - x*df1_dyf([1, y, 1, 0]) +
                            dh1_dyf([x, y, 1, 0])) + dY0_dyf([x, y, z, 0]))
        dA_dz = (1 - x)*f0f([0, y, 0, t]) - \
                (1 - x)*f0f([0, y, 1, t]) + \
                x*f1f([1, y, 0, t]) - x*f1f([1, y, 12, t]) + \
                (1 - y)*(g0f([x, 0, 0, t]) - (1 - x)*h0f([0, 0, 0, t]) - x*h0f([1, 0, 0, t])) + \
                y*(g1f([x, 1, 0, t]) - (1 - x)*h0f([0, 1, 0, t]) - x*h0f([1, 1, 0, t])) - h0f([x, y, 0, t]) - \
                (1 - y)*(g0f([x, 0, 1, t]) - (1 - x)*h1f([0, 0, 1, t]) - x*h1f([1, 0, 1, t])) - \
                y*(g1f([x, 1, 1, t]) - (1 - x)*h1f([0, 1, 1, t]) - x*h1f([1, 1, 1, t])) + \
                h1f([x, y, 1, t]) + (1 - x)*df0_dzf([0, y, z, t]) + x*df1_dzf([1, y, z, t]) + \
                (1 - y)*(-(1 - x)*df0_dzf([0, 0, z, t]) - x*df1_dzf([1, 0, z, t]) + dg0_dzf([x, 0, z, t])) + \
                y*(-(1 - x)*df0_dzf([0, 1, z, t]) - x*df1_dzf([1, 1, z, t]) + dg1_dzf([x, 1, z, t])) + \
                (1 - t)*(-(1 - x)*f0f([0, y, 0, 0]) + (1 - x)*f0f([0, y, 1, 0]) - x*f1f([1, y, 0, 0]) + x*f1f([1, y, 1, 0]) - \
                         (1 - y)*(-(1 - x)*f0f([0, 0, 0, 0]) - x*f1f([1, 0, 0, 0]) + g0f([x, 0, 0, 0])) +
                         (1 - y)*(-(1 - x)*f0f([0, 0, 1, 0]) - x*f1f([1, 0, 1, 0]) + g0f([x, 0, 1, 0])) -
                         y*(-(1 - x)*f0f([0, 1, 0, 0]) - x*f1f([1, 1, 0, 0]) + g1f([x, 1, 0, 0])) +
                         y*(-(1 - x)*f0f([0, 1, 1, 0]) - x*f1f([1, 1, 1, 0]) + g1f([x, 1, 1, 0])) +
                         h0f([x, y, 0, 0]) - h1f([x, y, 1, 0]) - (1 - x)*df0_dzf([0, y, z, 0]) - x*df1_dzf([1, y, z, 0]) -
                         (1 - y)*(-(1 - x)*df0_dzf([0, 0, z, 0]) - x*df1_dzf([1, 0, z, 0]) + dg0_dzf([x, 0, z, 0])) -
                         y*(-(1 - x)*df0_dzf([0, 1, z, 0]) - x*df1_dzf([1, 1, z, 0]) + dg1_dzf([x, 1, z, 0])) +
                         dY0_dzf([x, y, z, 0]))
        dA_dt = (1 - x)*f0f([0, y, z, 0]) + \
                x*f1f([1, y, z, 0]) + \
                (1 - y)*(-(1 - x)*f0f([0, 0, z, 0]) - x*f1f([1, 0, z, 0]) + g0f([x, 0, z, 0])) + \
                y*(-(1 - x)*f0f([0, 1, z, 0]) - x*f1f([1, 1, z, 0]) + g1f([x, 1, z, 0])) + \
                (1 - z)*(-(1 - x)*f0f([0, y, 0, 0]) - x*f1f([1, y, 0, 0]) -
                         (1 - y)*(-(1 - x)*f0f([0, 0, 0, 0]) - x*f1f([1, 0, 0, 0]) + g0f([x, 0, 0, 0])) -
                         y*(-(1 - x)*f0f([0, 1, 0, 0]) - x*f1f([1, 1, 0, 0]) + g1f([x, 1, 0, 0])) + h0f([x, y, 0, 0])) + \
                z*(-(1 - x)*f0f([0, y, 1, 0]) - x*f1f([1, y, 1, 0]) -
                   (1 - y)*(-(1 - x)*f0f([0, 0, 1, 0]) - x*f1f([1, 0, 1, 0]) + g0f([x, 0, 1, 0])) -
                   y*(-(1 - x)*f0f([0, 1, 1, 0]) - x*f1f([1, 1, 1, 0]) + g1f([x, 1, 1, 0])) +
                   h1f([x, y, 1, 0])) - \
                Y0f([x, y, z, 0]) + \
                (1 - x)*df0_dtf([0, y, z, t]) + x*df1_dtf([1, y, z, t]) + \
                (1 - y)*(-(1 - x)*df0_dtf([0, 0, z, t]) - x*df1_dtf([1, 0, z, t]) + dg0_dtf([x, 0, z, t])) + \
                y*(-(1 - x)*df0_dtf([0, 1, z, t]) - x*df1_dtf([1, 1, z, t]) + dg1_dtf([x, 1, z, t])) + \
                (1 - z)*(-(1 - x)*df0_dtf([0, y, 0, t]) - x*df1_dtf([1, y, 0, t]) -
                         (1 - y)*(dg0_dtf([x, 0, 0, t]) - (1 - x)*dh0_dtf([0, 0, 0, t]) - x*dh0_dtf([1, 0, 0, t])) -
                         y*(dg1_dtf([x, 1, 0, t]) - (1 - x)*dh0_dtf([0, 1, 0, t]) - x*dh0_dtf([1, 1, 0, t])) +
                         dh0_dtf([x, y, 0, t])) + \
                z*(-(1 - x)*df0_dtf([0, y, 1, t]) - x*df1_dtf([1, y, 1, t]) -
                   (1 - y)*(dg0_dtf([x, 0, 1, t]) - (1 - x)*dh1_dtf([0, 0, 1, t]) - x*dh1_dtf([1, 0, 1, t])) -
                   y*(dg1_dtf([x, 1, 1, t]) - (1 - x)*dh1_dtf([0, 1, 1, t]) - x*dh1_dtf([1, 1, 1, t])) + \
                   dh1_dtf([x, y, 1, t]))
        delA = [dA_dx, dA_dy, dA_dz, dA_dt]
        return delA

    def del2Af(self, xyzt):
        (x, y, z, t) = xyzt
        ((f0f, f1f), (g0f, g1f), (h0f, h1f), (Y0f, Y1f)) = self.bcf
        (((df0_dxf, df0_dyf, df0_dzf, df0_dtf), (df1_dxf, df1_dyf, df1_dzf, df1_dtf)),
         ((dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf), (dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf)),
         ((dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf), (dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf)),
         ((dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf), (dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf))
         ) = self.delbcf
        (((d2f0_dx2f, d2f0_dy2f, d2f0_dz2f, d2f0_dt2f), (d2f1_dx2f, d2f1_dy2f, d2f1_dz2f, d2f1_dt2f)),
         ((d2g0_dx2f, d2g0_dy2f, d2g0_dz2f, d2g0_dt2f), (d2g1_dx2f, d2g1_dy2f, d2g1_dz2f, d2g1_dt2f)),
         ((d2h0_dx2f, d2h0_dy2f, d2h0_dz2f, d2h0_dt2f), (d2h1_dx2f, d2h1_dy2f, d2h1_dz2f, d2h1_dt2f)),
         ((d2Y0_dx2f, d2Y0_dy2f, d2Y0_dz2f, d2Y0_dt2f), (d2Y1_dx2f, d2Y1_dy2f, d2Y1_dz2f, d2Y1_dt2f))
         ) = self.del2bcf

        d2A_dx2 = (1 - y)*d2g0_dx2f([x, 0, z, t]) + \
                  y*d2g1_dx2f([x, 1, z, t]) + \
                  (1 - z)*(-(1 - y)*d2g0_dx2f([x, 0, 0, t]) - y*d2g1_dx2f([x, 1, 0, t]) + d2h0_dx2f([x, y, 0, t])) + \
                  z*(-(1 - y)*d2g0_dx2f([x, 0, 1, t]) - y*d2g1_dx2f([x, 1, 1, t]) + d2h1_dx2f([x, y, 1, t])) + \
                  (1 - t)*(-(1 - y)*d2g0_dx2f([x, 0, z, 0]) - y*d2g1_dx2f([x, 1, z, 0]) -
                           (1 - z)*(-(1 - y)*d2g0_dx2f([x, 0, 0, 0]) - y*d2g1_dx2f([x, 1, 0, 0]) + d2h0_dx2f([x, y, 0, 0])) -
                           z*(-(1 - y)*d2g0_dx2f([x, 0, 1, 0]) - y*d2g1_dx2f([x, 1, 1, 0]) + d2h1_dx2f([x, y, 1, 0])) +
                           d2Y0_dx2f([x, y, z, 0]))
        d2A_dy2 = (1 - x)*d2f0_dy2f([0, y, z, t]) + \
                  x*d2f1_dy2f([1, y, z, t]) + \
                  (1 - z)*(-(1 - x)*d2f0_dy2f([0, y, 0, t]) - x*d2f1_dy2f([1, y, 0, t]) + d2h0_dy2f([x, y, 0, t])) + \
                  z*(-(1 - x)*d2f0_dy2f([0, y, 1, t]) - x*d2f1_dy2f([1, y, 1, t]) + d2h1_dy2f([x, y, 1, t])) + \
                  (1 - t)*(-(1 - x)*d2f0_dy2f([0, y, z, 0]) - x*d2f1_dy2f([1, y, z, 0]) -
                           (1 - z)*(-(1 - x)*d2f0_dy2f([0, y, 0, 0]) - x*d2f1_dy2f([1, y, 0, 0]) + d2h0_dy2f([x, y, 0, 0])) -
                           z*(-(1 - x)*d2f0_dy2f([0, y, 1, 0]) - x*d2f1_dy2f([1, y, 1, 0]) + d2h1_dy2f([x, y, 1, 0])) +
                           d2Y0_dy2f([x, y, z, 0]))
        d2A_dz2 = (1 - x)*d2f0_dz2f([0, y, z, t]) + \
                  x*d2f1_dz2f([1, y, z, t]) + \
                  (1 - y)*(-(1 - x)*d2f0_dz2f([0, 0, z, t]) - x*d2f1_dz2f([1, 0, z, t]) + d2g0_dz2f([x, 0, z, t])) + \
                  y*(-(1 - x)*d2f0_dz2f([0, 1, z, t]) - x*d2f1_dz2f([1, 1, z, t]) + d2g1_dz2f([x, 1, z, t])) + \
                  (1 - t)*(-(1 - x)*d2f0_dz2f([0, y, z, 0]) - x*d2f1_dz2f([1, y, z, 0]) -
                           (1 - y)*(-(1 - x)*d2f0_dz2f([0, 0, z, 0]) - x*d2f1_dz2f([1, 0, z, 0]) + d2g0_dz2f([x, 0, z, 0])) -
                           y*(-(1 - x)*d2f0_dz2f([0, 1, z, 0]) - x*d2f1_dz2f([1, 1, z, 0]) + d2g1_dz2f([x, 1, z, 0])) +
                           d2Y0_dz2f([x, y, z, 0]))
        d2A_dt2 = (1 - x)*d2f0_dt2f([0, y, z, t]) + \
                  x*d2f1_dt2f([1, y, z, t]) + \
                  (1 - y)*(-(1 - x)*d2f0_dt2f([0, 0, z, t]) - x*d2f1_dt2f([1, 0, z, t]) + d2g0_dt2f([x, 0, z, t])) + \
                  y*(-(1 - x)*d2f0_dt2f([0, 1, z, t]) - x*d2f1_dt2f([1, 1, z, t]) + d2g1_dt2f([x, 1, z, t])) + \
                  (1 - z)*(-(1 - x)*d2f0_dt2f([0, y, 0, t]) - x*d2f1_dt2f([1, y, 0, t]) -
                           (1 - y)*(d2g0_dt2f([x, 0, 0, t]) - (1 - x)*d2h0_dt2f([0, 0, 0, t]) - x*d2h0_dt2f([1, 0, 0, t])) -
                           y*(d2g1_dt2f([x, 1, 0, t]) - (1 - x)*d2h0_dt2f([0, 1, 0, t]) - x*d2h0_dt2f([1, 1, 0, t])) + d2h0_dt2f([x, y, 0, t])) + \
                  z*(-(1 - x)*d2f0_dt2f([0, y, 1, t]) - x*d2f1_dt2f([1, y, 1, t]) -
                     (1 - y)*(d2g0_dt2f([x, 0, 1, t]) - (1 - x)*d2h1_dt2f([0, 0, 1, t]) - x*d2h1_dt2f([1, 0, 1, t])) -
                     y*(d2g1_dt2f([x, 1, 1, t]) - (1 - x)*d2h1_dt2f([0, 1, 1, t]) - x*d2h1_dt2f([1, 1, 1, t])) +
                     d2h1_dt2f([x, y, 1, t]))

        del2A = [d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2]
        return del2A

    def Pf(self, xyzt):
        """Network coefficient function for 3D diffusion problems"""
        (x, y, z, t) = xyzt
        P = x*(1 - x)*y*(1 - y)*z*(1 - z)*t
        return P

    def delPf(self, xyzt):
        """Network coefficient function gradient"""
        (x, y, z, t) = xyzt
        dP_dx = (1 - 2*x)*y*(1 - y)*z*(1 - z)*t
        dP_dy = x*(1 - x)*(1 - 2*y)*z*(1 - z)*t
        dP_dz = x*(1 - x)*y*(1 - y)*(1 - 2*z)*t
        dP_dt = x*(1 - x)*y*(1 - y)*z*(1 - z)
        delP = [dP_dx, dP_dy, dP_dz, dP_dt]
        return delP

    def del2Pf(self, xyzt):
        """Network coefficient function Laplacian"""
        (x, y, z, t) = xyzt
        d2P_dx2 = -2*y*(1 - y)*z*(1 - z)*t
        d2P_dy2 = -2*x*(1 - x)*z*(1 - z)*t
        d2P_dz2 = -2*x*(1 - x)*y*(1 - y)*t
        d2P_dt2 = 0
        del2P = [d2P_dx2, d2P_dy2, d2P_dz2, d2P_dt2]
        return del2P

    def Ytf(self, xyzt, N):
        """Trial function"""
        A = self.Af(xyzt)
        P = self.Pf(xyzt)
        Yt = A + P*N
        return Yt

    def delYtf(self, xyzt, N, delN):
        """Trial function gradient"""
        (x, y, z, t) = xyzt
        (dN_dx, dN_dy, dN_dz, dN_dt) = delN
        (dA_dx, dA_dy, dA_dz, dA_dt) = self.delAf(xyzt)
        P = self.Pf(xyzt)
        (dP_dx, dP_dy, dP_dz, dP_dt) = self.delPf(xyzt)
        dYt_dx = dA_dx + P*dN_dx + dP_dx*N
        dYt_dy = dA_dy + P*dN_dy + dP_dy*N
        dYt_dz = dA_dz + P*dN_dz + dP_dz*N
        dYt_dt = dA_dt + P*dN_dt + dP_dt*N
        delYt = [dYt_dx, dYt_dy, dYt_dz, dYt_dt]
        return delYt

    def del2Ytf(self, xyzt, N, delN, del2N):
        """Trial function Laplacian"""
        (x, y, z, t) = xyzt
        (dN_dx, dN_dy, dN_dz, dN_dt) = delN
        (d2N_dx2, d2N_dy2, d2N_dz2, d2N_dt2) = del2N
        (d2A_dx2, d2A_dy2, d2A_dz2, d2A_dt2) = self.del2Af(xyzt)
        P = self.Pf(xyzt)
        (dP_dx, dP_dy, dP_dz, dP_dt) = self.delPf(xyzt)
        (d2P_dx2, d2P_dy2, d2P_dz2, d2P_dt2) = self.del2Pf(xyzt)
        d2Yt_dx2 = d2A_dx2 + P*d2N_dx2 + 2*dP_dx*dN_dx + d2P_dx2*N
        d2Yt_dy2 = d2A_dy2 + P*d2N_dy2 + 2*dP_dy*dN_dy + d2P_dy2*N
        d2Yt_dz2 = d2A_dz2 + P*d2N_dz2 + 2*dP_dz*dN_dz + d2P_dz2*N
        d2Yt_dt2 = d2A_dt2 + P*d2N_dt2 + 2*dP_dt*dN_dt + d2P_dt2*N
        del2Yt = [d2Yt_dx2, d2Yt_dy2, d2Yt_dz2, d2Yt_dt2]
        return del2Yt

#################


# Self-test code

from math import pi, sin, cos

if __name__ == '__main__':

    # Test boundary conditions
    f0f = lambda xyzt: 0
    f1f = lambda xyzt: 0
    g0f = lambda xyzt: 0
    g1f = lambda xyzt: 0
    h0f = lambda xyzt: 0
    h1f = lambda xyzt: 0
    Y0f = lambda xyzt: sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    Y1f = lambda xyzt: None
    bcf = [[f0f, f1f],
           [g0f, g1f],
           [h0f, h1f],
           [Y0f, Y1f]]

    # Test BC gradient
    df0_dxf = lambda xyzt: 0
    df0_dyf = lambda xyzt: 0
    df0_dzf = lambda xyzt: 0
    df0_dtf = lambda xyzt: 0
    df1_dxf = lambda xyzt: 0
    df1_dyf = lambda xyzt: 0
    df1_dzf = lambda xyzt: 0
    df1_dtf = lambda xyzt: 0
    dg0_dxf = lambda xyzt: 0
    dg0_dyf = lambda xyzt: 0
    dg0_dzf = lambda xyzt: 0
    dg0_dtf = lambda xyzt: 0
    dg1_dxf = lambda xyzt: 0
    dg1_dyf = lambda xyzt: 0
    dg1_dzf = lambda xyzt: 0
    dg1_dtf = lambda xyzt: 0
    dh0_dxf = lambda xyzt: 0
    dh0_dyf = lambda xyzt: 0
    dh0_dzf = lambda xyzt: 0
    dh0_dtf = lambda xyzt: 0
    dh1_dxf = lambda xyzt: 0
    dh1_dyf = lambda xyzt: 0
    dh1_dzf = lambda xyzt: 0
    dh1_dtf = lambda xyzt: 0
    dY0_dxf = lambda xyzt: pi*cos(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    dY0_dyf = lambda xyzt: pi*sin(pi*xyzt[0])*cos(pi*xyzt[1])*sin(pi*xyzt[2])/3
    dY0_dzf = lambda xyzt: pi*sin(pi*xyzt[0])*sin(pi*xyzt[1])*cos(pi*xyzt[2])/3
    dY0_dtf = lambda xyzt: 0
    dY1_dxf = lambda xyzt: None
    dY1_dyf = lambda xyzt: None
    dY1_dzf = lambda xyzt: None
    dY1_dtf = lambda xyzt: None
    delbcf = [[[df0_dxf, df0_dyf, df0_dzf, df0_dtf], [df1_dxf, df1_dyf, df1_dzf, df1_dtf]],
              [[dg0_dxf, dg0_dyf, dg0_dzf, dg0_dtf], [dg1_dxf, dg1_dyf, dg1_dzf, dg1_dtf]],
              [[dh0_dxf, dh0_dyf, dh0_dzf, dh0_dtf], [dh1_dxf, dh1_dyf, dh1_dzf, dh1_dtf]],
              [[dY0_dxf, dY0_dyf, dY0_dzf, dY0_dtf], [dY1_dxf, dY1_dyf, dY1_dzf, dY1_dtf]]]

    # Test BC Laplacian
    d2f0_dx2f = lambda xyzt: 0
    d2f0_dy2f = lambda xyzt: 0
    d2f0_dz2f = lambda xyzt: 0
    d2f0_dt2f = lambda xyzt: 0
    d2f1_dx2f = lambda xyzt: 0
    d2f1_dy2f = lambda xyzt: 0
    d2f1_dz2f = lambda xyzt: 0
    d2f1_dt2f = lambda xyzt: 0
    d2g0_dx2f = lambda xyzt: 0
    d2g0_dy2f = lambda xyzt: 0
    d2g0_dz2f = lambda xyzt: 0
    d2g0_dt2f = lambda xyzt: 0
    d2g1_dx2f = lambda xyzt: 0
    d2g1_dy2f = lambda xyzt: 0
    d2g1_dz2f = lambda xyzt: 0
    d2g1_dt2f = lambda xyzt: 0
    d2h0_dx2f = lambda xyzt: 0
    d2h0_dy2f = lambda xyzt: 0
    d2h0_dz2f = lambda xyzt: 0
    d2h0_dt2f = lambda xyzt: 0
    d2h1_dx2f = lambda xyzt: 0
    d2h1_dy2f = lambda xyzt: 0
    d2h1_dz2f = lambda xyzt: 0
    d2h1_dt2f = lambda xyzt: 0
    d2Y0_dx2f = lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    d2Y0_dy2f = lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    d2Y0_dz2f = lambda xyzt: -pi**2*sin(pi*xyzt[0])*sin(pi*xyzt[1])*sin(pi*xyzt[2])/3
    d2Y0_dt2f = lambda xyzt: 0
    d2Y1_dx2f = lambda xyzt: None
    d2Y1_dy2f = lambda xyzt: None
    d2Y1_dz2f = lambda xyzt: None
    d2Y1_dt2f = lambda xyzt: None
    del2bcf = [[[d2f0_dx2f, d2f0_dy2f, d2f0_dz2f, d2f0_dt2f], [d2f1_dx2f, d2f1_dy2f, d2f1_dz2f, d2f1_dt2f]],
               [[d2g0_dx2f, d2g0_dy2f, d2g0_dz2f, d2g0_dt2f], [d2g1_dx2f, d2g1_dy2f, d2g1_dz2f, d2g1_dt2f]],
               [[d2h0_dx2f, d2h0_dy2f, d2h0_dz2f, d2h0_dt2f], [d2h1_dx2f, d2h1_dy2f, d2h1_dz2f, d2h1_dt2f]],
               [[d2Y0_dx2f, d2Y0_dy2f, d2Y0_dz2f, d2Y0_dt2f], [d2Y1_dx2f, d2Y1_dy2f, d2Y1_dz2f, d2Y1_dt2f]]]

    # Reference values for tests.
    bc_ref = [[0, 0],
              [0, 0],
              [0, 0],
              [0.294867, None]]
    delbc_ref = [[[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0, 0, 0, 0], [0, 0, 0, 0]],
                 [[0.30099, 0.26913, 0.237847, 0], [None, None, None, None]]]
    del2bc_ref = [[[ 0, 0, 0, 0], [0, 0, 0, 0]],
                  [[ 0, 0, 0, 0], [0, 0, 0, 0]],
                  [[ 0, 0, 0, 0], [0, 0, 0, 0]],
                  [[-2.91022, -2.91022, -2.91022, 0], [None, None, None, None]]]
    A_ref = 0.168074
    delA_ref = [0.171564, 0.153404, 0.135573, -0.294867]
    del2A_ref = [-1.65883, -1.65883, -1.65883, 0]
    P_ref = 0.00608125
    delP_ref = [0.00506771, 0.00452511, 0.00399425, 0.0141424]
    del2P_ref = [-0.0506771, -0.050279, -0.0499282, 0]
    Yt_ref = 0.171115
    delYt_ref = [0.177808, 0.159437, 0.141401, -0.283904]
    del2Yt_ref = [-1.67366, -1.67398, -1.67432, 0.0226025]

    # Additional test variables.
    N_test = 0.5
    delN_test = [0.61, 0.62, 0.63, 0.64]
    del2N_test = [0.71, 0.72, 0.73, 0.74]

    # Test all functions near the center of the domain.
    xyzt_test = [0.4, 0.41, 0.42, 0.43]

    # Create a new trial function object.
    tf = Diff3DTrialFunction(bcf, delbcf, del2bcf)

    print("Testing boundary conditions.")
    for i in range(len(tf.bcf)):
        for (j, f) in enumerate(tf.bcf[i]):
            bc_test = f(xyzt_test)
            if ((bc_ref[i][j] is not None and
                 not np.isclose(bc_test, bc_ref[i][j]))
                or (bc_ref[i][j] is None and bc_test is not None)):
                print("ERROR: bc[%d][%d] = %s, vs ref %s" %
                      (i, j, bc_test, bc_ref[i][j]))

    print("Testing boundary condition gradients.")
    for i in range(len(tf.delbcf)):
        for j in range(len(tf.delbcf[i])):
            for (k, f) in enumerate(tf.delbcf[i][j]):
                delbc_test = f(xyzt_test)
                if ((delbc_ref[i][j][k] is not None and
                     not np.isclose(delbc_test, delbc_ref[i][j][k]))
                    or (delbc_ref[i][j][k] is None and
                        delbc_test is not None)):
                    print("ERROR: delbc[%d][%d][%d] = %s, vs ref %s" %
                          (i, j, k, delbc_test, delbc_ref[i][j][k]))

    print("Testing boundary condition Laplacians.")
    for i in range(len(tf.del2bcf)):
        for j in range(len(tf.del2bcf[i])):
            for (k, f) in enumerate(tf.del2bcf[i][j]):
                del2bc_test = f(xyzt_test)
                if ((del2bc_ref[i][j][k] is not None and
                     not np.isclose(del2bc_test, del2bc_ref[i][j][k]))
                    or (del2bc_ref[i][j][k] is None and
                        del2bc_test is not None)):
                    print("ERROR: del2bc[%d][%d][%d] = %s, vs ref %s" %
                          (i, j, k, del2bc_test, del2bc_ref[i][j][k]))

    print("Testing boundary condition function.")
    A_test = tf.Af(xyzt_test)
    if not np.isclose(A_test, A_ref):
        print("ERROR: A = %s, vs ref %s" % (A_test, A_ref))

    print("Testing boundary condition function gradient.")
    delA_test = tf.delAf(xyzt_test)
    for (i, delA_t) in enumerate(delA_test):
        if not np.isclose(delA_t, delA_ref[i]):
            print("ERROR: delA[%d] = %s, vs ref %s" % (i, delA_t, delA_ref[i]))

    print("Testing boundary condition function Laplacian.")
    del2A_test = tf.del2Af(xyzt_test)
    for (i, del2A_t) in enumerate(del2A_test):
        if not np.isclose(del2A_t, del2A_ref[i]):
            print("ERROR: del2A[%d] = %s, vs ref %s" % (i, del2A_t, del2A_ref[i]))

    print("Testing network coefficient function.")
    P_test = tf.Pf(xyzt_test)
    if not np.isclose(P_test, P_ref):
        print("ERROR: P = %s, vs ref %s" % (P_test, P_ref))

    print("Testing network coefficient function gradient.")
    delP_test = tf.delPf(xyzt_test)
    for (i, delP_t) in enumerate(delP_test):
        if not np.isclose(delP_t, delP_ref[i]):
            print("ERROR: delP[%d] = %s, vs ref %s" % (delP_t, delP_ref[i]))

    print("Testing network coefficient function Laplacian.")
    del2P_test = tf.del2Pf(xyzt_test)
    for (i, del2P_t) in enumerate(del2P_test):
        if not np.isclose(del2P_t, del2P_ref[i]):
            print("ERROR: del2P[%d] = %s, vs ref %s" % (del2P_t, del2P_ref[i]))

    print("Testing trial function.")
    Yt_test = tf.Ytf(xyzt_test, N_test)
    if not np.isclose(Yt_test, Yt_ref):
        print("ERROR: Yt = %s, vs ref %s" % (Yt_test, Yt_ref))

    print("Testing trial function gradient.")
    delYt_test = tf.delYtf(xyzt_test, N_test, delN_test)
    for (i, delYt_t) in enumerate(delYt_test):
        if not np.isclose(delYt_t, delYt_ref[i]):
            print("ERROR: delYt[%d] = %s, vs ref %s" % (i, delYt_t, delYt_ref[i]))

    print("Testing trial function Laplacian.")
    del2Yt_test = tf.del2Ytf(xyzt_test, N_test, delN_test, del2N_test)
    for (i, del2Yt_t) in enumerate(del2Yt_test):
        if not np.isclose(del2Yt_t, del2Yt_ref[i]):
            print("ERROR: del2Yt[%d] = %s, vs ref %s" % (i, del2Yt_t, del2Yt_ref[i]))

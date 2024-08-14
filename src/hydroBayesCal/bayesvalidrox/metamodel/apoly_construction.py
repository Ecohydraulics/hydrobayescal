#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def apoly_construction(Data, degree):
    """
    Construction of Data-driven Orthonormal Polynomial Basis
    Author: Dr.-Ing. habil. Sergey Oladyshkin
    Department of Stochastic Simulation and Safety Research for Hydrosystems
    Institute for Modelling Hydraulic and Environmental Systems
    Universitaet Stuttgart, Pfaffenwaldring 5a, 70569 Stuttgart
    E-mail: Sergey.Oladyshkin@iws.uni-stuttgart.de
    http://www.iws-ls3.uni-stuttgart.de
    The current script is based on definition of arbitrary polynomial chaos
    expansion (aPC), which is presented in the following manuscript:
    Oladyshkin, S. and W. Nowak. Data-driven uncertainty quantification using
    the arbitrary polynomial chaos expansion. Reliability Engineering & System
    Safety, Elsevier, V. 106, P.  179-190, 2012.
    DOI: 10.1016/j.ress.2012.05.002.

    Parameters
    ----------
    Data : array
        Raw data.
    degree : int
        Maximum polynomial degree.

    Returns
    -------
    Polynomial : array
        The coefficients of the univariate orthonormal polynomials.

    """
    if Data.ndim !=1:
        raise AttributeError('Data should be a 1D array')

    # Initialization
    dd = degree + 1
    nsamples = len(Data)

    # Forward linear transformation (Avoiding numerical issues)
    MeanOfData = np.mean(Data)
    Data = Data/MeanOfData

    # Compute raw moments of input data
    raw_moments = [np.sum(np.power(Data, p))/nsamples for p in range(2*dd+2)]

    # Main Loop for Polynomial with degree up to dd
    PolyCoeff_NonNorm = np.empty((0, 1))
    Polynomial = np.zeros((dd+1, dd+1))

    for degree in range(dd+1):
        Mm = np.zeros((degree+1, degree+1))
        Vc = np.zeros((degree+1))

        # Define Moments Matrix Mm
        for i in range(degree+1):
            for j in range(degree+1):
                if (i < degree):
                    Mm[i, j] = raw_moments[i+j]

                elif (i == degree) and (j == degree):
                    Mm[i, j] = 1

            # Numerical Optimization for Matrix Solver
            Mm[i] = Mm[i] / max(abs(Mm[i]))

        # Defenition of Right Hand side ortogonality conditions: Vc
        for i in range(degree+1):
            Vc[i] = 1 if i == degree else 0

        # Solution: Coefficients of Non-Normal Orthogonal Polynomial: Vp Eq.(4)
        try:
            Vp = np.linalg.solve(Mm, Vc)
        except:
            inv_Mm = np.linalg.pinv(Mm)
            Vp = np.dot(inv_Mm, Vc.T)

        if degree == 0:
            PolyCoeff_NonNorm = np.append(PolyCoeff_NonNorm, Vp)

        if degree != 0:
            if degree == 1:
                zero = [0]
            else:
                zero = np.zeros((degree, 1))
            PolyCoeff_NonNorm = np.hstack((PolyCoeff_NonNorm, zero))

            PolyCoeff_NonNorm = np.vstack((PolyCoeff_NonNorm, Vp))

        if 100*abs(sum(abs(np.dot(Mm, Vp)) - abs(Vc))) > 0.5:
            print('\n---> Attention: Computational Error too high !')
            print('\n---> Problem: Convergence of Linear Solver')

        # Original Numerical Normalization of Coefficients with Norm and
        # orthonormal Basis computation Matrix Storrage
        # Note: Polynomial(i,j) correspont to coefficient number "j-1"
        # of polynomial degree "i-1"
        P_norm = 0
        for i in range(nsamples):
            Poly = 0
            for k in range(degree+1):
                if degree == 0:
                    Poly += PolyCoeff_NonNorm[k] * (Data[i]**k)
                else:
                    Poly += PolyCoeff_NonNorm[degree, k] * (Data[i]**k)

            P_norm += Poly**2 / nsamples

        P_norm = np.sqrt(P_norm)

        for k in range(degree+1):
            if degree == 0:
                Polynomial[degree, k] = PolyCoeff_NonNorm[k]/P_norm
            else:
                Polynomial[degree, k] = PolyCoeff_NonNorm[degree, k]/P_norm

    # Backward linear transformation to the real data space
    Data *= MeanOfData
    for k in range(len(Polynomial)):
        Polynomial[:, k] = Polynomial[:, k] / (MeanOfData**(k))

    return Polynomial

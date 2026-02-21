from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import l0l2
import sys
import time
from simulated import Dataset

from sklearn.datasets import load_diabetes
from sklearn.linear_model import enet_path, lars_path, lasso_path
from sklearn.linear_model import ElasticNet

#X, y = load_diabetes(return_X_y=True)
#X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)

# Compute paths

eps = 5e-3  # the smaller it is the longer is the path
#np.savetxt("X.csv", X, delimiter=';')
#np.savetxt("y.csv", y, delimiter=';')

X = np.asfortranarray(np.loadtxt(f'refMat.csv', delimiter=';'))
y = np.asfortranarray(np.loadtxt(f'refVect.csv', delimiter=';'))

#X = np.asfortranarray(np.loadtxt(f'XSim.csv', delimiter=';'))
#y = np.asfortranarray(np.loadtxt(f'ySim.csv', delimiter=';'))

[m, n] = X.shape

delta = 2.5#0.5 #-1.0 #2.5#
beta = 1e-1
matrixIsCovariance = False

fullpath = delta < 0.0;

if fullpath:
    tic_ns = time.perf_counter_ns()

    results = l0l2.FullPathSolverf.fitAll(matData = X.ravel(order='F'),
                                    rows = m,
                                    cols = n,
                                    vectData = y.ravel(order='F'),
                                    matrixIsCovariance = matrixIsCovariance,
                                    beta = beta,
                                    doParallel = False,
                                    fromZero = False)
    
    toc_ns = time.perf_counter_ns()

    elapsed_nanoseconds = toc_ns - tic_ns

    elapsed_microseconds = elapsed_nanoseconds / 1000

    #print(f"L0L2 Elapsed time: {elapsed_microseconds:.2f} microseconds")

    for result in results:
        print(result, end="")

        status = result.isValidFor(beta)

        match status:
            case l0l2.LPSolutionStatus.Valid:
                print(" OK\n")

            case l0l2.LPSolutionStatus.RidgeConstraintFailed:
                print(" NOK! A Ridge constraint failed.\n")

            case l0l2.LPSolutionStatus.LassoConstraintFailed:
                print(" NOK! A Lasso constraint failed.\n")

            case l0l2.LPSolutionStatus.ZeroConstraintFailed:
                print(" NOK! A Zero constraint failed.\n")

            case _:
                print(" NOK! Waaw.\n")


    print(f"\nL0L2 Elapsed time: {elapsed_microseconds:.2f} microseconds")
else:
    tic_ns = time.perf_counter_ns()

    regressor = l0l2.L0L2Regressorf(delta = delta, beta = beta)#l0l2.FullPathSolverf(delta = delta, beta = beta)

    solution = regressor.fit(matData = X.ravel(order='F'),
                                    rows = m,
                                    cols = n,
                                    vectData = y.ravel(order='F'),
                                    matrixIsCovariance = matrixIsCovariance,
                                    maximumNumberOfIterations = 1000000,
                                    doParallel = False,
                                    fromZero = True)

    toc_ns = time.perf_counter_ns()

    elapsed_nanoseconds = toc_ns - tic_ns

    elapsed_microseconds = elapsed_nanoseconds / 1000

    print(solution)

    print(f"\nL0L2 Elapsed time: {elapsed_microseconds:.2f} microseconds")

    l1_ratio = delta / (1 + delta)
    alpha = beta * (delta + 1) / m

    enet_tic_ns = time.perf_counter_ns()

    enetregressor = ElasticNet(alpha=alpha, l1_ratio= l1_ratio, fit_intercept=False, max_iter=1000, tol = 0.0001)
    enetregressor.fit(X, y)

    enet_toc_ns = time.perf_counter_ns()

    enet_elapsed_nanoseconds = enet_toc_ns - enet_tic_ns

    enet_elapsed_microseconds = enet_elapsed_nanoseconds / 1000

    print(enetregressor.coef_)

    print(f"\nElasticNet Elapsed time: {enet_elapsed_microseconds:.2f} microseconds")
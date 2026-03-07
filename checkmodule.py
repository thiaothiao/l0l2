import numpy as np
import l0l2
import sys
import time
from sklearn.linear_model import ElasticNet

eps = 5e-3

X = np.asfortranarray(np.loadtxt(f'refMat.csv', delimiter=';'))
y = np.asfortranarray(np.loadtxt(f'refVect.csv', delimiter=';'))

[m, n] = X.shape

delta = 2.5#0.5 #-1.0 #2.5#
beta = 1e-1
matrixIsCovariance = False
withIntercept = False
tolerance = 1e-4
maximumNumberOfIterations = 10000
innerEpsilon = 1e-6
innerMaximumNumberOfIterations = 100000

if True:
    tic_ns = time.perf_counter_ns()

    strategy = l0l2.linearmodel.Strategy.FromZeroSolution
    paramf = l0l2.linearmodel.L0L2RegressorParamf(
            delta=delta, beta=beta,
            strategy = strategy,
            tolerance = tolerance,
            maximumNumberOfIterations = maximumNumberOfIterations,
            innerEpsilon = innerEpsilon,
            innerMaximumNumberOfIterations = innerMaximumNumberOfIterations)

    regressor = l0l2.linearmodel.L0L2Regressorf(paramf, withIntercept)

    solution = regressor.fit(matData=X, vectData=y, matrixIsCovariance=matrixIsCovariance)

    toc_ns = time.perf_counter_ns()

    elapsed_nanoseconds = toc_ns - tic_ns

    elapsed_microseconds = elapsed_nanoseconds / 1000

    print(solution)

    print(f"\nFrom zero L0L2 Elapsed time: {elapsed_microseconds:.2f} microseconds")

if True:
    tic_ns = time.perf_counter_ns()

    strategy = l0l2.linearmodel.Strategy.FromL2Solution
    paramf = l0l2.linearmodel.L0L2RegressorParamf(
            delta=delta, beta=beta,
            strategy = strategy,
            tolerance = tolerance,
            maximumNumberOfIterations = maximumNumberOfIterations,
            innerEpsilon = innerEpsilon,
            innerMaximumNumberOfIterations = innerMaximumNumberOfIterations)

    regressor = l0l2.linearmodel.L0L2Regressorf(paramf, withIntercept)

    solution = regressor.fit(matData=X, vectData=y, matrixIsCovariance=matrixIsCovariance)

    toc_ns = time.perf_counter_ns()

    elapsed_nanoseconds = toc_ns - tic_ns

    elapsed_microseconds = elapsed_nanoseconds / 1000

    print(solution)

    print(f"\nFrom L2 L0L2 Elapsed time: {elapsed_microseconds:.2f} microseconds")

if True:
    tic_ns = time.perf_counter_ns()

    strategy = l0l2.linearmodel.Strategy.FromBothSolutions
    paramf = l0l2.linearmodel.L0L2RegressorParamf(
            delta=delta, beta=beta,
            strategy = strategy,
            tolerance = tolerance,
            maximumNumberOfIterations = maximumNumberOfIterations,
            innerEpsilon = innerEpsilon,
            innerMaximumNumberOfIterations = innerMaximumNumberOfIterations)

    regressor = l0l2.linearmodel.L0L2Regressorf(paramf, withIntercept)

    solution = regressor.fit(matData=X, vectData=y, matrixIsCovariance=matrixIsCovariance)

    toc_ns = time.perf_counter_ns()

    elapsed_nanoseconds = toc_ns - tic_ns

    elapsed_microseconds = elapsed_nanoseconds / 1000

    print(solution)

    print(f"\nFrom both L0L2 Elapsed time: {elapsed_microseconds:.2f} microseconds")

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
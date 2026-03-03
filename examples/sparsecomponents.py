import numpy as np
import l0l2

Q = np.loadtxt(f'pitprops.csv', delimiter=';')

nbComponent = 3
beta = 0.01
delta = 15

param = l0l2.linearmodel.L0L2SPCAParamd(
    regressorDelta = delta, 
    regressorBeta = beta, 
    nbComponents = nbComponent)

components = l0l2.linearmodel.L0L2SPCAd(param=param).run(matData = Q, 
                                                         matrixIsCovariance = True)

print(components)

import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
import l0l2

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
X /= X.std(axis=0)

n = 2
m = 2
betas = np.logspace(1, m*n, num=m*n)
betas = betas.reshape((m, n))

fig, axs = plt.subplots(nrows=m, ncols=n, constrained_layout=True)
legend = None
for i in np.arange(m):
    for j in np.arange(n):

        beta = betas[i,j]
        
        results = l0l2.linearmodel.FullPathSolverd.fitAll(
            matData=X, vectData=y, matrixIsCovariance=False, beta=beta)

        l0l2Path = []
        deltasPath = []
        l0l2Line = None

        for result in results:
            l0l2Path.append(result.x)
            deltasPath.append(result.delta)

        l0l2Path.pop(0)
        deltasPath.pop(0)

        coefs = np.array(l0l2Path).T
        deltas = np.array(deltasPath).T

        colors = cycle(["b", "g", "c", "m", "y", "k", "r"])
        for coef, c in zip(coefs, colors):
            l0l2Line = axs[i, j].semilogx(deltas, coef, linestyle="-.", c=c)

        deltasLine = axs[i, j].semilogx(deltas, deltas, linestyle=":", c="r")
        minusdeltasLine = axs[i, j].semilogx(deltas, -deltas, linestyle=":", c="r")

        axs[i,j].set_xlabel(r'$\delta$')
        axs[i,j].set_title(f'$\\beta$ = {beta:.0e}', fontsize=9) 

        if legend is None:
            legend = plt.legend((l0l2Line[-1], deltasLine[-1]), ('l0l2',r'$\pm\delta$'))

fig.add_artist(legend)
plt.show()
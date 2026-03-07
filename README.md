# l0l2
**Biconjugate Convex Relaxation Solver for Sparse Modeling**\
The solver addresses solutions for sparse linear least squares problems via $l_0$, $l_2$  combined regularizations.
```math
\min \left\|Ax-b\right\|^2_2 + \rho\left\|x\right\|_0 + \beta\left\|x\right\|^2_2,\;x\in\mathbb{R}^n.
```
$\rho \geq 0$ is a parameter that controls the sparsity of the solutions. With $\rho = \beta\delta^2$, we simply consider
```math
(\mathbb{P}^{\beta, \delta}) \min \left\|Ax-b\right\|^2_2 + \beta\delta^2\left\|x\right\|_0 + \beta\left\|x\right\|^2_2,\;x\in\mathbb{R}^n.
```
The considered Biconjugate Convex Relaxation is given by
```math
(\mathbb{Q}^{\beta, \delta}) \min \left\|Ax-b\right\|^2_2 + \beta\delta^2(\left\|.\right\|_0 + \frac{1}{\delta^2}\left\|.\right\|_2^2)^{**}(x),\;x\in\mathbb{R}^n.
```
Implemented solutions:
- Full path solutions using a Gauss-Jordan elimination and the piece linearity of the solutions with respect to $\delta$.
- Cyclical Coordinante Descent.
- Interior Point Method. WIP

![description](https://github.com/thiaothiao/l0l2/blob/develop/examples/python/diabetes_path_0.png)

Interfaces for Sparse PCA cases are also availlable.

For the technical details see the technical report and references therein.

We use c++ 23 and later.

Dependencies:
- Eigen c++ library, https://libeigen.gitlab.io/
- thread-pool c++ libary, https://github.com/ptsouchlos/thread-pool
- pybind11 used for python interfaces, https://github.com/pybind/pybind11
- openMP

Builds: WIP
- windows with visual studio
- ubuntu

Benchmarks details: WIP

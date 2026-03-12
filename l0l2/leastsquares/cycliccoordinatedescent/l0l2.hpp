#ifndef L0L2_COORDINATE_DESCENT_SOLVER_HPP
#define L0L2_COORDINATE_DESCENT_SOLVER_HPP

#include <future>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <utility>

#include "l0l2/leastsquares/utils.hpp"
#include "l0l2/leastsquares/cycliccoordinatedescent/generic.hpp"

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            // L0L2 coordinate descent stepJ implementation
            template<std::floating_point ScalarType>
            class L0L2ModelImplementation final : public ModelImplementationBase<ScalarType>
            {
            public:
                using Base = ModelImplementationBase<ScalarType>;
                using typename Base::Scalar;

                struct Param final
                {
                    /*! \brief l0l2 model parameter object constructor.
                      \param deltaInput sparsity regularization parameter.
                      \param betaInput l2 regularization parameter.
                      \param strategyInput enum indicating a strategy: from zero, or l2 or both solutions.
                      \param toleranceInput covergence tolerance on the coordinates changes.
                      \param maximumNumberOfIterationsInput maximum number of iterations allowed.
                    */
                    Param(
                        Scalar deltaInput = static_cast<Scalar>(0),
                        Scalar betaInput = static_cast<Scalar>(1),
                        bool hasInterceptInput = false,
                        Strategy strategyInput = Strategy::FromZeroSolution,
                        Scalar toleranceInput = static_cast<Scalar>(1e-4),
                        unsigned int maximumNumberOfIterationsInput = 10000U,
                        Scalar innerEpsilonInput = static_cast<Scalar>(1e-6),
                        unsigned int innerMaximumNumberOfIterationsInput = 100000U)
                        :delta{ deltaInput },
                        beta{ betaInput },
                        hasIntercept{ hasInterceptInput },
                        strategy{ strategyInput },
                        tolerance{ toleranceInput },
                        maximumNumberOfIterations{ maximumNumberOfIterationsInput },
                        innerEpsilon{ innerEpsilonInput },
                        innerMaximumNumberOfIterations{ innerMaximumNumberOfIterationsInput },
                        deltaBeta{ deltaInput * betaInput }
                    {
                    }

                    Param(const Param&) = default;
                    Param& operator=(const Param&) = default;

                    Param(Param&&) = default;
                    Param& operator=(Param&&) = default;

                    const Scalar delta;
                    const Scalar beta;
                    const bool hasIntercept;
                    const Strategy strategy;
                    const Scalar tolerance;
                    const unsigned int maximumNumberOfIterations;
                    const Scalar innerEpsilon;
                    const unsigned int innerMaximumNumberOfIterations;
                    const Scalar deltaBeta;
                };

                L0L2ModelImplementation(const Param& paramInput = {})
                    :param{ paramInput }
                {
                }

                using Base::stepIntercept;

                Scalar stepJ(Scalar zJ, Scalar uJ) const;

                Scalar otherJ(Scalar zJ, Scalar uJ) const;

                Scalar computeDualityGap(const Matrix<Scalar>& matData, const Vector<Scalar>& vectData,
                    bool matrixIsCovariance, const CoordinateStates& coordinateStates, 
                    const Solution<Scalar>& solution) const;

                Param param;
            };

            template<std::floating_point ScalarType>
            inline L0L2ModelImplementation<ScalarType>::Scalar
                L0L2ModelImplementation<ScalarType>::stepJ(Scalar zJ, Scalar uJ) const
            {
                return (std::abs(uJ) >= param.deltaBeta + zJ * param.delta) ? (uJ / (param.beta + zJ))
                    : ((std::abs(uJ) > param.deltaBeta) ? ((uJ - param.deltaBeta * Utils<Scalar>::sign(uJ)) / zJ)
                        : static_cast<Scalar>(0));
            }

            template<std::floating_point ScalarType>
            inline L0L2ModelImplementation<ScalarType>::Scalar
                L0L2ModelImplementation<ScalarType>::otherJ(Scalar zJ, Scalar uJ) const
            {
                return uJ / (zJ + param.beta);
            }

            template<std::floating_point ScalarType>
            L0L2ModelImplementation<ScalarType>::Scalar
                L0L2ModelImplementation<ScalarType>::computeDualityGap(const Matrix<Scalar>& matData, 
                    const Vector<Scalar>& vectData, bool matrixIsCovariance, 
                    const CoordinateStates& coordinateStates, const Solution<Scalar>& solution) const
            {
                using Vector = Vector<Scalar>;
                using Utils = Utils<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto subIndicesCase = static_cast<Index>(coordinateStates.size()) == n;

                const auto& x = solution.x;
                const auto intercept = solution.intercept;

                const auto hasIntercept = !matrixIsCovariance && param.hasIntercept;

                Vector nu0;
                if (matrixIsCovariance)
                {
                    nu0 = matData * (x - vectData);
                }
                else
                {
                    nu0 = matData * x - vectData;
                }

                if (hasIntercept)
                {
                    nu0.array() += intercept;
                }

                const auto betaDeltaSquared = param.beta * param.delta * param.delta;
                const auto twoDeltaBeta = static_cast<Scalar>(2) * param.deltaBeta;
                const auto leastSquaresPartValue = matrixIsCovariance ? nu0.dot(x - vectData) : nu0.squaredNorm();
                auto primal = leastSquaresPartValue;
                for (auto xi : x)
                {
                    const auto absxi = std::abs(xi);
                    primal += (absxi > param.delta ? (param.beta * absxi * absxi + betaDeltaSquared)
                        : twoDeltaBeta * absxi);
                }

                if (hasIntercept)
                {
                    nu0.array() -= nu0.mean();
                }

                Vector theta;
                if (matrixIsCovariance)
                {
                    theta = nu0.cwiseAbs() / twoDeltaBeta;
                }
                else
                {
                    theta = (matData.transpose() * nu0).cwiseAbs() / twoDeltaBeta;
                }

                auto aCoeff = - static_cast<Scalar>(0.25) * leastSquaresPartValue;
                if (subIndicesCase)
                {// TODO try eigenlib select() method
                    auto tmp = static_cast<Scalar>(0);
                    for (Index j = 0; j < n; ++j)
                    {
                        if (coordinateStates[j] == CoordinateState::ZERO)
                        {
                            theta[j] = static_cast<Scalar>(0);
                        }
                        else if(coordinateStates[j] == CoordinateState::FREE)
                        {
                            tmp += theta[j] * theta[j];
                            theta[j] = static_cast<Scalar>(0);
                        }
                    }

                    aCoeff += betaDeltaSquared * (coordinateStates == CoordinateState::FREE).count() 
                        -static_cast<Scalar>(0.25) * tmp * betaDeltaSquared * betaDeltaSquared / param.beta;
                }

                std::sort(theta.begin(), theta.end(), [](Scalar u, Scalar v) {return u > v; });

                const auto bTnu0 = vectData.dot(nu0);
                const auto bCoeff = std::abs(bTnu0);

                auto cCoeff = static_cast<Scalar>(0);

                // solve on [s0,s1]
                auto dual = Utils::solveMaxConcaveQP1D(aCoeff, bCoeff, cCoeff, 
                    static_cast<Scalar>(0), 
                    theta[0] == static_cast<Scalar>(0)
                    ? std::numeric_limits<Scalar>::max()
                    : static_cast<Scalar>(1) / theta[0]);

                Scalar thetaI = theta[0];
                if (thetaI == static_cast<Scalar>(0))
                {
                    return (primal - dual) / primal;
                }

                bool dualValueFound = false;

                bool accumulationTreated = false;

                for (Index i = 1; i < theta.size() && theta[i] != static_cast<Scalar>(0); ++i)
                {
                    accumulationTreated = false;

                    auto thetaIPlus1 = theta[i];

                    aCoeff -= betaDeltaSquared * thetaI * thetaI;
                    cCoeff += betaDeltaSquared;

                    while (thetaIPlus1 >= thetaI)
                    {
                        aCoeff -= betaDeltaSquared * thetaI * thetaI;
                        cCoeff += betaDeltaSquared;

                        ++i;

                        if (i < theta.size() || theta[i] != static_cast<Scalar>(0))
                        {
                            thetaIPlus1 = theta[i];
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (thetaIPlus1 < thetaI)
                    {
                        accumulationTreated = true;

                        // solve on [s0,s1]
                        const auto duali = Utils::solveMaxConcaveQP1D(aCoeff, bCoeff, cCoeff, 
                            static_cast<Scalar>(1) / thetaI, static_cast<Scalar>(1) / thetaIPlus1);

                        if (duali > dual)
                        {
                            dual = duali;
                        }
                        else
                        {
                            dualValueFound = true;
                            break;
                        }

                        //s0 = s1;
                        thetaI = thetaIPlus1;
                    }
                }

                if (!dualValueFound)
                {
                    if (accumulationTreated)
                    {
                        aCoeff -= betaDeltaSquared * thetaI * thetaI;
                        cCoeff += betaDeltaSquared;
                    }

                    // solve on [s0,s1]
                    auto duali = Utils::solveMaxConcaveQP1D(aCoeff, bCoeff, cCoeff, 
                        static_cast<Scalar>(1) / thetaI, std::numeric_limits<Scalar>::max());

                    if (duali > dual)
                    {
                        dual = duali;
                    }
                }

                return (primal - dual) / primal;
            }

            template<std::floating_point ScalarType>
            using L0L2Regressor = CyclicCoordinateDescent<L0L2ModelImplementation<ScalarType>>;
        }
    }
}
#endif //L0L2_COORDINATE_DESCENT_SOLVER_HPP
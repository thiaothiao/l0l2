#ifndef L0L2_ELASTICNET_COORDINATE_DESCENT_SOLVER_HPP
#define L0L2_ELASTICNET_COORDINATE_DESCENT_SOLVER_HPP

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
            // ELASTICNET coordinate descent stepJ implementation
            template<std::floating_point ScalarType>
            class ELASTICNETModelImplementation final : public ModelImplementationBase<ScalarType>
            {
            public:
                using Base = ModelImplementationBase<ScalarType>;
                using typename Base::Scalar;

                struct Param final
                {
                    Param(
                        Scalar gammaInput = static_cast<Scalar>(0),
                        Scalar betaInput = static_cast<Scalar>(1),
                        bool hasInterceptInput = false,
                        Strategy strategyInput = Strategy::FromZeroSolution,
                        Scalar toleranceInput = static_cast<Scalar>(1e-4),
                        unsigned int maximumNumberOfIterationsInput = 10000U,
                        Scalar innerEpsilonInput = static_cast<Scalar>(1e-6),
                        unsigned int innerMaximumNumberOfIterationsInput = 100000U)
                        :gamma{ gammaInput },
                        beta{ betaInput },
                        hasIntercept{ hasInterceptInput },
                        strategy{ strategyInput },
                        tolerance{ toleranceInput },
                        maximumNumberOfIterations{ maximumNumberOfIterationsInput },
                        innerEpsilon{ innerEpsilonInput },
                        innerMaximumNumberOfIterations{ innerMaximumNumberOfIterationsInput },
                        gammaOver2{ gammaInput / static_cast<Scalar>(2) }
                    {
                    }

                    Param(const Param&) = default;
                    Param& operator=(const Param&) = default;

                    Param(Param&&) = default;
                    Param& operator=(Param&&) = default;

                    const Scalar gamma;
                    const Scalar beta;
                    const bool hasIntercept;
                    const Strategy strategy;
                    const Scalar tolerance;
                    const unsigned int maximumNumberOfIterations;
                    const Scalar innerEpsilon;
                    const unsigned int innerMaximumNumberOfIterations;
                    const Scalar gammaOver2;
                };

                ELASTICNETModelImplementation(const Param& paramInput = {})
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
            inline ELASTICNETModelImplementation<ScalarType>::Scalar
                ELASTICNETModelImplementation<ScalarType>::stepJ(Scalar zJ, Scalar uJ) const
            {
                return (std::abs(uJ) > param.gammaOver2) ?
                    ((uJ - Utils<Scalar>::sign(uJ) * param.gammaOver2) / (param.beta + zJ))
                    : static_cast<Scalar>(0);
            }

            template<std::floating_point ScalarType>
            inline ELASTICNETModelImplementation<ScalarType>::Scalar
                ELASTICNETModelImplementation<ScalarType>::otherJ([[maybe_unused]] Scalar zJ, [[maybe_unused]] Scalar uJ) const
            {
                return static_cast<Scalar>(0);
            }

            template<std::floating_point ScalarType>
            ELASTICNETModelImplementation<ScalarType>::Scalar
                ELASTICNETModelImplementation<ScalarType>::computeDualityGap(const Matrix<Scalar>& matData, 
                    const Vector<Scalar>& vectData, bool matrixIsCovariance, 
                    [[maybe_unused]] const CoordinateStates& coordinateStates, const Solution<Scalar>& solution) const
            {
                using Vector = Vector<Scalar>;
                using Utils = Utils<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

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

                auto leastSquaresPartValue = matrixIsCovariance ? nu0.dot(x - vectData) : nu0.squaredNorm();

                const auto betaXSquaredNorm = param.beta * x.squaredNorm();

                const auto primal = leastSquaresPartValue + betaXSquaredNorm + param.gamma * x.cwiseAbs().sum();

                if (hasIntercept)
                {
                    nu0.array() -= nu0.mean();

                    leastSquaresPartValue = nu0.squaredNorm();
                }

                const auto aTnu0NormInf = matrixIsCovariance ? (nu0 + param.beta * x).cwiseAbs().maxCoeff()
                    : (matData.transpose() * nu0 + param.beta * x).cwiseAbs().maxCoeff();

                const auto absSBound = param.gamma / aTnu0NormInf;
                
                const auto bTnu0 = vectData.dot(nu0);

                auto s = - static_cast<Scalar>(2) * bTnu0 / (leastSquaresPartValue + betaXSquaredNorm);

                if (std::abs(s) > absSBound)
                {
                    s = Utils::sign(s) * absSBound;
                }

                const auto dual = -static_cast<Scalar>(0.25) * (leastSquaresPartValue + betaXSquaredNorm) * s * s
                    -  s * bTnu0;

                return (primal - dual) / primal;
            }

            template<std::floating_point ScalarType>
            using ELASTICNETRegressor = CyclicCoordinateDescent<ELASTICNETModelImplementation<ScalarType>>;
        }
    }
}
#endif //L0L2_ELASTICNET_COORDINATE_DESCENT_SOLVER_HPP
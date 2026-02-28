#ifndef L0L2_COORDINATE_DESCENT_SOLVER_H
#define L0L2_COORDINATE_DESCENT_SOLVER_H

#include <future>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <utility>

#include "leastsquares/utils.h"
#include "leastsquares/cycliccoordinatedescent/generic.h"

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
                    Param(
                        Scalar deltaInput = static_cast<Scalar>(0),
                        Scalar betaInput = static_cast<Scalar>(1),
                        Strategy strategyInput = Strategy::FromZeroSolution,
                        Scalar toleranceInput = static_cast<Scalar>(1e-4),
                        unsigned int maximumNumberOfIterationsInput = 10000U,
                        Scalar innerEpsilonInput = static_cast<Scalar>(1e-6),
                        unsigned int innerMaximumNumberOfIterationsInput = 100000U)
                        :delta{ deltaInput },
                        beta{ betaInput },
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
                    const Strategy strategy;
                    const Scalar tolerance;
                    const unsigned int maximumNumberOfIterations;
                    const Scalar innerEpsilon;
                    const unsigned int innerMaximumNumberOfIterations;
                    const Scalar deltaBeta;
                };

                L0L2ModelImplementation() : Base()
                {
                }

                static Scalar stepJ(const Param& param, Scalar zJ, Scalar uJ);
            };


            template<std::floating_point ScalarType>
            inline L0L2ModelImplementation<ScalarType>::Scalar
                L0L2ModelImplementation<ScalarType>::stepJ(const Param& param, Scalar zJ, Scalar uJ)
            {
                return (std::abs(uJ) >= param.deltaBeta + zJ * param.delta) ? (uJ / (param.beta + zJ))
                    : ((std::abs(uJ) > param.deltaBeta) ? ((uJ - param.deltaBeta * Utils<ScalarType>::sign(uJ)) / zJ)
                        : static_cast<Scalar>(0));
            }

            template<std::floating_point ScalarType>
            using L0L2Regressor = CyclicalCoordinateDescent<L0L2ModelImplementation<ScalarType>>;
        }
    }
}
#endif //L0L2_COORDINATE_DESCENT_SOLVER_H
#ifndef L0L2_SPARSE_PCA_H
#define L0L2_SPARSE_PCA_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <future>
#include <limits>
#include <concepts>

#include "leastsquares/utils.h"
#include "leastsquares/cycliccoordinatedescent/solver.h"
#include "leastsquares/fullpath/solver.h"
#include "pca/generic.h"

namespace l0l2
{
    namespace linearmodel
    {
        template<std::floating_point ScalarType>
        class L0L2SPCAModelImplementation final
        {
        public:
            using Scalar = ScalarType;
            using Regressor = leastsquares::L0L2Regressor<Scalar>;
            using RegressorParam = Regressor::Param;

            struct Param final
            {
                Param(
                    Scalar deltaInput = static_cast<Scalar>(0),
                    Scalar betaInput = static_cast<Scalar>(1),
                    Index nbComponentsInput = static_cast<Index>(2),
                    leastsquares::Strategy strategyInput = leastsquares::Strategy::FromZeroSolution,
                    Scalar toleranceInput = static_cast<Scalar>(1e-4),
                    unsigned int maximumNumberOfIterationsInput = 10000U,
                    Scalar innerEpsilonInput = static_cast<Scalar>(1e-6),
                    unsigned int innerMaximumNumberOfIterationsInput = 100000U)
                    :regressorParam{ deltaInput, betaInput , strategyInput, 
                    toleranceInput, maximumNumberOfIterationsInput,
                    innerEpsilonInput, innerMaximumNumberOfIterationsInput },
                    nbComponents{ nbComponentsInput }//to optimize
                {
                }

                Param(const Param&) = default;
                Param& operator=(const Param&) = default;

                Param(Param&&) = default;
                Param& operator=(Param&&) = default;

                const RegressorParam regressorParam;
                const Index nbComponents;
            };

            L0L2SPCAModelImplementation()
            {
            }

            static Vector<Scalar> fitRegressor(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                const Param& param);
        };

        template<std::floating_point ScalarType>
        class FullPathL0L2SPCAModelImplementation final
        {
        public:
            using Scalar = ScalarType;
            using Regressor = leastsquares::FullPathSolver<Scalar>;
            using RegressorParam = Regressor::Param;

            struct Param final
            {
                Param(
                    Scalar deltaInput = static_cast<Scalar>(0),
                    Scalar betaInput = static_cast<Scalar>(1),
                    Index nbComponentsInput = static_cast<Index>(2),
                    leastsquares::Strategy strategyInput = leastsquares::Strategy::FromZeroSolution)
                    :regressorParam{ deltaInput, betaInput , strategyInput},
                    nbComponents{ nbComponentsInput }//to optimize
                {
                }

                Param(const Param&) = default;
                Param& operator=(const Param&) = default;

                Param(Param&&) = default;
                Param& operator=(Param&&) = default;

                const RegressorParam regressorParam;
                const Index nbComponents;
            };

            FullPathL0L2SPCAModelImplementation()
            {
            }

            static Vector<Scalar> fitRegressor(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                const Param& param);
        };

        template<std::floating_point ScalarType>
        inline Vector<typename L0L2SPCAModelImplementation<ScalarType>::Scalar>
            L0L2SPCAModelImplementation<ScalarType>::fitRegressor(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                const Param& param)
        {
            return Regressor(param.regressorParam)
                .fit(matData, vectData, matrixIsCovariance)
                .x;
        }

        template<std::floating_point ScalarType>
        inline Vector<typename FullPathL0L2SPCAModelImplementation<ScalarType>::Scalar>
            FullPathL0L2SPCAModelImplementation<ScalarType>::fitRegressor(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                const Param& param)
        {
            return Regressor(param.regressorParam).
                fit(matData, vectData, matrixIsCovariance)
                .x;
        }

        template <std::floating_point ScalarType>
        using L0L2SPCA = SPCA<L0L2SPCAModelImplementation<ScalarType>>;

        template <std::floating_point ScalarType>
        using FullPathL0L2SPCA = SPCA<FullPathL0L2SPCAModelImplementation<ScalarType>>;
    }
}
#endif //L0L2_SPARSE_PCA_H

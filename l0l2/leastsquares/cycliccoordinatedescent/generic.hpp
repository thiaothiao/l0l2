#ifndef L0L2_GENERIC_COORDINATE_DESCENT_SOLVER_HPP
#define L0L2_GENERIC_COORDINATE_DESCENT_SOLVER_HPP

#include <future>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <utility>

#include "l0l2/leastsquares/utils.hpp"
#include "l0l2/leastsquares/l2regressor/solver.hpp"

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            /*! \brief Cyclical Coordinate Descent Regressor Model concept.
            */
            template <class ModelImplementationType>
            concept ModelLike = requires(ModelImplementationType impl)
            {
                {
                    std::as_const(impl).stepJ(typename ModelImplementationType::Scalar{},
                        typename ModelImplementationType::Scalar{})
                } ->std::convertible_to<typename ModelImplementationType::Scalar>;

                {
                    std::as_const(impl).computeDualityGap(
                        Matrix<typename ModelImplementationType::Scalar>{}, 
                        Vector<typename ModelImplementationType::Scalar>{},
                        Solution<typename ModelImplementationType::Scalar>{},
                        bool{})
                } ->std::convertible_to<typename ModelImplementationType::Scalar>;
            };

            /*! \brief Cyclical Coordinate Descent Regressor class.
            *
            * It produces solutions using cyclical coordinate descent algorithm.
            */
            template <ModelLike ModelImplementationType>
            class CyclicCoordinateDescent final
            {
            public:
                using ModelImplementation = ModelImplementationType;/*!< Alias for the model implementation type */
                using Param = typename ModelImplementation::Param; /*!< Alias for the used parameter type */
                using Scalar = typename ModelImplementation::Scalar; /*!< Alias for the used scalar type */

                /*! \brief A cyclic coordinate descent solver object constructor.
                  \param param regularization parameters.
                  \param withIntercept boolean indicating with intercept or not. Default is false.
                */
                CyclicCoordinateDescent(const Param& param)
                    :m_Param{ param },
                    m_FromBothConverged{}
                {
                }

                /*! \brief Fit model.
                   \param matData contiguous data container representing matrix in column major layout.
                   \param vectData contiguous data container representing target vector.
                   \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
                   \return a solution in the format Solution.
                 */
                Solution<Scalar> fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance);

            private:
                Solution<Scalar>  fitFrom(
                    const Matrix<Scalar>& matData,
                    bool matrixIsCovariance,
                    const Vector<Scalar>& vectDataPtr,
                    Solution<Scalar>&& initialSolution);

                const Param m_Param;

                std::atomic_flag m_FromBothConverged;
            };

            template <ModelLike ModelImplementationType>
            Solution<typename ModelImplementationType::Scalar>
                CyclicCoordinateDescent<ModelImplementationType>::fitFrom(
                    const Matrix<Scalar>& matData,
                    bool matrixIsCovariance,
                    const Vector<Scalar>& vectData,
                    Solution<Scalar>&& initialSolution)
            {
                using Vector = Vector<Scalar>;
                using Solution = Solution<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                Solution solution = std::move(initialSolution);
                auto& x = solution.x;
                auto& intercept = solution.intercept;
                auto& dualityGap = solution.dualityGap;

                const auto hasIntercept = !matrixIsCovariance && m_Param.hasIntercept;
                if (!hasIntercept)
                {
                    intercept = static_cast<Scalar>(0);
                }

                auto R = matrixIsCovariance
                    ? static_cast<Vector>(matData * (vectData - x))
                    : static_cast<Vector>(vectData - matData * x);

                if (hasIntercept)
                {
                    R.array() -= intercept;
                }

                const auto zJs = matrixIsCovariance ? static_cast<Vector>(matData.diagonal())
                    : static_cast<Vector>(matData.colwise().squaredNorm().transpose());

                const ModelImplementation modelImplementation{ m_Param };

                unsigned int numberOfIterations = 0U;
                while (true)
                {
                    auto globalChange = static_cast<Scalar>(0);

                    // treat intercept first if any
                    if (hasIntercept)
                    {
                        const auto zJ = static_cast<Scalar>(matData.rows());

                        const auto oldIntercept = intercept;

                        const auto uIntercept = R.sum() + zJ * oldIntercept;

                        intercept = uIntercept / zJ;

                        const auto interceptDiff = oldIntercept - intercept;

                        if (interceptDiff != static_cast<Scalar>(0))
                        {
                            R.array() += interceptDiff;

                            const auto interceptChange = std::abs(interceptDiff);

                            if (interceptChange > globalChange)
                            {
                                globalChange = interceptChange;
                            }
                        }
                    }

                    for (Index j = 0; j < n; ++j)
                    {
                        const auto zJ = zJs[j];

                        if (zJ == static_cast<Scalar>(0))
                        {
                            continue;
                        }

                        const auto oldxJ = x[j];

                        const auto uJ = matrixIsCovariance ? R[j] + zJ * oldxJ : matData.col(j).dot(R) + zJ * oldxJ;

                        const auto newxJ = modelImplementation.stepJ(zJ, uJ);

                        x[j] = newxJ;

                        const auto xsDiff = oldxJ - newxJ;

                        if (xsDiff != static_cast<Scalar>(0))
                        {
                            R += xsDiff * matData.col(j);

                            const auto xJChange = std::abs(xsDiff);

                            if (xJChange > globalChange)
                            {
                                globalChange = xJChange;
                            }
                        }
                    }

                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        if (globalChange <= m_Param.tolerance
                            || numberOfIterations >= m_Param.maximumNumberOfIterations)
                        {
                            m_FromBothConverged.test_and_set(std::memory_order_relaxed);
                        }
                        else if (m_FromBothConverged.test(std::memory_order_relaxed))
                        {
                            break;
                        }
                    }

                    if (globalChange <= m_Param.tolerance)
                    {
                        break;
                    }

                    if (numberOfIterations >= m_Param.maximumNumberOfIterations)
                    {
                        break;
                    }

                    ++numberOfIterations;
                }

                dualityGap = modelImplementation.computeDualityGap(
                    matData, vectData, solution, matrixIsCovariance);

                return solution;
            }

            template <ModelLike ModelImplementationType>
            Solution<typename ModelImplementationType::Scalar>
                CyclicCoordinateDescent<ModelImplementationType>::fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance)
            {
                using Vector = Vector<Scalar>;
                using L2Regressor = L2RegressorPCG<Scalar>;
                using Solution = Solution<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto hasIntercept = !matrixIsCovariance && m_Param.hasIntercept;

                if (m_Param.strategy != Strategy::FromBothSolutions)
                {
                    return fitFrom(matData,
                        matrixIsCovariance,
                        vectData,
                        m_Param.strategy == Strategy::FromZeroSolution
                        ? Solution(Vector::Zero(n))
                        : L2Regressor(m_Param.beta, hasIntercept).fit(matData, vectData, matrixIsCovariance,
                            m_Param.innerEpsilon,
                            m_Param.innerMaximumNumberOfIterations));
                }
                else
                {
                    m_FromBothConverged.clear(std::memory_order_relaxed);

                    auto fromZeroFuture =
                        std::async(std::launch::async, &CyclicCoordinateDescent::fitFrom, this,
                            matData,
                            matrixIsCovariance,
                            vectData,
                            Solution(Vector::Zero(n)));

                    auto fromL2Solution = fitFrom(matData,
                        matrixIsCovariance,
                        vectData,
                        L2Regressor(m_Param.beta, hasIntercept).fit(matData, vectData, matrixIsCovariance,
                            m_Param.innerEpsilon,
                            m_Param.innerMaximumNumberOfIterations));

                    auto fromZeroSolution = fromZeroFuture.get();

                    if (fromZeroSolution.dualityGap <= fromL2Solution.dualityGap)
                    {
                        return fromZeroSolution;
                    }
                    else
                    {
                        return fromL2Solution;
                    }
                }
            }
        }
    }
}
#endif //L0L2_GENERIC_COORDINATE_DESCENT_SOLVER_HPP
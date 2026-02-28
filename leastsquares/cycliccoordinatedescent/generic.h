#ifndef L0L2_GENERIC_COORDINATE_DESCENT_SOLVER_H
#define L0L2_GENERIC_COORDINATE_DESCENT_SOLVER_H

#include <future>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <utility>

#include "leastsquares/utils.h"
#include "leastsquares/l2regressor/solver.h"

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            /*! \brief Cyclical Coordinate Descent Regressor Model concept.
            */
            template <class ModelImplementationType>
            concept ModelLike = requires()
            {
                {
                    ModelImplementationType::stepJ(typename ModelImplementationType::Param{},
                        typename ModelImplementationType::Scalar{},
                        typename ModelImplementationType::Scalar{})
                } ->std::convertible_to<typename ModelImplementationType::Scalar>;

                //TODO add constraint for duality gap computations
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
                CyclicCoordinateDescent(const Param& param,
                    bool withIntercept = false)
                    :m_Param{ param },
                    m_WithIntercept{ withIntercept },
                    m_FromBothConverged{}
                {
                }

                /*! \brief Fit model.
                   \param matData contiguous data container representing matrix in column major layout.
                   \param vectData contiguous data container representing target vector.
                   \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
                   \return a solution in the format CDSolution.
                 */
                CDSolution<Scalar> fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance);

            private:
                CDSolution<Scalar> fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance);

                CDSolution<Scalar>  fitFrom(
                    const Matrix<Scalar>& matData,
                    bool matrixIsCovariance,
                    const Vector<Scalar>& vectDataPtr,
                    Vector<Scalar>&& w0);

                const Param m_Param;

                const bool m_WithIntercept;

                std::atomic_flag m_FromBothConverged;
            };

            template<std::floating_point ScalarType>
            class ModelImplementationBase
            {
            public:
                ModelImplementationBase()
                {
                }

                using Scalar = ScalarType;
            };

            template <ModelLike ModelImplementationType>
            CDSolution<typename ModelImplementationType::Scalar>
                CyclicCoordinateDescent<ModelImplementationType>::fitFrom(
                    const Matrix<Scalar>& matData,
                    bool matrixIsCovariance,
                    const Vector<Scalar>& vectData,
                    Vector<Scalar>&& w0)
            {
                using Vector = Vector<Scalar>;
                using CDSolution = CDSolution<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                CDSolution solution{ n };
                solution.x = std::move(w0);

                auto& x = solution.x;
                auto& numberOfIterations = solution.numberOfIterations;
                auto& globalChange = solution.globalChange;
                auto& dualityGap = solution.dualityGap;
                auto& status = solution.status;

                auto R = matrixIsCovariance
                    ? static_cast<Vector>(matData * (vectData - x))
                    : static_cast<Vector>(vectData - matData * x);

                const auto zJs = matrixIsCovariance ? static_cast<Vector>(matData.diagonal())
                    : static_cast<Vector>(matData.colwise().squaredNorm().transpose());

                numberOfIterations = 0U;

                while (true)
                {
                    globalChange = static_cast<Scalar>(0);

                    for (Index j = 0; j < n; ++j)
                    {
                        const auto zJ = zJs[j];

                        if (zJ == static_cast<Scalar>(0))
                        {
                            continue;
                        }

                        const auto oldxJ = x[j];

                        const auto uJ = matrixIsCovariance ? R[j] + zJ * oldxJ : matData.col(j).dot(R) + zJ * oldxJ;

                        const auto newxJ = ModelImplementation::stepJ(m_Param, zJ, uJ);

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
                        status = CDStatus::Converged;
                        break;
                    }

                    if (numberOfIterations >= m_Param.maximumNumberOfIterations)
                    {
                        status = CDStatus::LimitReached;
                        break;
                    }

                    ++numberOfIterations;
                }

                return solution;
            }

            template <ModelLike ModelImplementationType>
            CDSolution<typename ModelImplementationType::Scalar>
                CyclicCoordinateDescent<ModelImplementationType>::fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance)
            {
                const auto withIntercept = m_WithIntercept && !matrixIsCovariance;

                auto solution = fitNoIntercept(withIntercept ? (matData.rowwise() - matData.colwise().mean()).eval()
                    : matData,
                    withIntercept ? (vectData.array() - vectData.mean()).matrix() : vectData,
                    matrixIsCovariance);

                if (withIntercept)
                {
                    solution.intercept = (vectData - matData * solution.x).mean();
                }

                return solution;
            }

            template <ModelLike ModelImplementationType>
            CDSolution<typename ModelImplementationType::Scalar>
                CyclicCoordinateDescent<ModelImplementationType>::fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance)
            {
                using Vector = Vector<Scalar>;
                using L2Regressor = L2RegressorPCG<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                if (m_Param.strategy != Strategy::FromBothSolutions)
                {
                    return fitFrom(matData,
                        matrixIsCovariance,
                        vectData,
                        m_Param.strategy == Strategy::FromZeroSolution
                        ? Vector::Zero(n)
                        : L2Regressor(m_Param.beta).fitNoIntercept(matData, vectData, matrixIsCovariance,
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
                            Vector::Zero(n));

                    auto fromL2Solution = fitFrom(matData,
                        matrixIsCovariance,
                        vectData,
                        L2Regressor(m_Param.beta).fitNoIntercept(matData, vectData, matrixIsCovariance,
                            m_Param.innerEpsilon,
                            m_Param.innerMaximumNumberOfIterations));

                    auto fromZeroSolution = fromZeroFuture.get();

                    if (fromZeroSolution.status == CDStatus::Converged)
                    {
                        return fromZeroSolution;
                    }
                    else if (fromL2Solution.status == CDStatus::Converged)
                    {
                        return fromL2Solution;
                    }
                    else if (fromZeroSolution.globalChange <= fromL2Solution.globalChange)
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
#endif //L0L2_GENERIC_COORDINATE_DESCENT_SOLVER_H
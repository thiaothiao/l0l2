#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <concepts>
#include <future>
#include <utility>

#include <l0l2/leastsquares/l2regressor/solver.hpp>
#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            /**
             * @brief Concept for a type that can be used as coordinate descent
             * concrete implementation in the generic solver.
             * @details This concept ensures the type supports
             * @param Scalar an inner type used as scalar type.
             * @param stepJ a const member function that produces a scalar as a
             * new value for coordinate j.
             * @param stepIntercept a const member function that produces a
             * scalar as a new value for the intercept.
             * @param otherJ a const member function that produces a scalar as a
             * new value for coordinate j if not concerned by the coordinate
             * descent scheme.
             * @param computeDualityGap a const member function that produces a
             * scalar representing a duality gap.
             * @tparam ImplementationType The type to check.
             */
            template <class ImplementationType>
            concept CyclicCoordinateDescentImplementationLike = requires(
                ImplementationType impl) {
                {
                    std::as_const(impl).stepJ(
                        typename ImplementationType::Scalar{},
                        typename ImplementationType::Scalar{})
                } -> std::convertible_to<typename ImplementationType::Scalar>;

                {
                    std::as_const(impl).stepIntercept(
                        typename ImplementationType::Scalar{},
                        typename ImplementationType::Scalar{})
                } -> std::convertible_to<typename ImplementationType::Scalar>;

                {
                    std::as_const(impl).otherJ(
                        typename ImplementationType::Scalar{},
                        typename ImplementationType::Scalar{})
                } -> std::convertible_to<typename ImplementationType::Scalar>;

                {
                    std::as_const(impl).computeDualityGap(
                        Matrix<typename ImplementationType::Scalar>{},
                        Vector<typename ImplementationType::Scalar>{},
                        CoordinateStates{},
                        Solution<typename ImplementationType::Scalar>{})
                } -> std::convertible_to<typename ImplementationType::Scalar>;
            };

            /*! \brief Cyclical Coordinate Descent Regressor class.
             *
             * It produces solutions using cyclical coordinate descent
             * algorithm.
             */
            template <
                CyclicCoordinateDescentImplementationLike ImplementationType>
            class CyclicCoordinateDescent final
            {
              public:
                using Implementation =
                    ImplementationType; /*!< Alias for the implementation type
                                         */
                using Param =
                    typename Implementation::Param; /*!< Alias for the used
                                                       parameter type */
                using Scalar =
                    typename Implementation::Scalar; /*!< Alias for the used
                                                        scalar type */

                /*! \brief A cyclic coordinate descent solver object
                  constructor.
                  \param param regularization parameters.
                  \param withIntercept boolean indicating with intercept or not.
                  Default is false.
                */
                CyclicCoordinateDescent(const Param &param)
                    : m_Param{param}, m_ParallelConverged{}
                {
                }

                /*! \brief Fit model.
                   \param matData contiguous data container representing matrix
                   in column major layout.
                   \param vectData contiguous data container representing target
                   vector.
                   \return a solution in the format Solution.
                 */
                Solution<Scalar>
                fit(const Matrix<Scalar> &matData,
                    const Vector<Scalar> &vectData,
                    const CoordinateStates &coordinateStates = {});

              private:
                Solution<Scalar>
                fitFrom(const Matrix<Scalar> &matData,
                        const Vector<Scalar> &vectData,
                        Solution<Scalar> &&initialSolution,
                        const CoordinateStates &coordinateStates);

                const Param m_Param;

                std::atomic_flag m_ParallelConverged;
            };

            template <
                CyclicCoordinateDescentImplementationLike ImplementationType>
            Solution<
                typename CyclicCoordinateDescent<ImplementationType>::Scalar>
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                Solution<Scalar> &&initialSolution,
                const CoordinateStates &coordinateStates)
            {
                using Vector = Vector<Scalar>;
                using Solution = Solution<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                Solution solution = std::move(initialSolution);
                auto &x = solution.x;
                auto &intercept = solution.intercept;
                auto &dualityGap = solution.dualityGap;

                const auto subIndicesCase =
                    static_cast<Index>(coordinateStates.size()) == n;

                if (subIndicesCase)
                {
                    for (Index j = 0; j < n; ++j)
                    {
                        if (coordinateStates[j] == CoordinateState::Zero)
                        {
                            x[j] = static_cast<Scalar>(0);
                        }
                    }
                }

                const auto hasIntercept = m_Param.hasIntercept;
                if (!hasIntercept)
                {
                    intercept = static_cast<Scalar>(0);
                }

                Vector R = vectData - matData * x;
                if (hasIntercept)
                {
                    R.array() -= intercept;
                }

                const Vector zJs = matData.colwise().squaredNorm().transpose();

                const Implementation Implementation{m_Param};

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

                        intercept =
                            Implementation.stepIntercept(zJ, uIntercept);

                        const auto interceptDiff = oldIntercept - intercept;

                        if (interceptDiff != static_cast<Scalar>(0))
                        {
                            R.array() += interceptDiff;

                            const auto interceptChange =
                                std::abs(interceptDiff);

                            if (interceptChange > globalChange)
                            {
                                globalChange = interceptChange;
                            }
                        }
                    }

                    for (Index j = 0; j < n; ++j)
                    {
                        if (subIndicesCase &&
                            coordinateStates[j] == CoordinateState::Zero)
                        {
                            continue;
                        }

                        const auto zJ = zJs[j];

                        if (zJ == static_cast<Scalar>(0))
                        {
                            continue;
                        }

                        const auto oldxJ = x[j];

                        const auto uJ = matData.col(j).dot(R) + zJ * oldxJ;

                        const auto newxJ =
                            (subIndicesCase &&
                             coordinateStates[j] == CoordinateState::Nonzero)
                                ? Implementation.otherJ(zJ, uJ)
                                : Implementation.stepJ(zJ, uJ);

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

                    if (m_Param.strategy == Strategy::Parallel)
                    {
                        if (globalChange <= m_Param.tolerance ||
                            numberOfIterations >=
                                m_Param.maximumNumberOfIterations)
                        {
                            m_ParallelConverged.test_and_set(
                                std::memory_order_relaxed);
                        }
                        else if (m_ParallelConverged.test(
                                     std::memory_order_relaxed))
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

                dualityGap = Implementation.computeDualityGap(
                    matData, vectData, coordinateStates, solution);

                return solution;
            }

            template <
                CyclicCoordinateDescentImplementationLike ImplementationType>
            Solution<
                typename CyclicCoordinateDescent<ImplementationType>::Scalar>
            CyclicCoordinateDescent<ImplementationType>::fit(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                const CoordinateStates &coordinateStates)
            {
                using Vector = Vector<Scalar>;
                using L2Regressor = PCGL2Regressor<Scalar>;
                using Solution = Solution<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto hasIntercept = m_Param.hasIntercept;

                if (m_Param.strategy != Strategy::Parallel)
                {
                    return fitFrom(
                        matData, vectData,
                        m_Param.strategy == Strategy::SequentialFromZeroSolution
                            ? Solution(Vector::Zero(n))
                            : L2Regressor(m_Param.beta, hasIntercept)
                                  .fit(matData, vectData, m_Param.innerEpsilon,
                                       m_Param.innerMaximumNumberOfIterations,
                                       {}, coordinateStates),
                        coordinateStates);
                }
                else
                {
                    m_ParallelConverged.clear(std::memory_order_relaxed);

                    auto fromZeroFuture = std::async(
                        std::launch::async, &CyclicCoordinateDescent::fitFrom,
                        this, matData, vectData, Solution(Vector::Zero(n)),
                        coordinateStates);

                    auto fromL2Solution = fitFrom(
                        matData, vectData,
                        L2Regressor(m_Param.beta, hasIntercept)
                            .fit(matData, vectData, m_Param.innerEpsilon,
                                 m_Param.innerMaximumNumberOfIterations, {},
                                 coordinateStates),
                        coordinateStates);

                    auto fromZeroSolution = fromZeroFuture.get();

                    if (fromZeroSolution.dualityGap <=
                        fromL2Solution.dualityGap)
                    {
                        return fromZeroSolution;
                    }
                    else
                    {
                        return fromL2Solution;
                    }
                }
            }

            template <std::floating_point ScalarType> class ImplementationBase
            {
              public:
                using Scalar = ScalarType;

                Scalar stepIntercept(Scalar zJ, Scalar uIntercept) const;
            };

            template <std::floating_point ScalarType>
            inline ImplementationBase<ScalarType>::Scalar
            ImplementationBase<ScalarType>::stepIntercept(
                Scalar zJ, Scalar uIntercept) const
            {
                return uIntercept / zJ;
            }
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

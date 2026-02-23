#ifndef L0L2_COORDINATE_DESCENT_SOLVER_H
#define L0L2_COORDINATE_DESCENT_SOLVER_H

#include <future>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <utility>

#include "Utils.h"

namespace l0l2
{
    namespace linearmodel
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
        class CyclicalCoordinateDescent final
        {
        public:
            using ModelImplementation = ModelImplementationType;/*!< Alias for the model implementation type */
            using Param = typename ModelImplementation::Param; /*!< Alias for the used parameter type */
            using Scalar = typename ModelImplementation::Scalar; /*!< Alias for the used scalar type */

            /*! \brief A solver object constructor.
              \param delta sparsity regularization parameter.
              \param beta l2 regularization parameter.
              \param strategy enum indicating a strategy: from zero, or l2 or both solutions.
              \param tolerance covergence tolerance on the coordinates changes.
              \param maximumNumberOfIterations maximum number of iterations allowed.
            */
            CyclicalCoordinateDescent(const Param& param,
                bool withIntercept = false)
                :m_Param{param},
                m_WithIntercept{ withIntercept },
                m_FromBothConverged{}
            {
            }

            /*! \brief Fit full path solutions.
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
        class L2RegressorPCG
        {
        public:
            using Scalar = ScalarType;

            L2RegressorPCG(Scalar beta) : m_Beta{ beta }
            {
            }

            Vector<Scalar> fitNoIntercept(
                const Matrix<Scalar>& matData/*colmajor*/,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar epsilon,
                unsigned int maxNumberOfIterations) const;

        private:
            const Scalar m_Beta;
        };

        template<std::floating_point ScalarType>
        Vector<typename L2RegressorPCG<ScalarType>::Scalar>
            L2RegressorPCG<ScalarType>::fitNoIntercept(
                const Matrix<Scalar>& matData/*colmajor*/,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar epsilon,
                unsigned int maxNumberOfIterations) const
        {
            using Vector = Vector<Scalar>;
            using Utils = Utils<Scalar>;

            const auto n = static_cast<Index>(matData.cols());

            Vector x = Vector::Zero(n);

            const Vector M = matrixIsCovariance ?
                static_cast<Vector>(matData.diagonal().array() + m_Beta)
                : static_cast<Vector>(matData.cwiseAbs2().colwise().sum().transpose().array() + m_Beta);

            Vector r = matrixIsCovariance ?
                static_cast<Vector>(matData * (vectData - x) + m_Beta * x)
                : static_cast<Vector>(matData.transpose() * (vectData - matData * x) + m_Beta * x);

            Vector p = r.cwiseQuotient(M);

            Scalar r0dotz0 = r.dot(p);

            {
                const Vector w = matrixIsCovariance ?
                    static_cast<Vector>(matData * p + m_Beta * p)
                    : static_cast<Vector>(matData.transpose() * (matData * p) + m_Beta * p);
                
                const Scalar alphaValue = r0dotz0 / p.dot(w);

                x += alphaValue * p;
                r -= alphaValue * w;
            }

            int iter = 0;
            while (r.norm() > epsilon)
            {
                const Vector z = r.cwiseQuotient(M);// Mzk = rk

                const Scalar rdotz = r.dot(z);

                p = (rdotz / r0dotz0) * p + z;

                const Vector w = matrixIsCovariance ?
                    static_cast<Vector>(matData * p + m_Beta * p)
                    : static_cast<Vector>(matData.transpose() * (matData * p) + m_Beta * p);

                const Scalar alphaValue = rdotz / p.dot(w);

                x += alphaValue * p;
                r -= alphaValue * w;

                r0dotz0 = rdotz;

                ++iter;

                if (iter > maxNumberOfIterations)
                {
                    break;
                }
            }
            
            return x;
        }


        template<std::floating_point ScalarType>
        inline L0L2ModelImplementation<ScalarType>::Scalar
            L0L2ModelImplementation<ScalarType>::stepJ(const Param& param, Scalar zJ, Scalar uJ)
        {
            return (std::abs(uJ) >= param.deltaBeta + zJ * param.delta) ? (uJ / (param.beta + zJ))
                : ((std::abs(uJ) > param.deltaBeta) ? ((uJ - param.deltaBeta * Utils<ScalarType>::sign(uJ)) / zJ)
                    : static_cast<Scalar>(0));
        }

        template <ModelLike ModelImplementationType>
        CDSolution<typename ModelImplementationType::Scalar>
            CyclicalCoordinateDescent<ModelImplementationType>::fitFrom(
                const Matrix<Scalar>& matData,
                bool matrixIsCovariance,
                const Vector<Scalar>& vectData,
                Vector<Scalar>&& w0)
        {
            using Vector = Vector<Scalar>;
            using CDSolution = CDSolution<Scalar>;

            const auto n = static_cast<Index>(matData.cols());

            CDSolution solution{ n };
            solution.x = std::move(w0);// be carefull with rvalue reference. w0 is moved!

            auto& x = solution.x;
            auto& numberOfIterations = solution.numberOfIterations;
            auto& globalChange = solution.globalChange;
            auto& dualityGap = solution.dualityGap;
            auto& status = solution.status;

            Vector R = matrixIsCovariance 
                ? static_cast<Vector>(matData * (vectData - x))
                : static_cast<Vector>(vectData - matData * x);

            const auto zJs = matrixIsCovariance ? static_cast<Vector>(matData.diagonal())
                : static_cast<Vector>(matData.colwise().squaredNorm().transpose());

            numberOfIterations = 0U;

            while (true)
            {
                globalChange = static_cast<Scalar>(0);

                // cyclical part
                for (Index j = 0; j < n; ++j)
                {//Not auto vectorized
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
            CyclicalCoordinateDescent<ModelImplementationType>::fit(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance)
        {
            const auto withIntercept = m_WithIntercept && !matrixIsCovariance;

            auto solution = fitNoIntercept(withIntercept ? (matData.array() - matData.colwise().mean().array()).matrix() : matData,
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
            CyclicalCoordinateDescent<ModelImplementationType>::fitNoIntercept(
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
                    : L2Regressor(m_Param.beta).fitNoIntercept(matData,vectData, matrixIsCovariance,
                        m_Param.innerEpsilon,
                        m_Param.innerMaximumNumberOfIterations));
            }
            else
            {
                m_FromBothConverged.clear(std::memory_order_relaxed);

                auto fromZeroFuture =
                    std::async(std::launch::async, &CyclicalCoordinateDescent::fitFrom, this,
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

        template<std::floating_point ScalarType>
        using L0L2Regressor = CyclicalCoordinateDescent<L0L2ModelImplementation<ScalarType>>;
    }
}
#endif //L0L2_COORDINATE_DESCENT_SOLVER_H

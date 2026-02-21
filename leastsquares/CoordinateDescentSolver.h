#ifndef L0L2_COORDINATE_DESCENT_SOLVER_H
#define L0L2_COORDINATE_DESCENT_SOLVER_H

#include <future>
#include <cmath>
#include <functional>
#include <execution>
#include <numeric>
#include <atomic>
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
        concept ModelLike = requires()//requires(ModelImplementationType impl)
        {
            {
                ModelImplementationType::stepJ(typename ModelImplementationType::Param{},
                    typename ModelImplementationType::Scalar{},
                    typename ModelImplementationType::Scalar{})
            } ->std::convertible_to<typename ModelImplementationType::Scalar>;

            //TODO add constraint for compute duality gap
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
            /*CyclicalCoordinateDescent(
                Scalar delta,
                Scalar beta,
                Strategy strategy = Strategy::FromZeroSolution,
                Scalar tolerance = static_cast<Scalar>(1e-4),
                unsigned int maximumNumberOfIterations = 10000U)
                :m_Param{ delta, beta, strategy, tolerance, maximumNumberOfIterations },
                m_FromBothConverged{}
            {
            }*/

            CyclicalCoordinateDescent(const Param& param)
                :m_Param{param},
                m_FromBothConverged{}
            {
            }

            /*! \brief Fit full path solutions.
               \param matData contiguous data container representing matrix in column major layout.
               \param numberOfRows matrix number of rows.
               \param numberOfColumns matrix number of columns.
               \param vectData contiguous data container representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \return a solution in the format CDSolution.
             */
            CDSolution<Scalar> fit(
                const ContiguousDataContainer<Scalar>& matData,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
                bool matrixIsCovariance);

            CDSolution<Scalar> fitt(
                const ContiguousDataContainer<Scalar>& matData,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance);

        private:
            CDSolution<Scalar>  fitFrom(
                const ContiguousDataContainer<Scalar>& matData,
                Index numberOfRows, Index numberOfColumns,
                bool matrixIsCovariance,
                const Scalar* vectDataPtr,
                ContiguousDataContainer<Scalar>&& w0);

            const Param m_Param;

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
                    unsigned int maximumNumberOfIterationsInput = 10000U)
                    :delta{ deltaInput },
                    beta{ betaInput },
                    strategy{ strategyInput },
                    tolerance{ toleranceInput },
                    maximumNumberOfIterations{ maximumNumberOfIterationsInput },
                    deltaBeta{ deltaInput * betaInput }
                    //to optimize
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

            ContiguousDataContainer<Scalar> fit(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance,
                unsigned int maxNumberOfIterations = 100000U) const;

        private:
            const Scalar m_Beta;
        };

        template<std::floating_point ScalarType>
        ContiguousDataContainer<typename L2RegressorPCG<ScalarType>::Scalar>
            L2RegressorPCG<ScalarType>::fit(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance,
                unsigned int maxNumberOfIterations) const
        {
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using Utils = Utils<Scalar>;
            using Index = Index;

            const auto n = numberOfColumns;
            const auto m = numberOfRows;

            auto x = ContiguousDataContainer(n, static_cast<Scalar>(0));
            auto xPtr = x.data();

            if (matrixIsCovariance)
            {
                //const auto& alpha = vectData;
                //const auto alphaPtr = alpha.data();
                const auto alphaPtr = vectDataPtr;

                const auto& Q = matData;
                const auto QPtr = Q.data();

                ContiguousDataContainer M(n, m_Beta);
                auto mPtr = M.data();
#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    mPtr[i] += QPtr[i * m + i];
                }

                ContiguousDataContainer r(m);
                auto rPtr = r.data();
#pragma omp parallel for
                for (Index i = 0; i < m; ++i)
                {//Not auto vectorized
                    const auto QPtrRowi = QPtr + i * m;

                    auto total = m_Beta * xPtr[i];
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        // qij == qji
                        total += QPtrRowi[j] * (alphaPtr[j] - xPtr[j]);
                    }

                    rPtr[i] = total;
                }

                auto p = r;
                auto pPtr = p.data();
#pragma omp simd
                for (Index j = 0; j < n; ++j)
                {//Vectorized
                    pPtr[j] /= mPtr[j];
                }

                auto r0dotz0 = static_cast<Scalar>(0);
#pragma omp simd
                for (Index j = 0; j < n; ++j)
                {//Vectorized
                    r0dotz0 += rPtr[j] * pPtr[j];
                }

                {
                    ContiguousDataContainer w(m);
                    auto wPtr = w.data();
#pragma omp parallel for
                    for (Index i = 0; i < m; ++i)
                    {//Not auto vectorized
                        // TODO optimize vectorization
                        const auto QPtrRowi = QPtr + i * m;

                        auto total = m_Beta * pPtr[i];
#pragma omp simd
                        for (Index j = 0; j < n; ++j)
                        {//Vectorized
                            // qij == qji
                            total += QPtrRowi[j] * pPtr[j];
                        }

                        wPtr[i] = total;
                    }

                    auto pdotw = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        pdotw += pPtr[j] * wPtr[j];
                    }

                    const Scalar alphaValue = r0dotz0 / pdotw;

#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        xPtr[j] += alphaValue * pPtr[j];
                    }

#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        rPtr[j] -= alphaValue * wPtr[j];
                    }
                }

                unsigned int iter = 0U;
                while (norm(r) > Utils::epsilon)
                {
                    auto z = r;
                    auto zPtr = z.data();
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        zPtr[j] /= mPtr[j];
                    }

                    auto rdotz = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        rdotz += rPtr[j] * zPtr[j];
                    }

                    const auto rdotzOverr0dotz0 = rdotz / r0dotz0;
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        pPtr[j] = rdotzOverr0dotz0 * pPtr[j] + zPtr[j];
                    }

                    ContiguousDataContainer w(m);
                    auto wPtr = w.data();
#pragma omp parallel for
                    for (Index i = 0; i < m; ++i)
                    {//Not auto vectorized
                        // TODO optimize vectorization
                        const auto QPtrRowi = QPtr + i * m;

                        auto total = m_Beta * pPtr[i];
#pragma omp simd
                        for (Index j = 0; j < n; ++j)
                        {//Vectorized
                            // qij == qji
                            total += QPtrRowi[j] * pPtr[j];
                        }

                        wPtr[i] = total;
                    }

                    auto pdotw = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        pdotw += pPtr[j] * wPtr[j];
                    }

                    const Scalar alphaValue = rdotz / pdotw;

                    r0dotz0 = rdotz;

#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        xPtr[j] += alphaValue * pPtr[j];
                    }

#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        rPtr[j] -= alphaValue * wPtr[j];
                    }

                    ++iter;

                    if (iter > maxNumberOfIterations)
                    {
                        //std::cout << "\nMax iter attained\n";
                        break;
                    }
                }
            }
            else
            {
                //const auto& b = vectData;
                //const auto bPtr = b.data();
                const auto bPtr = vectDataPtr;

                const auto& A = matData;
                const auto APtr = A.data();

                ContiguousDataContainer M(n);
                auto mPtr = M.data();
#pragma omp parallel for
                for (Index j = 0; j < n; ++j)
                {//Not auto vectorized
                    const auto APtrColj = APtr + j * m;
                    auto total = m_Beta;
#pragma omp simd
                    for (Index i = 0; i < m; ++i)
                    {//Vectorized
                        total += APtrColj[i] * APtrColj[i];
                    }

                    mPtr[j] = total;
                }

                ContiguousDataContainer r(n);
                auto rPtr = r.data();
                {
                    //auto bMinusAx = b;
                    ContiguousDataContainer bMinusAx(vectDataPtr, vectDataPtr + m);
                    auto bMinusAxPtr = bMinusAx.data();
#pragma omp parallel for
                    for (Index i = 0; i < m; ++i)
                    {//Not auto vectorized
                        for (Index j = 0; j < n; ++j)
                        {//Not auto vectorized
                            bMinusAxPtr[i] -= APtr[j * m + i] * xPtr[j];
                        }
                    }

#pragma omp parallel for
                    for (Index i = 0; i < n; ++i)
                    {//Not auto vectorized
                        const auto APtrColi = APtr + i * m;
                        auto total = m_Beta * xPtr[i];
#pragma omp simd
                        for (Index j = 0; j < m; ++j)
                        {//Vectorized
                            total += APtrColi[j] * bMinusAxPtr[j];
                        }

                        rPtr[i] = total;
                    }
                }

                auto p = r;
                auto pPtr = p.data();
#pragma omp simd
                for (Index i = 0; i < n; ++i)
                {//Vectorized
                    pPtr[i] /= mPtr[i];
                }

                auto r0dotz0 = static_cast<Scalar>(0);
#pragma omp simd
                for (Index j = 0; j < n; ++j)
                {//Vectorized
                    r0dotz0 += rPtr[j] * pPtr[j];
                }

                {
                    ContiguousDataContainer w(n);
                    auto wPtr = w.data();
                    {
                        auto Ap = ContiguousDataContainer(m, static_cast<Scalar>(0));
                        auto ApPtr = Ap.data();
#pragma omp parallel for
                        for (Index i = 0; i < m; ++i)
                        {//Not auto vectorized
                            for (Index j = 0; j < n; ++j)
                            {//Not auto vectorized
                                ApPtr[i] += APtr[j * m + i] * pPtr[j];
                            }
                        }
#pragma omp parallel for
                        for (Index i = 0; i < n; ++i)
                        {//Not auto vectorized
                            const auto APtrColi = APtr + i * m;
                            auto total = m_Beta * pPtr[i];
#pragma omp simd
                            for (Index j = 0; j < m; ++j)
                            {//Vectorized
                                total += APtrColi[j] * ApPtr[j];
                            }

                            wPtr[i] = total;
                        }
                    }

                    auto pdotw = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        pdotw += pPtr[j] * wPtr[j];
                    }

                    const Scalar alphaValue = r0dotz0 / pdotw;

#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        xPtr[j] += alphaValue * pPtr[j];
                    }

#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        rPtr[j] -= alphaValue * wPtr[j];
                    }
                }

                unsigned int iter = 0;
                while (norm(r) > Utils::epsilon)
                {
                    auto z = r;
                    auto zPtr = z.data();
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        zPtr[j] /= mPtr[j];
                    }

                    auto rdotz = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        rdotz += rPtr[j] * zPtr[j];
                    }

                    const auto rdotzOverr0dotz0 = rdotz / r0dotz0;
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        pPtr[j] = rdotzOverr0dotz0 * pPtr[j] + zPtr[j];
                    }

                    {
                        ContiguousDataContainer w(n);
                        auto wPtr = w.data();
                        {
                            auto Ap = ContiguousDataContainer(m, static_cast<Scalar>(0));
                            auto ApPtr = Ap.data();
#pragma omp parallel for
                            for (Index i = 0; i < m; ++i)
                            {//Not auto vectorized
                                for (Index j = 0; j < n; ++j)
                                {//Not auto vectorized
                                    ApPtr[i] += APtr[j * m + i] * pPtr[j];
                                }
                            }
#pragma omp parallel for
                            for (Index i = 0; i < n; ++i)
                            {//Not auto vectorized
                                const auto APtrColi = APtr + i * m;
                                auto total = m_Beta * pPtr[i];
#pragma omp simd
                                for (Index j = 0; j < m; ++j)
                                {//Vectorized
                                    total += APtrColi[j] * ApPtr[j];
                                }

                                wPtr[i] = total;
                            }
                        }

                        auto pdotw = static_cast<Scalar>(0);
#pragma omp simd
                        for (Index j = 0; j < n; ++j)
                        {//Vectorized
                            pdotw += pPtr[j] * wPtr[j];
                        }

                        const Scalar alphaValue = rdotz / pdotw;

                        r0dotz0 = rdotz;
#pragma omp simd
                        for (Index j = 0; j < n; ++j)
                        {//Vectorized
                            xPtr[j] += alphaValue * pPtr[j];
                        }
#pragma omp simd
                        for (Index j = 0; j < n; ++j)
                        {//Vectorized
                            rPtr[j] -= alphaValue * wPtr[j];
                        }
                    }

                    ++iter;

                    if (iter > maxNumberOfIterations)
                    {
                        //std::cout << "\nMax iter attained\n";
                        break;
                    }
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
                const ContiguousDataContainer<Scalar>& matData,
                Index numberOfRows, Index numberOfColumns,
                bool matrixIsCovariance,
                const Scalar* vectDataPtr,
                ContiguousDataContainer<Scalar>&& w0)
        {
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using CDSolution = CDSolution<Scalar>;

            const auto m = numberOfRows;
            const auto n = numberOfColumns;

            const auto matDataPtr = matData.data();
            //const auto vectDataPtr = vectData.data();

            CDSolution solution{ n };
            solution.x = std::move(w0);// be carefull with rvalue reference. w0 is moved!

            auto weightsPtr = solution.x.data();
            auto& numberOfIterations = solution.numberOfIterations;
            auto& globalChange = solution.globalChange;
            auto& dualityGap = solution.dualityGap;
            auto& status = solution.status;

            //ContiguousDataContainer R = matrixIsCovariance ? (M * (warmStart ? (v - m_Weights) : v))
            //    : (warmStart ? v - M * m_Weights : v); 

            //ContiguousDataContainer R = matrixIsCovariance ? (M * (v - m_Weights))
            //    : (v - M * m_Weights); 
            // 
            // TODO optimize m_Weights == 0 case
            ContiguousDataContainer R(m);
            if (matrixIsCovariance)
            {
#pragma omp parallel for
                for (Index i = 0; i < m; ++i)
                {//Not auto vectorized
                    const auto matDataPtrRowi = matDataPtr + i * m;
                    auto total = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        //M.coeff(i, j)==M.coeff(j,i) ==> matDataPtr[j*m + i] == matDataPtr[i*m + j]
                        //R[i] += matDataPtr[i*m + j] * (v[j] - weightsPtr[j]);
                        total += matDataPtrRowi[j] * (vectDataPtr[j] - weightsPtr[j]);
                    }

                    R[i] = total;
                }
            }
            else
            {
#pragma omp parallel for
                for (Index i = 0; i < m; ++i)
                {//Not auto vectorized
                    //const auto matDataPtrRowi = matDataPtr + i*m;
                    auto total = vectDataPtr[i];
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        total -= matDataPtr[j * m + i] * weightsPtr[j];
                    }

                    R[i] = total;
                }
            }

            ContiguousDataContainer zJs(n);// , static_cast<Scalar>(0));
            auto zJsPtr = zJs.data();

            if (matrixIsCovariance)
            {
#pragma omp parallel for
                for (Index j = 0; j < n; ++j)
                {//Not auto vectorized
                    zJsPtr[j] = matDataPtr[j * m + j];
                }
            }
            else
            {
#pragma omp parallel for
                for (Index j = 0; j < n; ++j)
                {//Not auto vectorized
                    const auto matDataPtrColj = matDataPtr + j * m;
                    auto total = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index i = 0; i < m; ++i)
                    {//Vectorized
                        //zJs[j] += matDataPtr[j*m + i] * matDataPtr[j*m + i];
                        //zJsPtr[j] += matDataPtrColj[i] * matDataPtrColj[i];
                        total += matDataPtrColj[i] * matDataPtrColj[i];
                    }

                    zJsPtr[j] = total;
                }
            }

            numberOfIterations = 0;

            while (true)
            {
                globalChange = static_cast<Scalar>(0);

                // cyclical part
                for (Index j = 0; j < n; ++j)
                {//Not auto vectorized
                    const auto zJ = zJsPtr[j];

                    if (zJ == static_cast<Scalar>(0))
                    {
                        continue;
                    }

                    const auto oldWeightJ = weightsPtr[j];

                    //const auto uJ = matrixIsCovariance ? R[j] + zJ * oldWeightJ : M.col(j).dot(R) + zJ * oldWeightJ;
                    auto uJ = zJ * oldWeightJ;
                    if (matrixIsCovariance)
                    {
                        uJ += R[j];
                    }
                    else
                    {
                        const auto matDataPtrColj = matDataPtr + j * m;
#pragma omp simd
                        for (Index i = 0; i < m; ++i)
                        {//Vectorized
                            //uJ += matDataPtr[j*m + i] * R[i];
                            uJ += matDataPtrColj[i] * R[i];
                        }
                        //uJ += M.col(j).dot(R);
                    }

                    const auto newWeightJ = ModelImplementation::stepJ(m_Param, zJ, uJ);

                    weightsPtr[j] = newWeightJ;

                    const auto weightsDiff = oldWeightJ - newWeightJ;

                    if (weightsDiff != static_cast<Scalar>(0))
                    {
                        //R += weightsDiff * M.col(j);
                        const auto matDataPtrColj = matDataPtr + j * m;
#pragma omp simd
                        for (Index i = 0; i < m; ++i)
                        {//Vectorized
                            R[i] += weightsDiff * matDataPtrColj[i];
                        }

                        const auto weightJChange = std::abs(weightsDiff);

                        if (weightJChange > globalChange)
                        {
                            globalChange = weightJChange;
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
        inline CDSolution<typename ModelImplementationType::Scalar>
            CyclicalCoordinateDescent<ModelImplementationType>::fit(
                const ContiguousDataContainer<Scalar>& matData,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
                bool matrixIsCovariance)
        {
            return fitt(matData, numberOfRows, numberOfColumns, vectData.data(), matrixIsCovariance);
        }

        template <ModelLike ModelImplementationType>
        CDSolution<typename ModelImplementationType::Scalar>
            CyclicalCoordinateDescent<ModelImplementationType>::fitt(
                const ContiguousDataContainer<Scalar>& matData,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance)
        {
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using L2Regressor = L2RegressorPCG<Scalar>;

            if (m_Param.strategy != Strategy::FromBothSolutions)
            {
                return fitFrom(matData, numberOfRows, numberOfColumns, matrixIsCovariance, vectDataPtr,
                    m_Param.strategy == Strategy::FromZeroSolution
                    ? ContiguousDataContainer(numberOfColumns, static_cast<Scalar>(0))
                    : L2Regressor(m_Param.beta).fit(matData, numberOfRows, numberOfColumns, vectDataPtr, matrixIsCovariance));
            }
            else
            {
                m_FromBothConverged.clear(std::memory_order_relaxed);

                auto fromZeroFuture =
                    std::async(std::launch::async, &CyclicalCoordinateDescent::fitFrom, this,
                        matData, numberOfRows, numberOfColumns,
                        matrixIsCovariance,
                        vectDataPtr,
                        ContiguousDataContainer(numberOfColumns, static_cast<Scalar>(0)));

                auto fromL2Solution = fitFrom(matData, numberOfRows, numberOfColumns,
                    matrixIsCovariance,
                    vectDataPtr,
                    L2Regressor(m_Param.beta).fit(matData, numberOfRows, numberOfColumns, vectDataPtr, matrixIsCovariance));

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

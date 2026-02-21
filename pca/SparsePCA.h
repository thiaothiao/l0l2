#ifndef L0L2_SPARSE_PCA_H
#define L0L2_SPARSE_PCA_H

#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <future>
#include <limits>
#include <concepts>

#include <Eigen/Dense>

#include "thread_pool/thread_pool.h"
//#include <unsupported/Eigen/CXX11/ThreadPool>

#include "CoordinateDescentSolver.h"
#include "FullPathSolver.h"
#include "Utils.h"

namespace l0l2
{
    namespace linearmodel
    {
        template <class ModelImplementationType>
        concept SPCAModelLike = requires()//requires(const ModelImplementationType & impl)
        {
            {
                ModelImplementationType::fitRegressor(
                    Matrix<typename ModelImplementationType::Scalar>{},
                    Vector<typename ModelImplementationType::Scalar>{},
                    bool{},
                    typename ModelImplementationType::Param{})
            } ->std::convertible_to<Vector<typename ModelImplementationType::Scalar>>;
        };

        // General implementation
        template <SPCAModelLike ModelImplementationType>
        class SPCA final
        {
        public:
            using ModelImplementation = ModelImplementationType;
            using Param = typename ModelImplementation::Param;
            using Scalar = typename ModelImplementation::Scalar;
            using Regressor = typename ModelImplementation::Regressor;
            using RegressorParam = typename ModelImplementation::Param;

            SPCA(const Param& param, unsigned int nbJobs = 1U, 
                Scalar epsilon = static_cast<Scalar>(1e-5),
                unsigned int numberOfTrialsMax = 10000U)
                :m_Param{ param },
                m_NbJobs{ nbJobs },
                m_Epsilon{ epsilon },
                m_NumberOfTrialsMax{ numberOfTrialsMax }
            {
            }

            Matrix<Scalar> run(
                const Matrix<Scalar>& matData,
                bool matrixIsCovariance);

        private:
            const Param m_Param;
            const unsigned int m_NbJobs;
            const Scalar m_Epsilon;
            const unsigned int m_NumberOfTrialsMax;
        };

        template<std::floating_point ScalarType>
        class L0L2SPCAModelImplementation final
        {
        public:
            using Scalar = ScalarType;
            using Regressor = L0L2Regressor<Scalar>;
            using RegressorParam = Regressor::Param;

            struct Param final
            {
                Param(
                    Scalar deltaInput = static_cast<Scalar>(0),
                    Scalar betaInput = static_cast<Scalar>(1),
                    Index nbComponentsInput = static_cast<Index>(2),
                    Strategy strategyInput = Strategy::FromZeroSolution,
                    Scalar toleranceInput = static_cast<Scalar>(1e-4),
                    unsigned int maximumNumberOfIterationsInput = 10000U)
                    :regressorParam{ deltaInput, betaInput , strategyInput, toleranceInput, maximumNumberOfIterationsInput },
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
            using Regressor = FullPathSolver<Scalar>;
            using RegressorParam = Regressor::Param;

            struct Param final
            {
                Param(
                    Scalar deltaInput = static_cast<Scalar>(0),
                    Scalar betaInput = static_cast<Scalar>(1),
                    Index nbComponentsInput = static_cast<Index>(2),
                    Strategy strategyInput = Strategy::FromZeroSolution)
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

        template <SPCAModelLike ModelImplementationType>
        Matrix<typename SPCA<ModelImplementationType>::Scalar>
            SPCA<ModelImplementationType>::run(
                const Matrix<Scalar>& matData,
                bool matrixIsCovariance)
        {
            using Vector = Vector<Scalar>;
            using Matrix = Matrix<Scalar>;

            const auto n = static_cast<Index>(matData.cols());
            const auto m = static_cast<Index>(matData.rows());

            Matrix alpha;

            if (matrixIsCovariance)
            {
                Matrix alphaTmp(n, m_Param.nbComponents);

                Eigen::SelfAdjointEigenSolver<Matrix> es(matData);

                const auto& eigenValues = es.eigenvalues();
                const auto& eigenVectors = es.eigenvectors();

                Index j = 0;
                Scalar previousEigenValue = std::numeric_limits<Scalar>::max();
                for (Index idx = eigenValues.size() - 1; idx >= 0; --idx)
                {
                    const auto currentEigenvalue = eigenValues[idx];
                    if (currentEigenvalue < previousEigenValue - Utils<Scalar>::epsilon)
                    {
                        if (j >= m_Param.nbComponents)
                        {
                            break;
                        }

                        alphaTmp.col(j) = eigenVectors.col(idx);
                        ++j;
                    }
                }

                alpha = std::move(alphaTmp);
            }
            else
            {
                Eigen::JacobiSVD<Matrix, Eigen::ComputeThinV> svd(matData);

                alpha = svd.matrixV()(Eigen::seqN(0, n), Eigen::seqN(0, m_Param.nbComponents));
            }

            const auto multipleJobs = m_NbJobs > 1U;

            // Create a thread pool
            //Eigen::ThreadPool threadPool(4); // Example: 4 threads

            dp::thread_pool pool(multipleJobs ? m_NbJobs : 0);
            std::vector<std::future<Vector>> futures;

            if (multipleJobs)
            {
                futures.reserve(alpha.cols());
            }

            Matrix result = alpha;
            const auto  resultNbRows = static_cast<Index>(result.rows());
            const auto  resultNbColumns = static_cast<Index>(result.cols());

            unsigned int trial = 0;

            while (true)
            {
                if (multipleJobs)
                {
                    for (Index j = 0; j < alpha.cols(); ++j)
                    {
                        futures.push_back(pool.enqueue(ModelImplementation::fitRegressor,
                            matData,
                            matrixIsCovariance
                                ? static_cast<Vector>(alpha.col(j))
                                : static_cast<Vector>(matData * alpha.col(j)),
                            matrixIsCovariance, m_Param));
                    }
                }

                Scalar changes = static_cast<Scalar>(0);
                for (Index j = 0; j < result.cols(); ++j)
                {// TODO optimize using move and swaps
                    const Vector oldWeights = result.col(j);

                    //const auto resultj 
                    result.col(j) = multipleJobs ? futures[j].get()
                        : ModelImplementation::fitRegressor(matData,
                            matrixIsCovariance
                                ? static_cast<Vector>(alpha.col(j))
                                : static_cast<Vector>(matData * alpha.col(j)),
                            matrixIsCovariance, m_Param);

                    changes = std::max(changes, (oldWeights - result.col(j)).norm());
                }

                if (multipleJobs)
                {
                    pool.clear_tasks();
                    futures.clear();
                }

                Eigen::JacobiSVD<Matrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd
                (matrixIsCovariance ? static_cast<Matrix>(matData * result)
                    : static_cast<Matrix>(matData.transpose() * (matData * result)));

                for (Index j = 0; j < m_Param.nbComponents; ++j)
                {
                    alpha.col(j) = svd.matrixU() * svd.matrixV().transpose().col(j);
                }

                ++trial;

                if (changes < m_Epsilon)
                {
                    break;
                }

                if (trial > m_NumberOfTrialsMax)
                {
                    break;
                }
            }

            result.colwise().normalize();

            return result;
        }

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

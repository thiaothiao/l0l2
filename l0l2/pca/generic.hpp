#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <future>
#include <limits>
#include <concepts>

#include <Eigen/Dense>

#include <thread_pool/thread_pool.h>

#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
    namespace linearmodel
    {
        template <class ModelImplementationType>
        concept SPCAModelLike = requires()
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

            //Eigen::JacobiSVD<Matrix, Eigen::ComputeThinV> svd(matData);

            Matrix alpha = Eigen::JacobiSVD<Matrix, Eigen::ComputeThinV>(matData)
                .matrixV()(Eigen::seqN(0, n), Eigen::seqN(0, m_Param.nbComponents));

            const auto multipleJobs = m_NbJobs > 1U;

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
                        if (matrixIsCovariance)
                        {
                            futures.push_back(pool.enqueue(ModelImplementation::fitRegressor,
                                matData, alpha.col(j), matrixIsCovariance, m_Param));
                        }
                        else 
                        {
                            futures.push_back(pool.enqueue(ModelImplementation::fitRegressor,
                                matData, matData * alpha.col(j), matrixIsCovariance, m_Param));
                        }
                    }
                }

                Scalar changes = static_cast<Scalar>(0);
                for (Index j = 0; j < result.cols(); ++j)
                {// TODO optimize using move and swaps
                    const Vector oldWeights = result.col(j);

                    if (matrixIsCovariance)
                    {
                        result.col(j) = multipleJobs ? futures[j].get()
                            : ModelImplementation::fitRegressor(matData, alpha.col(j),
                                matrixIsCovariance, m_Param);
                    }
                    else
                    {
                        result.col(j) = multipleJobs ? futures[j].get()
                            : ModelImplementation::fitRegressor(matData, matData * alpha.col(j),
                                matrixIsCovariance, m_Param);
                    }

                    changes = std::max(changes, (oldWeights - result.col(j)).norm());
                }

                if (multipleJobs)
                {
                    pool.clear_tasks();
                    futures.clear();
                }

                Eigen::JacobiSVD<Matrix, Eigen::ComputeThinU | Eigen::ComputeThinV> svd;

                if (matrixIsCovariance)
                {
                    svd.compute(matData * result);
                }
                else
                {
                    svd.compute(matData.transpose() * (matData * result));
                }

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
    }
} // namespace l0l2

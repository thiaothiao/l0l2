#ifndef L0L2_FULL_PATH_SOLVER_H
#define L0L2_FULL_PATH_SOLVER_H

#include <list>
#include <limits>
#include <mutex>
#include <iostream>
#include <ios>
#include <iomanip>
#include <future>
#include <utility>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <concepts>

#include "Utils.h"
#include "L2Regressors.h"
#include "FullPathStep.h"

namespace l0l2
{
    namespace linearmodel
    {
        /* \brief Full path solver class.
        *
        * It produces solutions from partial path or full path computations.
        */
        template<std::floating_point ScalarType>
        class FullPathSolver final
        {
        public:
            using Scalar = ScalarType; /*!< Alias for the used scalar type */

            struct Param final
            {
                Param(
                    Scalar deltaInput = static_cast<Scalar>(0),
                    Scalar betaInput = static_cast<Scalar>(1),
                    Strategy strategyInput = Strategy::FromZeroSolution)
                    :delta{ deltaInput },
                    beta{ betaInput },
                    strategy{ strategyInput }
                {
                }

                const Scalar delta;
                const Scalar beta;
                const Strategy strategy;
            };

            /*! \brief A solver object constructor.
              \param delta sparsity regularization parameter.
              \param beta l2 regularization parameter.
              \param strategy an enum indicating a strategy: from zero, or l2 or both solutions.
            */
            FullPathSolver(const Param& param,
                bool withIntercept = false) :
                m_Param{ param },
                m_WithIntercept{ withIntercept },
                m_DeltaFromZeroSolution{ std::numeric_limits<Scalar>::max() },
                m_DeltaFromL2Solution{ static_cast<Scalar>(0) }
            {
            }

            /*! \brief Fit full path solutions.
               \param matData contiguous data container representing matrix in column major layout.
               \param vectData contiguous data container representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \param beta l2 regularization parameter.
               \param strategy an enum indicating a strategy: from zero, or l2 or both solutions.
               \return a list of solutions generating entire piecewise linear path solutions
             */
            static std::list<Solution<Scalar>>
                fitAll(const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar beta,
                    bool withIntercept = false,
                    Strategy strategy = Strategy::FromZeroSolution);

            /*! \brief Fit one solution.
               \param matData contiguous data container representing matrix in column major layout..
               \param vectData contiguous data container representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \return a solution corresponding to the regularization parameters.
             */
            Solution<Scalar>  fit(const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance);

        private:
            static std::list<Solution<Scalar>>
                fitAllNoIntercept(const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar beta,
                    Strategy strategy = Strategy::FromZeroSolution);

            Solution<Scalar>  fitNoIntercept(const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance);

            std::list<Solution<Scalar>> solve(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance, bool fromZeroSolution);

            const Param m_Param;

            const bool m_WithIntercept;

            volatile Scalar m_DeltaFromZeroSolution;
            volatile Scalar m_DeltaFromL2Solution;

            std::mutex m_DeltasMutex;
        };


        template<std::floating_point ScalarType>
        std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
            FullPathSolver<ScalarType>::fitAll(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar beta,
                bool withIntercept,
                Strategy strategy)
        {// TODO optimize intercept case
            const auto consideringIntercept = withIntercept && !matrixIsCovariance;

            auto results = fitAllNoIntercept(consideringIntercept ? (matData.rowwise() - matData.colwise().mean()).eval()
                : matData,
                consideringIntercept ? (vectData.array() - vectData.mean()).matrix() : vectData,
                matrixIsCovariance,
                beta,
                strategy);

            if (consideringIntercept)
            {
                for (auto& result : results)
                {
                    result.intercept = (vectData - matData * result.x).mean();// TODO use grad!
                }
            }

            return results;
        }

        template<std::floating_point ScalarType>
        std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
            FullPathSolver<ScalarType>::fitAllNoIntercept(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar beta,
                Strategy strategy)
        {
            using Solution = Solution<Scalar>;
            using Vector = Vector<Scalar>;
            using Utils = Utils<Scalar>;

            FullPathSolver regressor{ Param{static_cast<Scalar>(-1), beta, strategy}, false};

            if (strategy != Strategy::FromBothSolutions)
            {
                return regressor.solve(
                    matData,
                    vectData,
                    matrixIsCovariance,
                    strategy == Strategy::FromZeroSolution);
            }
            else
            {
                auto fromZeroFuture =
                    std::async(std::launch::async, &FullPathSolver::solve, &regressor,
                        matData,
                        vectData,
                        matrixIsCovariance,
                        true);

                auto fromL2SolutionResults = regressor.solve(
                    matData,
                    vectData,
                    matrixIsCovariance,
                    false);

                auto fromZeroSolutionResults = fromZeroFuture.get();

                for (auto it = fromL2SolutionResults.begin(); it != fromL2SolutionResults.end(); ++it)
                {
                    if (it->delta < fromZeroSolutionResults.back().delta - Utils::epsilon)
                    {
                        fromZeroSolutionResults.push_back(std::move(*it));
                    }
                }

                return fromZeroSolutionResults;// move
            }
        }


        template<std::floating_point ScalarType>
        Solution<typename FullPathSolver<ScalarType>::Scalar>
            FullPathSolver<ScalarType>::fit(
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
                solution.intercept = (vectData - matData * solution.x).mean();// TODO use grad!
            }

            return solution;
        }

        template<std::floating_point ScalarType>
        Solution<typename FullPathSolver<ScalarType>::Scalar>
            FullPathSolver<ScalarType>::fitNoIntercept(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance)
        {
            using Solution = Solution<Scalar>;
            using Utils = Utils<Scalar>;

            if (m_Param.delta < static_cast<Scalar>(0))
            {
                // Should throw an exception!!!
                std::cout << "\nNegative deltas not allowed: delta == " << m_Param.delta << "\n";
                return Solution{};
            }

            Solution minBoundSolution{};
            Solution maxBoundSolution{};

            if (m_Param.strategy != Strategy::FromBothSolutions)
            {
                auto results = solve(matData, vectData, matrixIsCovariance,
                    m_Param.strategy == Strategy::FromZeroSolution);
                if (!results.empty())
                {
                    auto minBoundIt = results.begin();

                    minBoundSolution = std::move(*minBoundIt);

                    maxBoundSolution = std::move(*++minBoundIt);
                }
            }
            else
            {
                auto fromZeroFuture =
                    std::async(std::launch::async, &FullPathSolver::solve, this,
                        matData,
                        vectData,
                        matrixIsCovariance,
                        true);

                auto fromL2SolutionResults = solve(
                    matData,
                    vectData,
                    matrixIsCovariance,
                    false);

                auto fromZeroSolutionResults = fromZeroFuture.get();

                if (!fromZeroSolutionResults.empty() || !fromL2SolutionResults.empty())
                {
                    auto minBoundIt = fromZeroSolutionResults.empty()
                        ? fromL2SolutionResults.begin()
                        : fromZeroSolutionResults.begin();

                    minBoundSolution = std::move(*minBoundIt);

                    maxBoundSolution = std::move(*++minBoundIt);
                }
            }

            if (minBoundSolution.x.size()!= 0 && maxBoundSolution.x.size() != 0)
            {
                // interval found. minBoundSolution.delta <= delta <= maxBoundSolution.delta
                auto& deltaMin = minBoundSolution.delta;
                const auto deltaMax = maxBoundSolution.delta;

                const auto& maxBoundSolutionx = maxBoundSolution.x;
                const auto& maxBoundSolutiongrad = maxBoundSolution.grad;

                auto& minBoundSolutionx = minBoundSolution.x;
                auto& minBoundSolutiongrad = minBoundSolution.grad;

                const auto diff = deltaMax - deltaMin;
                const auto alpha = (diff > Utils::epsilon)
                    ? (m_Param.delta - deltaMin) / diff : static_cast<Scalar>(0);

                const auto oneMinusAlpha = static_cast<Scalar>(1) - alpha;

                const auto n = static_cast<Index>(minBoundSolution.x.size());

                deltaMin = oneMinusAlpha * deltaMin + alpha * deltaMax;

                minBoundSolutionx = oneMinusAlpha * minBoundSolutionx + alpha * maxBoundSolutionx;

                minBoundSolutiongrad = oneMinusAlpha * minBoundSolutiongrad + alpha * maxBoundSolutiongrad;

                return minBoundSolution;
            }

            // Should throw an exception!!!
            std::cout << "\nBad situation: delta == " << m_Param.delta << "\n";
            return Solution{};
        }

        template<std::floating_point ScalarType>
        std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
            FullPathSolver<ScalarType>::solve(const Matrix<Scalar>& matData/*colmajor*/,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance, bool fromZeroSolution)
        {
            using Vector = Vector<Scalar>;
            using Solution = Solution<Scalar>;
            using Utils = Utils<Scalar>;
            using L2Regressor = L2Regressor<Scalar>;
            using FullPathStep = FullPathStep<Scalar>;

            const auto n = static_cast<Index>(matData.cols());

            const auto tau = fromZeroSolution ? static_cast<Scalar>(1) : static_cast<Scalar>(-1);

            std::list<Solution> results;

            const Vector ATb = matData.transpose() * vectData;// same as Q\alpha

            if (fromZeroSolution)
            {
                const auto deltaZero = ATb.cwiseAbs().maxCoeff() / m_Param.beta;

                results.emplace_back(std::numeric_limits<Scalar>::max(),
                    Vector::Zero(n), -ATb);

                results.emplace_back(deltaZero,
                    results.back().x,
                    results.back().grad);

                bool leave = false;
                bool iamOnTarget = false;
                bool otherOnTarget = false;

                if (m_Param.strategy == Strategy::FromBothSolutions)
                {
                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                    m_DeltaFromZeroSolution = deltaZero;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_DeltaFromZeroSolution <= m_Param.delta)
                        {
                            iamOnTarget = true;
                        }
                        else if (m_Param.delta <= m_DeltaFromL2Solution)
                        {
                            otherOnTarget = true;
                        }
                    }

                    if (!iamOnTarget && !otherOnTarget
                        && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                    {
                        leave = true;
                    }
                }
                else
                {
                    m_DeltaFromZeroSolution = deltaZero;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_DeltaFromZeroSolution <= m_Param.delta)
                        {
                            iamOnTarget = true;
                        }
                    }

                    if (!iamOnTarget && m_DeltaFromZeroSolution <= Utils::epsilon)
                    {
                        leave = true;
                    }
                }

                if (iamOnTarget)
                {
                    auto minBoundIt = results.crbegin();

                    const auto& minBoundSolution = *minBoundIt;

                    const auto& maxBoundSolution = *++minBoundIt;


                    return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                }
                else if (otherOnTarget)
                {
                    return std::list<Solution>{};
                }
                else if (leave)
                {
                    return results;
                }
            }
            else
            {
                Solution solutionBar{ n };

                solutionBar.x = L2Regressor(m_Param.beta).fitNoIntercept(matData, vectData, matrixIsCovariance);

                solutionBar.delta = std::numeric_limits<Scalar>::max();
                for (Index i = 0; i < n; ++i)
                {
                    // TODO optimize
                    const auto absWeight = std::abs(solutionBar.x[i]);
                    if (Utils::epsilon < absWeight && absWeight < solutionBar.delta)
                    {
                        solutionBar.delta = absWeight;
                    }
                }

                results.emplace_front(static_cast<Scalar>(0),
                    solutionBar.x,
                    solutionBar.grad);

                results.push_front(std::move(solutionBar));

                bool leave = false;
                bool iamOnTarget = false;
                bool otherOnTarget = false;
                if (m_Param.strategy == Strategy::FromBothSolutions)
                {
                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                    m_DeltaFromL2Solution = solutionBar.delta;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_Param.delta <= m_DeltaFromL2Solution)
                        {
                            iamOnTarget = true;
                        }
                        else if (m_DeltaFromZeroSolution <= m_Param.delta)
                        {
                            otherOnTarget = true;
                        }
                    }

                    if (!iamOnTarget && !otherOnTarget
                        && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                    {
                        leave = true;
                    }
                }
                else
                {
                    m_DeltaFromL2Solution = solutionBar.delta;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_Param.delta <= m_DeltaFromL2Solution)
                        {
                            iamOnTarget = true;
                        }
                    }

                    if (!iamOnTarget &&
                        std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution)// TODO use delta of zero
                    {
                        leave = true;
                    }
                }

                if (iamOnTarget)
                {
                    auto maxBoundIt = results.cbegin();

                    const auto& maxBoundSolution = *maxBoundIt;

                    const auto& minBoundSolution = *++maxBoundIt;

                    return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                }
                else if (otherOnTarget)
                {
                    return std::list<Solution>{};
                }
                else if (leave)
                {
                    return results;
                }
            }

            unsigned int numberIters = 0;

            while (true)
            {
                ++numberIters;

                {
                    const auto& solutionMaam = fromZeroSolution ? *(++results.crbegin()) : *(++results.cbegin());

                    const auto& solutionYaay = fromZeroSolution ? results.back() : results.front();

                    const auto step = FullPathStep(m_Param.beta);

                    auto solutionNew = step.run(matData,
                        ATb,
                        solutionYaay,
                        solutionMaam,
                        matrixIsCovariance,
                        tau);

                    const auto gammaNew = tau * (solutionYaay.delta - solutionNew.delta);

                    if (gammaNew <= Utils::epsilon)
                    {
                        std::cout << "\nSTART CYCLING!\n";

                        std::cout << std::fixed << std::setprecision(15);
                        std::cout << "\nGamma: " << gammaNew
                            << " is less than " << Utils::epsilon << "\n";

                        break;
                    }

                    bool leave = false;
                    bool iamOnTarget = false;
                    bool otherOnTarget = false;
                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        if (fromZeroSolution)
                        {
                            m_DeltaFromZeroSolution = solutionNew.delta;
                        }
                        else
                        {
                            m_DeltaFromL2Solution = solutionNew.delta;
                        }

                        if (m_Param.delta > static_cast<Scalar>(0))
                        {
                            if (fromZeroSolution)
                            {
                                if (m_DeltaFromZeroSolution <= m_Param.delta)
                                {
                                    iamOnTarget = true;
                                }
                                else if (m_Param.delta <= m_DeltaFromL2Solution)
                                {
                                    otherOnTarget = true;
                                }
                            }
                            else
                            {
                                if (m_Param.delta <= m_DeltaFromL2Solution)
                                {
                                    iamOnTarget = true;
                                }
                                else if (m_DeltaFromZeroSolution <= m_Param.delta)
                                {
                                    otherOnTarget = true;
                                }
                            }
                        }

                        if (!iamOnTarget && !otherOnTarget
                            && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                        {
                            leave = true;
                        }
                    }
                    else
                    {
                        if (fromZeroSolution)
                        {
                            m_DeltaFromZeroSolution = solutionNew.delta;
                        }
                        else
                        {
                            m_DeltaFromL2Solution = solutionNew.delta;
                        }

                        if (m_Param.delta > static_cast<Scalar>(0))
                        {
                            if (fromZeroSolution)
                            {
                                if (m_DeltaFromZeroSolution <= m_Param.delta)
                                {
                                    iamOnTarget = true;
                                }
                            }
                            else
                            {
                                if (m_Param.delta <= m_DeltaFromL2Solution)
                                {
                                    iamOnTarget = true;
                                }
                            }
                        }

                        if (fromZeroSolution)
                        {
                            if (!iamOnTarget && m_DeltaFromZeroSolution <= Utils::epsilon)
                            {
                                leave = true;
                            }
                        }
                        else
                        {
                            if (!iamOnTarget
                                && std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution)
                            {
                                leave = true;
                            }
                        }
                    }

                    if (fromZeroSolution)
                    {
                        results.push_back(std::move(solutionNew));

                        if (iamOnTarget)
                        {
                            auto minBoundIt = results.crbegin();

                            const auto& minBoundSolution = *minBoundIt;

                            const auto& maxBoundSolution = *++minBoundIt;


                            return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                        }
                        else if (otherOnTarget)
                        {
                            return std::list<Solution>{};
                        }
                    }
                    else
                    {
                        results.push_front(std::move(solutionNew));

                        if (iamOnTarget)
                        {
                            auto maxBoundIt = results.cbegin();

                            const auto& maxBoundSolution = *maxBoundIt;

                            const auto& minBoundSolution = *++maxBoundIt;

                            return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                        }
                        else if (otherOnTarget)
                        {
                            return std::list<Solution>{};
                        }
                    }

                    if (leave)
                    {
                        return results;
                    }
                }

                const auto& solutionLast = fromZeroSolution ? results.back() : results.front();

                {
                    const auto fullPathDone = (!fromZeroSolution) && solutionLast.x.norm() <= Utils::epsilon
                        || fromZeroSolution && solutionLast.x.cwiseAbs().minCoeff() >= solutionLast.delta - Utils::epsilon;

                    if (fullPathDone)
                    {
                        return results;
                    }
                }

                {
                    auto normMax = std::numeric_limits<Scalar>::max();
                    for (Index i = 0; i < n; ++i)
                    {
                        // TODO optimize
                        const auto absXi = std::abs(solutionLast.x[i]);
                        if (Utils::epsilon < absXi && absXi < normMax)
                        {
                            normMax = absXi;
                        }
                    }

                    if (normMax < std::numeric_limits<Scalar>::max())
                    {
                        if (fromZeroSolution && normMax >= solutionLast.delta - Utils::epsilon)
                        {// constant for delta in [deltaCandidate, delta]. improve delta

                            auto absGradMax = std::numeric_limits<Scalar>::min();
                            for (Index i = 0; i < n; ++i)
                            {
                                // TODO optimize
                                const auto absXi = std::abs(solutionLast.x[i]);
                                if (absXi <= Utils::epsilon)
                                {
                                    const auto absGradi = std::abs(solutionLast.grad[i]);
                                    if (absGradi > absGradMax)
                                    {
                                        absGradMax = absGradi;
                                    }
                                }
                            }

                            if (absGradMax > std::numeric_limits<Scalar>::min()
                                && absGradMax / m_Param.beta < solutionLast.delta - Utils::epsilon)
                            {
                                const auto deltaCandidate = absGradMax / m_Param.beta;

                                bool leave = false;
                                bool iamOnTarget = false;
                                bool otherOnTarget = false;
                                if (m_Param.strategy == Strategy::FromBothSolutions)
                                {
                                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                                    m_DeltaFromZeroSolution = deltaCandidate;

                                    if (m_Param.delta > static_cast<Scalar>(0))
                                    {
                                        if (m_DeltaFromZeroSolution <= m_Param.delta)
                                        {
                                            iamOnTarget = true;
                                        }
                                        else if (m_Param.delta <= m_DeltaFromL2Solution)
                                        {
                                            otherOnTarget = true;
                                        }
                                    }

                                    if (!iamOnTarget && !otherOnTarget
                                        && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                                    {
                                        leave = true;
                                    }
                                }
                                else
                                {
                                    m_DeltaFromZeroSolution = deltaCandidate;

                                    if (m_Param.delta > static_cast<Scalar>(0))
                                    {
                                        if (m_DeltaFromZeroSolution <= m_Param.delta)
                                        {
                                            iamOnTarget = true;
                                        }
                                    }

                                    if (!iamOnTarget
                                        && m_DeltaFromZeroSolution <= Utils::epsilon)
                                    {
                                        leave = true;
                                    }
                                }

                                results.emplace_back(deltaCandidate, solutionLast.x, solutionLast.grad);

                                if (iamOnTarget)
                                {
                                    auto minBoundIt = results.crbegin();

                                    const auto& minBoundSolution = *minBoundIt;

                                    const auto& maxBoundSolution = *++minBoundIt;


                                    return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                                }
                                else if (otherOnTarget)
                                {
                                    return std::list<Solution>{};
                                }
                                else if (leave)
                                {
                                    return results;
                                }

                                // TODO try to optimize using minmax and avoid abs
                                const auto fullPathDone = results.back().x.cwiseAbs().minCoeff()
                                    >= results.back().delta - Utils::epsilon;

                                if (fullPathDone)
                                {
                                    return results;
                                }
                            }
                        }
                        else if (!fromZeroSolution && normMax > solutionLast.delta + Utils::epsilon)
                        {// constant for delta in [delta, normMax]. improve delta

                            const auto deltaCandidate = normMax;

                            bool leave = false;
                            bool iamOnTarget = false;
                            bool otherOnTarget = false;

                            if (m_Param.strategy == Strategy::FromBothSolutions)
                            {
                                const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                                m_DeltaFromL2Solution = deltaCandidate;

                                if (m_Param.delta > static_cast<Scalar>(0))
                                {
                                    if (m_Param.delta <= m_DeltaFromL2Solution)
                                    {
                                        iamOnTarget = true;
                                    }
                                    else if (m_DeltaFromZeroSolution <= m_Param.delta)
                                    {
                                        otherOnTarget = true;
                                    }
                                }

                                if (!iamOnTarget && !otherOnTarget
                                    && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                                {
                                    leave = true;
                                }
                            }
                            else
                            {
                                m_DeltaFromL2Solution = deltaCandidate;

                                if (m_Param.delta > static_cast<Scalar>(0))
                                {
                                    if (m_Param.delta <= m_DeltaFromL2Solution)
                                    {
                                        iamOnTarget = true;
                                    }
                                }

                                if (!iamOnTarget
                                    && std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution)
                                {
                                    leave = true;
                                }
                            }

                            results.emplace_front(deltaCandidate, solutionLast.x, solutionLast.grad);

                            if (iamOnTarget)
                            {
                                auto maxBoundIt = results.cbegin();

                                const auto& maxBoundSolution = *maxBoundIt;

                                const auto& minBoundSolution = *++maxBoundIt;

                                return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                            }
                            else if (otherOnTarget)
                            {
                                return std::list<Solution>{};
                            }
                            else if (leave)
                            {
                                return results;
                            }

                            const auto fullPathDone = results.front().x.norm() <= Utils::epsilon;

                            if (fullPathDone)
                            {
                                return results;
                            }
                        }
                    }
                }
            }

            return results;
        }
    }
}
#endif //L0L2_FULL_PATH_SOLVER_H
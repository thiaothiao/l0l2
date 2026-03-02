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

#include "leastsquares/utils.h"
#include "leastsquares/l2regressor/solver.h"
#include "leastsquares/fullpath/step.h"

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
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
                    /*! \brief l0l2 model parameter object constructor.
                      \param deltaInput sparsity regularization parameter.
                      \param betaInput l2 regularization parameter.
                      \param strategyInput enum indicating a strategy: from zero, or l2 or both solutions.
                    */
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

                /*! \brief A Full path solver object constructor.
                  \param param regularization parameters.
                  \param withIntercept boolean indicating with intercept or not. Default is false.
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

                std::list<Solution<Scalar>> solveFromZero(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance);

                std::list<Solution<Scalar>> solveFromL2(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance);

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

                FullPathSolver regressor{ Param{static_cast<Scalar>(-1), beta, strategy}, false };

                if (strategy == Strategy::FromZeroSolution)
                {
                    return regressor.solveFromZero(matData, vectData, matrixIsCovariance);
                }
                else if (strategy == Strategy::FromL2Solution)
                {
                    return regressor.solveFromL2(matData, vectData, matrixIsCovariance);
                }
                else
                {
                    auto fromZeroFuture =
                        std::async(std::launch::async, &FullPathSolver::solveFromZero, 
                            &regressor, matData, vectData, matrixIsCovariance);

                    auto fromL2SolutionResults = 
                        regressor.solveFromL2(matData, vectData, matrixIsCovariance);

                    auto fromZeroSolutionResults = fromZeroFuture.get();

                    const auto fromZeroSmallestDelta = fromZeroSolutionResults.back().delta - Utils::epsilon;

                    for (auto it = fromL2SolutionResults.begin(); it != fromL2SolutionResults.end(); ++it)
                    {
                        if (it->delta < fromZeroSmallestDelta)
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

                if (m_Param.strategy == Strategy::FromZeroSolution)
                {
                    auto results = solveFromZero(matData, vectData, matrixIsCovariance);
                    if (!results.empty())
                    {
                        auto minBoundIt = results.rbegin();

                        minBoundSolution = std::move(*minBoundIt);

                        maxBoundSolution = std::move(*++minBoundIt);
                    }
                }
                else if (m_Param.strategy == Strategy::FromL2Solution)
                {
                    auto results = solveFromL2(matData, vectData, matrixIsCovariance);
                    if (!results.empty())
                    {
                        auto maxBoundIt = results.begin();

                        maxBoundSolution = std::move(*maxBoundIt);

                        minBoundSolution = std::move(*++maxBoundIt);
                    }
                }
                else
                {
                    auto fromZeroFuture =
                        std::async(std::launch::async, &FullPathSolver::solveFromZero, 
                            this, matData, vectData, matrixIsCovariance);

                    auto fromL2SolutionResults = 
                        solveFromL2(matData, vectData, matrixIsCovariance);

                    auto fromZeroSolutionResults = fromZeroFuture.get();

                    if (!fromZeroSolutionResults.empty())
                    {
                        auto minBoundIt = fromZeroSolutionResults.rbegin();

                        minBoundSolution = std::move(*minBoundIt);

                        maxBoundSolution = std::move(*++minBoundIt);
                    }
                    else
                    {
                        auto maxBoundIt = fromL2SolutionResults.begin();

                        maxBoundSolution = std::move(*maxBoundIt);

                        minBoundSolution = std::move(*++maxBoundIt);
                    }
                }

                if (minBoundSolution.x.size() != 0 && maxBoundSolution.x.size() != 0)
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
                FullPathSolver<ScalarType>::solveFromZero(const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance)
            {
                using Vector = Vector<Scalar>;
                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;
                using L2Regressor = L2Regressor<Scalar>;
                using FullPathStep = FullPathStep<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto tau = static_cast<Scalar>(1);

                std::list<Solution> results;

                const Vector ATb = matData.transpose() * vectData;// same as Q\alpha
                {
                    const auto deltaZero = ATb.cwiseAbs().maxCoeff() / m_Param.beta;

                    results.emplace_back(
                        std::numeric_limits<Scalar>::max(), Vector::Zero(n), -ATb);

                    results.emplace_back(
                        deltaZero, Vector::Zero(n), -ATb);//TODO avoid repeating and optimize
                    
                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        m_DeltaFromZeroSolution = deltaZero;

                        if (m_DeltaFromZeroSolution <= std::max(Utils::epsilon, m_Param.delta))
                        {
                            return results;
                        }
                    }
                    else
                    {
                        m_DeltaFromZeroSolution = deltaZero;

                        if (m_DeltaFromZeroSolution <= std::max(Utils::epsilon, m_Param.delta))
                        {
                            return results;
                        }
                    }
                }

                unsigned int numberIters = 0;

                while (true)
                {
                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        if (m_Param.delta >= static_cast<Scalar>(0))
                        {
                            if (m_Param.delta <= m_DeltaFromL2Solution)
                            {
                                return std::list<Solution>{};
                            }
                        }
                        else
                        {
                            if (m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                            {
                                break;
                            }
                        }
                    }

                    ++numberIters;

                    {
                        const auto& solutionMaam = *(++results.crbegin()) ;

                        const auto& solutionYaay = results.back();

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

                        const auto solutionNewDelta = solutionNew.delta;

                        results.push_back(std::move(solutionNew));

                        if (m_Param.strategy == Strategy::FromBothSolutions)
                        {
                            const std::lock_guard<std::mutex> lock(m_DeltasMutex);

                            m_DeltaFromZeroSolution = solutionNewDelta;

                            if (m_DeltaFromZeroSolution <= std::max(Utils::epsilon, m_Param.delta))
                            {
                                return results;
                            }
                        }
                        else
                        {
                            m_DeltaFromZeroSolution = solutionNewDelta;

                            if (m_DeltaFromZeroSolution <= std::max(Utils::epsilon, m_Param.delta))
                            {
                                return results;
                            }
                        }
                    }

                    const auto& solutionLast =  results.back();
                    {
                        const auto fullPathDone = 
                            solutionLast.x.cwiseAbs().minCoeff() >= solutionLast.delta - Utils::epsilon;

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
                            if (normMax >= solutionLast.delta - Utils::epsilon)
                            {// constant for delta in [deltaCandidate, delta]. improve delta

                                auto absGradMax = std::numeric_limits<Scalar>::min();
                                for (Index i = 0; i < n; ++i)
                                {  // TODO optimize
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

                                    results.emplace_back(deltaCandidate, solutionLast.x, solutionLast.grad);

                                    if (m_Param.strategy == Strategy::FromBothSolutions)
                                    {
                                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                                        m_DeltaFromZeroSolution = deltaCandidate;
                                        if (m_DeltaFromZeroSolution <= std::max(Utils::epsilon, m_Param.delta))
                                        {
                                            return results;
                                        }
                                    }
                                    else
                                    {
                                        m_DeltaFromZeroSolution = deltaCandidate;
                                        if (m_DeltaFromZeroSolution <= std::max(Utils::epsilon, m_Param.delta))
                                        {
                                            return results;
                                        }
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
                        }
                    }
                }

                return results;
            }

            template<std::floating_point ScalarType>
            std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
                FullPathSolver<ScalarType>::solveFromL2(const Matrix<Scalar>& matData/*colmajor*/,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance)
            {
                using Vector = Vector<Scalar>;
                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;
                using L2Regressor = L2Regressor<Scalar>;
                using FullPathStep = FullPathStep<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto tau = static_cast<Scalar>(-1);

                std::list<Solution> results;

                const Vector ATb = matData.transpose() * vectData;// same as Q\alpha

                {
                    Solution solutionBar{ n };

                    solutionBar.x = L2Regressor(m_Param.beta).fitNoIntercept(matData, vectData, matrixIsCovariance);

                    auto deltaBar = std::numeric_limits<Scalar>::max();
                    for (Index i = 0; i < n; ++i)
                    {
                        // TODO optimize
                        const auto absWeight = std::abs(solutionBar.x[i]);
                        if (Utils::epsilon < absWeight && absWeight < deltaBar)
                        {
                            deltaBar = absWeight;
                        }
                    }

                    results.emplace_front(static_cast<Scalar>(0),
                        solutionBar.x,
                        solutionBar.grad);

                    solutionBar.delta = deltaBar;

                    results.push_front(std::move(solutionBar));

                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        m_DeltaFromL2Solution = deltaBar;

                        if (std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution
                            || static_cast<Scalar>(0) <= m_Param.delta &&
                            m_Param.delta <= m_DeltaFromL2Solution)
                        {
                            return results;
                        }
                    }
                    else
                    {
                        m_DeltaFromL2Solution = deltaBar;

                        if (std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution
                            || static_cast<Scalar>(0) <= m_Param.delta &&
                            m_Param.delta <= m_DeltaFromL2Solution)
                        {
                            return results;
                        }
                    }
                }

                unsigned int numberIters = 0;

                while (true)
                {
                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        if (m_Param.delta >= static_cast<Scalar>(0))
                        {
                            if (m_DeltaFromZeroSolution <= m_Param.delta)
                            {
                                return std::list<Solution>{};
                            }
                        }
                        else
                        {
                            if (m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)
                            {
                                break;
                            }
                        }
                    }

                    ++numberIters;

                    {
                        const auto& solutionMaam = *(++results.cbegin());

                        const auto& solutionYaay = results.front();

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

                        const auto deltaNew = solutionNew.delta;

                        results.push_front(std::move(solutionNew));

                        if (m_Param.strategy == Strategy::FromBothSolutions)
                        {
                            const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                            m_DeltaFromL2Solution = deltaNew;

                            if (std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution
                                || static_cast<Scalar>(0) <= m_Param.delta &&
                                m_Param.delta <= m_DeltaFromL2Solution)
                            {
                                return results;
                            }
                        }
                        else
                        {
                            m_DeltaFromL2Solution = deltaNew;

                            if (std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution
                                || static_cast<Scalar>(0) <= m_Param.delta &&
                                m_Param.delta <= m_DeltaFromL2Solution)
                            {
                                return results;
                            }
                        }
                    }

                    const auto& solutionLast = results.front();
                    {
                        const auto fullPathDone = solutionLast.x.norm() <= Utils::epsilon;

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
                            if (normMax > solutionLast.delta + Utils::epsilon)
                            {// constant for delta in [delta, normMax]. improve delta

                                const auto deltaCandidate = normMax;

                                results.emplace_front(deltaCandidate, solutionLast.x, solutionLast.grad);

                                if (m_Param.strategy == Strategy::FromBothSolutions)
                                {
                                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                                    m_DeltaFromL2Solution = deltaCandidate;

                                    if (std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution
                                        || static_cast<Scalar>(0) <= m_Param.delta &&
                                        m_Param.delta <= m_DeltaFromL2Solution)
                                    {
                                        return results;
                                    }
                                }
                                else
                                {
                                    m_DeltaFromL2Solution = deltaCandidate;

                                    if (std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution
                                        || static_cast<Scalar>(0) <= m_Param.delta &&
                                        m_Param.delta <= m_DeltaFromL2Solution)
                                    {
                                        return results;
                                    }
                                }

                                // TODO Not necessary to check zero a second time!
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
}
#endif //L0L2_FULL_PATH_SOLVER_H
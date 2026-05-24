#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <future>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <list>
#include <mutex>
#include <utility>

#include <l0l2/leastsquares/l2regressor/solver.hpp>
#include <l0l2/leastsquares/linear/step.hpp>
#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            /* \brief Linear solver class.
             *
             * It produces solutions from partial path or full path
             * computations.
             */
            template <std::floating_point ScalarType> class LinearSolver final
            {
              public:
                using Scalar = ScalarType;

                struct Param final
                {
                    /*! \brief l0l2 model parameter object constructor.
                      \param deltaInput sparsity regularization parameter.
                      \param betaInput l2 regularization parameter.
                      \param withInterceptInput consider intercept or not.
                      \param strategyInput enum indicating a strategy: from
                      zero, or l2 or parallel.
                    */
                    Param(Scalar deltaInput = static_cast<Scalar>(0),
                          Scalar betaInput = static_cast<Scalar>(1),
                          bool withInterceptInput = false,
                          Strategy strategyInput =
                              Strategy::SequentialFromZeroSolution)
                        : delta{deltaInput}, beta{betaInput},
                          withIntercept{withInterceptInput},
                          strategy{strategyInput}
                    {
                    }

                    const Scalar delta;
                    const Scalar beta;
                    const bool withIntercept;
                    const Strategy strategy;
                };

                /*! \brief A linear solver object constructor.
                  \param param regularization and other parameters.
                */
                LinearSolver(const Param &param)
                    : m_Param{param},
                      m_DeltaFromZeroSolution{
                          std::numeric_limits<Scalar>::max()},
                      m_DeltaFromL2Solution{static_cast<Scalar>(0)}
                {
                }

                /*! \brief Fit model.
                   \param matData matrix containing the features data, #columns
                   = #features, #rows = #samples.
                   \param vectData vector containing the targets.
                   \return a list containing one solution corresponding to the
                   regularization parameters if delta > 0 and a list of
                   solutions generating entire piecewise linear path solutions
                   otherwise.
                 */
                std::list<Solution<Scalar>> fit(const Matrix<Scalar> &matData,
                                                const Vector<Scalar> &vectData);

              private:
                std::list<Solution<Scalar>>
                fitNoIntercept(const Matrix<Scalar> &matData,
                               const Vector<Scalar> &vectData);

                std::list<Solution<Scalar>>
                fitAllNoIntercept(const Matrix<Scalar> &matData,
                                  const Vector<Scalar> &vectData);

                std::list<Solution<Scalar>>
                solveFromZero(const Matrix<Scalar> &matData,
                              const Vector<Scalar> &vectData);

                std::list<Solution<Scalar>>
                solveFromL2(const Matrix<Scalar> &matData,
                            const Vector<Scalar> &vectData);

                bool isDeltaTargetReachedFromZero(Scalar newDelta)
                {
                    m_DeltaFromZeroSolution = newDelta;
                    if (m_DeltaFromZeroSolution <=
                        std::max(Utils<Scalar>::epsilon, m_Param.delta))
                    {
                        return true;
                    }

                    return false;
                }

                bool isDeltaTargetReachedFromL2(Scalar newDelta)
                {
                    m_DeltaFromL2Solution = newDelta;
                    if (std::numeric_limits<Scalar>::max() <=
                            m_DeltaFromL2Solution ||
                        static_cast<Scalar>(0) <= m_Param.delta &&
                            m_Param.delta <= m_DeltaFromL2Solution)
                    {
                        return true;
                    }

                    return false;
                }

                const Param m_Param;

                volatile Scalar m_DeltaFromZeroSolution;
                volatile Scalar m_DeltaFromL2Solution;

                std::mutex m_DeltasMutex;
            };

            template <std::floating_point ScalarType>
            inline std::list<
                Solution<typename LinearSolver<ScalarType>::Scalar>>
            LinearSolver<ScalarType>::fit(const Matrix<Scalar> &matData,
                                            const Vector<Scalar> &vectData)
            {
                if (m_Param.withIntercept)
                {
                    auto results = fitNoIntercept(
                        matData.rowwise() - matData.colwise().mean(),
                        vectData.array() - vectData.mean());

                    for (auto &result : results)
                    {
                        result.intercept = (vectData - matData * result.x)
                                               .mean(); // TODO use grad!
                    }

                    return results;
                }

                return fitNoIntercept(matData, vectData);
            }

            template <std::floating_point ScalarType>
            std::list<Solution<typename LinearSolver<ScalarType>::Scalar>>
            LinearSolver<ScalarType>::fitNoIntercept(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData)
            {
                if (m_Param.delta < static_cast<Scalar>(0))
                {
                    return fitAllNoIntercept(matData, vectData);
                }

                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;

                Solution minBoundSolution{};
                Solution maxBoundSolution{};

                if (m_Param.strategy == Strategy::SequentialFromZeroSolution)
                {
                    auto results = solveFromZero(matData, vectData);
                    if (!results.empty())
                    {
                        auto minBoundIt = results.rbegin();

                        minBoundSolution = std::move(*minBoundIt);

                        maxBoundSolution = std::move(*++minBoundIt);
                    }
                }
                else if (m_Param.strategy == Strategy::SequentialFromL2Solution)
                {
                    auto results = solveFromL2(matData, vectData);
                    if (!results.empty())
                    {
                        auto maxBoundIt = results.begin();

                        maxBoundSolution = std::move(*maxBoundIt);

                        minBoundSolution = std::move(*++maxBoundIt);
                    }
                }
                else
                {
                    auto fromZeroFuture = std::async(
                        std::launch::async, &LinearSolver::solveFromZero,
                        this, matData, vectData);

                    auto fromL2SolutionResults = solveFromL2(matData, vectData);

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

                if (minBoundSolution.x.size() != 0 &&
                    maxBoundSolution.x.size() != 0)
                {
                    // interval found. minBoundSolution.delta <= delta <=
                    // maxBoundSolution.delta
                    auto &deltaMin = minBoundSolution.delta;
                    const auto deltaMax = maxBoundSolution.delta;

                    const auto &maxBoundSolutionx = maxBoundSolution.x;
                    const auto &maxBoundSolutiongrad = maxBoundSolution.grad;

                    auto &minBoundSolutionx = minBoundSolution.x;
                    auto &minBoundSolutiongrad = minBoundSolution.grad;

                    const auto diff = deltaMax - deltaMin;
                    const auto alpha = (diff > Utils::epsilon)
                                           ? (m_Param.delta - deltaMin) / diff
                                           : static_cast<Scalar>(0);

                    const auto oneMinusAlpha = static_cast<Scalar>(1) - alpha;

                    const auto n =
                        static_cast<Index>(minBoundSolution.x.size());

                    deltaMin = oneMinusAlpha * deltaMin + alpha * deltaMax;

                    minBoundSolutionx = oneMinusAlpha * minBoundSolutionx +
                                        alpha * maxBoundSolutionx;

                    minBoundSolutiongrad =
                        oneMinusAlpha * minBoundSolutiongrad +
                        alpha * maxBoundSolutiongrad;

                    return {minBoundSolution};
                }

                // Should throw an exception!!!
                return {Solution{}};
            }

            template <std::floating_point ScalarType>
            std::list<Solution<typename LinearSolver<ScalarType>::Scalar>>
            LinearSolver<ScalarType>::fitAllNoIntercept(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData)
            {
                using Solution = Solution<Scalar>;
                using Vector = Vector<Scalar>;
                using Utils = Utils<Scalar>;

                LinearSolver regressor{Param{m_Param.delta, m_Param.beta,
                                               false, m_Param.strategy}};

                if (m_Param.strategy == Strategy::SequentialFromZeroSolution)
                {
                    auto results = regressor.solveFromZero(matData, vectData);

                    // remove delta infinity solution
                    if (!results.empty())
                    {
                        results.pop_front();
                    }

                    return results;
                }
                else if (m_Param.strategy == Strategy::SequentialFromL2Solution)
                {
                    auto results = regressor.solveFromL2(matData, vectData);

                    // remove delta 0 solution
                    if (!results.empty())
                    {
                        results.pop_back();
                    }

                    return results;
                }
                else
                {
                    auto fromZeroFuture = std::async(
                        std::launch::async, &LinearSolver::solveFromZero,
                        &regressor, matData, vectData);

                    auto fromL2SolutionResults =
                        regressor.solveFromL2(matData, vectData);

                    auto fromZeroSolutionResults = fromZeroFuture.get();

                    if (!fromZeroSolutionResults.empty())
                    { // remove delta infinity solution
                        fromZeroSolutionResults.pop_front();
                    }

                    if (!fromL2SolutionResults.empty())
                    { // remove delta 0 solution
                        fromL2SolutionResults.pop_back();
                    }

                    const auto fromZeroSmallestDelta =
                        fromZeroSolutionResults.back().delta - Utils::epsilon;

                    for (auto it = fromL2SolutionResults.begin();
                         it != fromL2SolutionResults.end(); ++it)
                    {
                        if (it->delta < fromZeroSmallestDelta)
                        {
                            fromZeroSolutionResults.push_back(std::move(*it));
                        }
                    }

                    return fromZeroSolutionResults;
                }
            }

            template <std::floating_point ScalarType>
            std::list<Solution<typename LinearSolver<ScalarType>::Scalar>>
            LinearSolver<ScalarType>::solveFromZero(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData)
            {
                using Vector = Vector<Scalar>;
                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;
                using L2Regressor = LDLTL2Regressor<Scalar>;
                using PathStep = PathStep<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto tau = static_cast<Scalar>(1);

                std::list<Solution> results;

                const Vector ATb =
                    matData.transpose() * vectData; // same as Q\alpha
                {
                    const auto deltaZero =
                        ATb.cwiseAbs().maxCoeff() / m_Param.beta;

                    results.emplace_back(std::numeric_limits<Scalar>::max(),
                                         Vector::Zero(n), -ATb);

                    results.emplace_back(
                        deltaZero, Vector::Zero(n),
                        -ATb); // TODO avoid repeating and optimize

                    bool deltaTargetReached = false;
                    if (m_Param.strategy == Strategy::Parallel)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        deltaTargetReached =
                            isDeltaTargetReachedFromZero(deltaZero);
                    }
                    else
                    {
                        deltaTargetReached =
                            isDeltaTargetReachedFromZero(deltaZero);
                    }

                    if (deltaTargetReached)
                    {
                        return results;
                    }
                }

                unsigned int numberIters = 0;

                while (true)
                {
                    if (m_Param.strategy == Strategy::Parallel)
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
                            if (m_DeltaFromZeroSolution <=
                                m_DeltaFromL2Solution)
                            {
                                break;
                            }
                        }
                    }

                    ++numberIters;

                    {
                        const auto &solutionMaam = *(++results.crbegin());

                        const auto &solutionYaay = results.back();

                        const auto step = PathStep(m_Param.beta);

                        auto solutionNew = step.run(matData, ATb, solutionYaay,
                                                    solutionMaam, tau);

                        const auto gammaNew =
                            tau * (solutionYaay.delta - solutionNew.delta);

                        if (gammaNew <= Utils::epsilon)
                        {
                            std::cout << "\nSTART CYCLING!\n";

                            std::cout << std::fixed << std::setprecision(15);
                            std::cout << "\nGamma: " << gammaNew
                                      << " is less than " << Utils::epsilon
                                      << "\n";

                            break;
                        }

                        const auto solutionNewDelta = solutionNew.delta;

                        results.push_back(std::move(solutionNew));

                        bool deltaTargetReached = false;
                        if (m_Param.strategy == Strategy::Parallel)
                        {
                            const std::lock_guard<std::mutex> lock(
                                m_DeltasMutex);
                            deltaTargetReached =
                                isDeltaTargetReachedFromZero(solutionNewDelta);
                        }
                        else
                        {
                            deltaTargetReached =
                                isDeltaTargetReachedFromZero(solutionNewDelta);
                        }

                        if (deltaTargetReached)
                        {
                            return results;
                        }
                    }

                    const auto &solutionLast = results.back();
                    {
                        const auto fullPathDone =
                            solutionLast.x.cwiseAbs().minCoeff() >=
                            solutionLast.delta - Utils::epsilon;

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
                            { // constant for delta in [deltaCandidate, delta].
                              // improve delta

                                auto absGradMax =
                                    std::numeric_limits<Scalar>::min();
                                for (Index i = 0; i < n; ++i)
                                { // TODO optimize
                                    const auto absXi =
                                        std::abs(solutionLast.x[i]);
                                    if (absXi <= Utils::epsilon)
                                    {
                                        const auto absGradi =
                                            std::abs(solutionLast.grad[i]);
                                        if (absGradi > absGradMax)
                                        {
                                            absGradMax = absGradi;
                                        }
                                    }
                                }

                                if (absGradMax >
                                        std::numeric_limits<Scalar>::min() &&
                                    absGradMax / m_Param.beta <
                                        solutionLast.delta - Utils::epsilon)
                                {
                                    const auto deltaCandidate =
                                        absGradMax / m_Param.beta;

                                    results.emplace_back(deltaCandidate,
                                                         solutionLast.x,
                                                         solutionLast.grad);

                                    bool deltaTargetReached = false;
                                    if (m_Param.strategy ==
                                        Strategy::Parallel)
                                    {
                                        const std::lock_guard<std::mutex> lock(
                                            m_DeltasMutex);
                                        deltaTargetReached =
                                            isDeltaTargetReachedFromZero(
                                                deltaCandidate);
                                    }
                                    else
                                    {
                                        deltaTargetReached =
                                            isDeltaTargetReachedFromZero(
                                                deltaCandidate);
                                    }

                                    if (deltaTargetReached)
                                    {
                                        return results;
                                    }

                                    // TODO try to optimize using minmax and
                                    // avoid abs
                                    const auto fullPathDone =
                                        results.back()
                                            .x.cwiseAbs()
                                            .minCoeff() >=
                                        results.back().delta - Utils::epsilon;

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

            template <std::floating_point ScalarType>
            std::list<Solution<typename LinearSolver<ScalarType>::Scalar>>
            LinearSolver<ScalarType>::solveFromL2(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData)
            {
                using Vector = Vector<Scalar>;
                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;
                using L2Regressor = LDLTL2Regressor<Scalar>;
                using PathStep = PathStep<Scalar>;

                const auto n = static_cast<Index>(matData.cols());

                const auto tau = static_cast<Scalar>(-1);

                std::list<Solution> results;

                const Vector ATb =
                    matData.transpose() * vectData; // same as Q\alpha

                {
                    auto solutionBar =
                        L2Regressor(m_Param.beta, false).fit(matData, vectData);

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

                    results.emplace_front(static_cast<Scalar>(0), solutionBar.x,
                                          solutionBar.grad);

                    solutionBar.delta = deltaBar;

                    results.push_front(std::move(solutionBar));

                    bool deltaTargetReached = false;
                    if (m_Param.strategy == Strategy::Parallel)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        deltaTargetReached =
                            isDeltaTargetReachedFromL2(deltaBar);
                    }
                    else
                    {
                        deltaTargetReached =
                            isDeltaTargetReachedFromL2(deltaBar);
                    }

                    if (deltaTargetReached)
                    {
                        return results;
                    }
                }

                unsigned int numberIters = 0;

                while (true)
                {
                    if (m_Param.strategy == Strategy::Parallel)
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
                            if (m_DeltaFromZeroSolution <=
                                m_DeltaFromL2Solution)
                            {
                                break;
                            }
                        }
                    }

                    ++numberIters;

                    {
                        const auto &solutionMaam = *(++results.cbegin());

                        const auto &solutionYaay = results.front();

                        const auto step = PathStep(m_Param.beta);

                        auto solutionNew = step.run(matData, ATb, solutionYaay,
                                                    solutionMaam, tau);

                        const auto gammaNew =
                            tau * (solutionYaay.delta - solutionNew.delta);

                        if (gammaNew <= Utils::epsilon)
                        {
                            std::cout << "\nSTART CYCLING!\n";

                            std::cout << std::fixed << std::setprecision(15);
                            std::cout << "\nGamma: " << gammaNew
                                      << " is less than " << Utils::epsilon
                                      << "\n";

                            break;
                        }

                        const auto deltaNew = solutionNew.delta;

                        results.push_front(std::move(solutionNew));

                        bool deltaTargetReached = false;
                        if (m_Param.strategy == Strategy::Parallel)
                        {
                            const std::lock_guard<std::mutex> lock(
                                m_DeltasMutex);
                            deltaTargetReached =
                                isDeltaTargetReachedFromL2(deltaNew);
                        }
                        else
                        {
                            deltaTargetReached =
                                isDeltaTargetReachedFromL2(deltaNew);
                        }

                        if (deltaTargetReached)
                        {
                            return results;
                        }
                    }

                    const auto &solutionLast = results.front();
                    {
                        const auto fullPathDone =
                            solutionLast.x.norm() <= Utils::epsilon;

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
                            { // constant for delta in [delta, normMax]. improve
                              // delta

                                const auto deltaCandidate = normMax;

                                results.emplace_front(deltaCandidate,
                                                      solutionLast.x,
                                                      solutionLast.grad);

                                bool deltaTargetReached = false;
                                if (m_Param.strategy ==
                                    Strategy::Parallel)
                                {
                                    const std::lock_guard<std::mutex> lock(
                                        m_DeltasMutex);
                                    deltaTargetReached =
                                        isDeltaTargetReachedFromL2(
                                            deltaCandidate);
                                }
                                else
                                {
                                    deltaTargetReached =
                                        isDeltaTargetReachedFromL2(
                                            deltaCandidate);
                                }

                                if (deltaTargetReached)
                                {
                                    return results;
                                }

                                // TODO Not necessary to check zero a second
                                // time!
                                const auto fullPathDone =
                                    results.front().x.norm() <= Utils::epsilon;

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
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

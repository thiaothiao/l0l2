#pragma once

#include <cmath>
#include <concepts>
#include <iostream>
#include <set>
#include <utility>

#include <l0l2/leastsquares/cycliccoordinatedescent/l0l2.hpp>
#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            // L0L2 branch and bound implementation
            template <std::floating_point ScalarType>
            class SimpleBranchAndBound final
            {
              public:
                using Scalar = ScalarType;
                using Regressor = L0L2Regressor<Scalar>;
                using RegressorParam = L0L2Regressor<Scalar>::Param;

                class Branch
                {
                  public:
                    Branch(const CoordinateStates &coordinateStatesInput = {},
                           Scalar lbInput = static_cast<Scalar>(0))
                        : coordinateStates{coordinateStatesInput}, lb{lbInput}
                    {
                    }

                    Branch(const Branch &) = delete;
                    Branch &operator=(const Branch &) = delete;

                    Branch(Branch &&) = default;
                    Branch &operator=(Branch &&) = default;

                    CoordinateStates coordinateStates;
                    Scalar lb;
                };

                struct BranchCompare
                {
                    bool operator()(const Branch &lhs, const Branch &rhs) const
                    {
                        return lhs.lb < rhs.lb;
                    }
                };

                struct Param final
                {
                    Param(
                        const RegressorParam &regressorParamInput,
                        Scalar globalEpsilonInput = static_cast<Scalar>(1e-8),
                        Scalar localEpsilonInput = static_cast<Scalar>(1e-8),
                        unsigned int maximumNumberOfIterationsInput = 1000000U)
                        : regressorParam{regressorParamInput},
                          globalEpsilon{globalEpsilonInput},
                          localEpsilon{localEpsilonInput},
                          maximumNumberOfIterations{
                              maximumNumberOfIterationsInput}
                    {
                    }

                    Param(const Param &) = default;
                    Param &operator=(const Param &) = default;

                    Param(Param &&) = default;
                    Param &operator=(Param &&) = default;

                    const RegressorParam regressorParam;
                    const Scalar globalEpsilon;
                    const Scalar localEpsilon;
                    const unsigned int maximumNumberOfIterations;
                };

                SimpleBranchAndBound(const Param &param) : m_Param{param} {}

                Solution<Scalar> fit(const Matrix<Scalar> &matData,
                                     const Vector<Scalar> &vectData);

                Scalar objectiveValue(const Matrix<Scalar> &matData,
                                      const Vector<Scalar> &vectData,
                                      const Solution<Scalar> &solution) const;

                Scalar relaxationValue(const Matrix<Scalar> &matData,
                                       const Vector<Scalar> &vectData,
                                       const CoordinateStates &coordinateStates,
                                       const Solution<Scalar> &solution) const;

              private:
                const Param m_Param;
            };

            template <std::floating_point ScalarType>
            inline SimpleBranchAndBound<ScalarType>::Scalar
            SimpleBranchAndBound<ScalarType>::objectiveValue(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                const Solution<Scalar> &solution) const
            {
                return ((vectData - matData * solution.x).array() -
                        solution.intercept)
                           .matrix()
                           .squaredNorm() +
                       m_Param.regressorParam.beta * solution.x.squaredNorm() +
                       m_Param.regressorParam.beta *
                           m_Param.regressorParam.delta *
                           m_Param.regressorParam.delta *
                           (solution.x.cwiseAbs().array() >
                            Utils<Scalar>::epsilon)
                               .count();
            }

            template <std::floating_point ScalarType>
            SimpleBranchAndBound<ScalarType>::Scalar
            SimpleBranchAndBound<ScalarType>::relaxationValue(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                const CoordinateStates &coordinateStates,
                const Solution<Scalar> &solution) const
            {
                const auto n = static_cast<Index>(matData.cols());
                const auto delta = m_Param.regressorParam.delta;
                const auto beta = m_Param.regressorParam.beta;

                const auto &x = solution.x;
                const auto intercept = solution.intercept;

                const auto twoDeltaBeta = static_cast<Scalar>(2) * delta * beta;
                const auto betaSquaredDelta = beta * delta * delta;

                auto objVal = ((vectData - matData * x).array() - intercept)
                                  .matrix()
                                  .squaredNorm();

                for (Index j = 0; j < n; ++j)
                {
                    if (coordinateStates[j] == CoordinateState::Unknown)
                    {
                        if (std::abs(x[j]) >= delta)
                        {
                            objVal += beta * x[j] * x[j] + betaSquaredDelta;
                        }
                        else
                        {
                            objVal += twoDeltaBeta * std::abs(x[j]);
                        }
                    }
                    else if (std::abs(x[j]) > Utils<Scalar>::epsilon)
                    {
                        objVal += beta * x[j] * x[j] + betaSquaredDelta;
                    }
                }

                return objVal;
            }

            template <std::floating_point ScalarType>
            Solution<typename SimpleBranchAndBound<ScalarType>::Scalar>
            SimpleBranchAndBound<ScalarType>::fit(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData)
            {
                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;

                const auto n = static_cast<Index>(matData.cols());
                const auto delta = m_Param.regressorParam.delta;

                std::set<Branch, BranchCompare> allBranches;

                allBranches.emplace(
                    CoordinateStates::Constant(n, CoordinateState::Unknown),
                    static_cast<Scalar>(0));

                auto solution = Solution(n);
                auto lowerBound = static_cast<Scalar>(0);
                auto upperBound = vectData.squaredNorm();

                unsigned int numberOfIterations = 0;
                while (!allBranches.empty())
                {
                    Branch branch = std::move(
                        allBranches.extract(allBranches.begin()).value());

                    lowerBound = branch.lb;

                    const auto globalGap =
                        (upperBound - lowerBound) / upperBound;
                    if (globalGap <= m_Param.globalEpsilon)
                    {
                        break;
                    }

                    auto relaxationSolution =
                        Regressor{m_Param.regressorParam}.fit(
                            matData, vectData, branch.coordinateStates);

                    const auto ub =
                        objectiveValue(matData, vectData, relaxationSolution);

                    const auto primal = relaxationValue(matData, vectData,
                                                        branch.coordinateStates,
                                                        relaxationSolution);

                    const auto lb = primal * (static_cast<Scalar>(1) -
                                              relaxationSolution.dualityGap);

                    const auto localGap = (ub - lb) / ub;
                    if (localGap > m_Param.localEpsilon)
                    { // Branching. Taking the smallest non compliant coordinate
                        auto splitIndex = static_cast<Index>(-1);
                        auto minAbsXj = delta;
                        for (Index j = 0; j < n; ++j)
                        {
                            if (branch.coordinateStates[j] ==
                                CoordinateState::Unknown)
                            {
                                const auto absXj =
                                    std::abs(relaxationSolution.x[j]);
                                if (Utils::epsilon < absXj && absXj < minAbsXj)
                                {
                                    minAbsXj = absXj;
                                    splitIndex = j;
                                }
                            }
                        }

                        if (splitIndex >= static_cast<Index>(0))
                        {
                            branch.lb = std::max(lowerBound, lb);

                            branch.coordinateStates[splitIndex] =
                                CoordinateState::Zero;
                            allBranches.emplace(branch.coordinateStates,
                                                branch.lb);

                            branch.coordinateStates[splitIndex] =
                                CoordinateState::Nonzero;
                            allBranches.insert(std::move(branch));
                        }
                    }

                    if (ub < upperBound)
                    {
                        upperBound = ub;
                        solution = std::move(relaxationSolution);
                    }

                    ++numberOfIterations;
                    if (numberOfIterations > m_Param.maximumNumberOfIterations)
                    { // leave too many branches
                        break;
                    }
                }

                std::cout << "\nNumber of iterates: " << numberOfIterations
                          << "\n";
                return solution;
            }
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

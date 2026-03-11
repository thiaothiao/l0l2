#ifndef L0L2_BRANCH_AND_BOUND_SOLVER_HPP
#define L0L2_BRANCH_AND_BOUND_SOLVER_HPP

#include <iostream>
#include <future>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <concepts>
#include <utility>
#include <set>

#include "l0l2/leastsquares/utils.hpp"
#include "l0l2/leastsquares/cycliccoordinatedescent/solver.hpp"

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            // L0L2 branch and bound implementation
            template<std::floating_point ScalarType>
            class SimpleBranchAndBound final
            {
            public:
                using Scalar = ScalarType;
                using Regressor = L0L2Regressor<Scalar>;
                using RegressorParam = L0L2Regressor<Scalar>::Param;
                using ConstraintsType = Regressor::ConstraintsType;

                class Branch
                {
                public:
                    Branch(const std::vector<ConstraintsType>& indices_ = {},
                        double lb_ = 0.0) :
                        indices{ indices_ },
                        lb{ lb_ }
                    {
                    }

                    Branch(const Branch&) = delete;
                    Branch& operator=(const Branch&) = delete;

                    Branch(Branch&&) = default;
                    Branch& operator=(Branch&&) = default;

                    std::vector<ConstraintsType> indices;
                    Scalar lb;
                };

                struct Param final
                {
                    Param(
                        Scalar deltaInput = static_cast<Scalar>(0),
                        Scalar betaInput = static_cast<Scalar>(1),
                        bool hasInterceptInput = false,
                        leastsquares::Strategy strategyInput = leastsquares::Strategy::FromZeroSolution,
                        Scalar toleranceInput = static_cast<Scalar>(1e-4),
                        unsigned int maximumNumberOfIterationsInput = 10000U,
                        Scalar innerEpsilonInput = static_cast<Scalar>(1e-6),
                        unsigned int innerMaximumNumberOfIterationsInput = 100000U)
                        :regressorParam{ deltaInput, betaInput, hasInterceptInput, strategyInput,
                        toleranceInput, maximumNumberOfIterationsInput,
                        innerEpsilonInput, innerMaximumNumberOfIterationsInput }
                    {
                    }

                    Param(const Param&) = default;
                    Param& operator=(const Param&) = default;

                    Param(Param&&) = default;
                    Param& operator=(Param&&) = default;

                    const RegressorParam regressorParam;
                };
                
                SimpleBranchAndBound(const Param& param)
                    : m_Param{ param }
                {
                }

                Solution<Scalar> fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance);

                Scalar objectiveValue(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const Solution<Scalar>& solution) const;

                Scalar relaxationValue(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const std::vector<ConstraintsType>& indices,
                    const Solution<Scalar>& solution) const;

            private:
                const Param m_Param;
            };

            template<std::floating_point ScalarType>
            inline SimpleBranchAndBound<ScalarType>::Scalar
                SimpleBranchAndBound<ScalarType>::objectiveValue(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                const Solution<Scalar>& solution) const
            {
                return matrixIsCovariance
                    ? (vectData - solution.x).dot(matData * (vectData - solution.x))
                    : ((vectData - matData * solution.x).array() - solution.intercept).matrix().squaredNorm()
                    + m_Param.regressorParam.beta * solution.x.squaredNorm()
                    + m_Param.regressorParam.beta * m_Param.regressorParam.delta * m_Param.regressorParam.delta
                    * (solution.x.cwiseAbs().array() > Utils<Scalar>::epsilon).count();
            }

            template<std::floating_point ScalarType>
            SimpleBranchAndBound<ScalarType>::Scalar
                SimpleBranchAndBound<ScalarType>::relaxationValue(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const std::vector<ConstraintsType>& indices,
                    const Solution<Scalar>& solution) const
            {
                const auto n = static_cast<Index>(matData.cols());
                const auto delta = m_Param.regressorParam.delta;
                const auto beta = m_Param.regressorParam.beta;

                const auto& x = solution.x;
                const auto intercept = solution.intercept;

                const auto twoDeltaBeta = static_cast<Scalar>(2) * delta * beta;
                const auto betaSquaredDelta = beta * delta * delta;

                auto objVal = matrixIsCovariance
                    ? (vectData - x).dot(matData * (vectData - x))
                    : ((vectData - matData * x).array() - intercept).matrix().squaredNorm();

                for (Index j = 0; j < n; ++j)
                {
                    if (indices[j] == ConstraintsType::L0)
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

            template<std::floating_point ScalarType>
            Solution<typename SimpleBranchAndBound<ScalarType>::Scalar>
                SimpleBranchAndBound<ScalarType>::fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance)
            {
                using Solution = Solution<Scalar>;
                using Utils = Utils<Scalar>;

                const auto n = static_cast<Index>(matData.cols());
                const auto m = static_cast<Index>(matData.rows());
                const auto delta = m_Param.regressorParam.delta;
                const auto beta = m_Param.regressorParam.beta;

                auto compareLambda =
                    [](const Branch& lhs, const Branch& rhs){ return lhs.lb < rhs.lb; };

                std::set<Branch, decltype(compareLambda)> allBranches(compareLambda);
                allBranches.emplace(std::vector<ConstraintsType>(n, ConstraintsType::L0), 
                        static_cast<Scalar>(0));

                auto solution = Solution(n);
                auto lowerBound = static_cast<Scalar>(0);
                auto upperBound = vectData.squaredNorm();

                std::cout << "\nStarting b&b.\n";
                const auto globalEpsilon = static_cast<Scalar>(1e-8);
                const auto localEpsilon = static_cast<Scalar>(1e-8);
                const unsigned int maximumNumberOfIterations = 1000000U;
                unsigned int numberOfIterations = 0;
                while (!allBranches.empty())
                {
                    Branch branch = std::move(allBranches.extract(allBranches.begin()).value());
                    
                    lowerBound = branch.lb;

                    const auto globalGap = (upperBound - lowerBound) / upperBound;
                    if (globalGap <= globalEpsilon)
                    {
                        break;
                    }

                    auto branchSolution = Regressor{ m_Param.regressorParam}
                    .fitPartial(matData, vectData, matrixIsCovariance, branch.indices);

                    const auto ub =
                        objectiveValue(matData, vectData, matrixIsCovariance, branchSolution);

                    const auto lb =
                        relaxationValue(matData, vectData, matrixIsCovariance, branch.indices, branchSolution);

                    const auto localGap = (ub - lb) / ub;
                    if (localGap > localEpsilon)
                    {//Branching. Taking the smallest non compliant coordinate
                        auto splitIndex = static_cast<Index>(-1);
                        auto minAbsXj = delta;
                        for (Index j = 0; j < n; ++j)
                        {
                            if (branch.indices[j] == ConstraintsType::L0)
                            {
                                const auto absXj = std::abs(branchSolution.x[j]);
                                if (Utils::epsilon < absXj && absXj < minAbsXj)
                                {
                                    minAbsXj = absXj;
                                    splitIndex = j;

                                    //break;
                                }
                            }
                        }

                        if (splitIndex >= static_cast<Index>(0))
                        {
                            branch.lb = std::max(lowerBound, lb);

                            branch.indices[splitIndex] = ConstraintsType::ZERO;
                            allBranches.emplace(branch.indices, branch.lb);

                            branch.indices[splitIndex] = ConstraintsType::FREE;
                            allBranches.insert(std::move(branch));
                        }
                    }

                    if (ub < upperBound)
                    {
                        upperBound = ub;
                        solution = std::move(branchSolution);
                    }

                    ++numberOfIterations;
                    if (numberOfIterations > maximumNumberOfIterations)
                    {// leave too many branches
                        break;
                    }
                }

                std::cout << "\nNumber of iterates: " << numberOfIterations << "\n";
                return solution;
            }
        }
    }
}
#endif //L0L2_BRANCH_AND_BOUND_SOLVER_HPP
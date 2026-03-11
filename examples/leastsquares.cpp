#include <iostream>
#include <ios>
#include <chrono>

#include <string>
#include <string_view>

#include "l0l2/version.hpp"
#include "l0l2/core.hpp"
#include "simu.hpp"

int main()
{
    using Scalar = double;

    using Vector = l0l2::Vector<Scalar>;
    using Matrix = l0l2::Matrix<Scalar>;
    using Index = l0l2::Index;
    using FullPathSolver = l0l2::linearmodel::leastsquares::FullPathSolver<Scalar>;
    using FullPathSolverParam = FullPathSolver::Param;
    using L0L2Regressor = l0l2::linearmodel::leastsquares::L0L2Regressor<Scalar>;
    using L0L2RegressorParam = L0L2Regressor::Param;
    using Utils = l0l2::Utils<Scalar>;
    using Strategy = l0l2::linearmodel::leastsquares::Strategy;
    using SimpleBranchAndBound = l0l2::linearmodel::leastsquares::SimpleBranchAndBound<Scalar>;
    using SimpleBranchAndBoundParam = SimpleBranchAndBound::Param;

    constexpr std::string_view version = L0L2_MACRO_STRINGIFY(L0L2_VERSION);
    std::cout << "l0l2 library version " << version << "\n";

    const auto matrixIsCovariance = false;
    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(10000.0); //static_cast<Scalar>(-1); //static_cast<Scalar>(10000.0);

    const auto withIntercept = false;

    const auto fullpath = delta < static_cast<Scalar>(0);

    std::cout << std::boolalpha;
    std::cout << "\nmatrix is covariance: " << matrixIsCovariance;
    std::cout << "\nbeta: " << beta;
    std::cout << "\ndelta: " << delta;
    std::cout << "\nfullpath: " << fullpath << "\n";

    const std::streamsize streamSize = 6;

    const auto [mat, vect] = l0l2::linearmodel::diabetes<Scalar>();

    const auto n = static_cast<Index>(mat.cols());
    const auto m = static_cast<Index>(mat.rows());

    const Scalar tolerance = static_cast<Scalar>(1e-6);
    const unsigned int maximumNumberOfIterations = 100000U;

    const Scalar innerEpsilon = static_cast<Scalar>(1e-6);
    const unsigned int innerMaximumNumberOfIterations = 100000U;

    std::vector<L0L2Regressor::ConstraintsType> indices(n, L0L2Regressor::ConstraintsType::L0);

    if (fullpath)
    {
        const auto start = std::chrono::high_resolution_clock::now();

        auto results = FullPathSolver::fitAll( mat, vect, matrixIsCovariance, 
            beta, withIntercept, Strategy::FromL2Solution);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto durationUs = 
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count() << " microseconds!\n\n";

        for (auto it = results.begin(); it != results.end(); ++it)
        {
            const auto& result = *it;
            std::cout << l0l2::linearmodel::toStringAll(result, streamSize);
            std::cout << " \n";
        }
    }
    else
    {
        const auto bbstart = std::chrono::high_resolution_clock::now();

        const Strategy bbstrategy = Strategy::FromZeroSolution;

        //SimpleBranchAndBound::RegressorParam regressorParam{ delta, beta, strategy, tolerance,
        //    maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

        SimpleBranchAndBoundParam param{ delta, beta, withIntercept, bbstrategy, tolerance,
            maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

        auto branchandbound = SimpleBranchAndBound{ param };

        const auto bbsolution = branchandbound.fit(mat, vect, matrixIsCovariance);

        const auto bbstop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto bbdurationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(bbstop - bbstart);
        std::cout << "\n\nbranch and bound job done in "
            << bbdurationUs.count() << " microseconds!\n\n";
        std::cout << "\nobjective value: "
            << branchandbound.objectiveValue(mat, vect, matrixIsCovariance, bbsolution) << "\n";
        std::cout << "\nsolution\n"
            << l0l2::linearmodel::toString(bbsolution, streamSize) << "\n";

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromZeroSolution;

            L0L2RegressorParam param{ delta, beta, withIntercept, strategy, tolerance,
                maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

            const auto solution = L0L2Regressor{ param }
                .fit(mat, vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs = 
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in " 
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nobjective value: "
                << branchandbound.objectiveValue(mat, vect, matrixIsCovariance, solution) << "\n";
            std::cout << "\nsolution\n" 
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromL2Solution;

            L0L2RegressorParam param{ delta, beta, withIntercept, strategy, tolerance,
                maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

            const auto solution = L0L2Regressor{ param }
                .fit(mat, vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs = 
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in " 
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n" 
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromBothSolutions;

            L0L2RegressorParam param{ delta, beta, withIntercept, strategy, tolerance,
                maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

            const auto solution = L0L2Regressor{ param }
                .fit(mat, vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs = 
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in " 
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n" 
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromZeroSolution;

            L0L2RegressorParam param{ delta, beta, withIntercept, strategy, tolerance,
                maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

            const auto solution = L0L2Regressor{ param }
            .fitPartial(mat, vect, matrixIsCovariance, indices);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs =
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in "
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n"
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromL2Solution;

            L0L2RegressorParam param{ delta, beta, withIntercept, strategy, tolerance,
                maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

            const auto solution = L0L2Regressor{ param }
            .fitPartial(mat, vect, matrixIsCovariance, indices);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs =
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in "
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n"
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromBothSolutions;

            L0L2RegressorParam param{ delta, beta, withIntercept, strategy, tolerance,
                maximumNumberOfIterations, innerEpsilon, innerMaximumNumberOfIterations };

            const auto solution = L0L2Regressor{ param }
            .fitPartial(mat, vect, matrixIsCovariance, indices);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs =
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in "
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n"
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            FullPathSolverParam param{ delta, beta, Strategy::FromZeroSolution };

            auto solution = FullPathSolver{param , withIntercept}
                .fit(mat, vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs = 
                std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in " 
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n" 
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            FullPathSolverParam param{ delta, beta, Strategy::FromL2Solution };

            auto solution = FullPathSolver{param , withIntercept}
                .fit(mat, vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto durationUs 
                = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\njob done in " 
                << durationUs.count() << " microseconds!\n\n";
            std::cout << "\nsolution\n" 
                << l0l2::linearmodel::toString(solution, streamSize) << "\n";
        }
    }

    return 0;
}
#include <chrono>
#include <ios>
#include <iostream>

#include <string_view>

#include <../../simu.hpp>
#include <l0l2/core.hpp>
#include <l0l2/version.hpp>

int main()
{
    using Scalar = double;

    using L0L2Regressor =
        l0l2::linearmodel::leastsquares::L0L2Regressor<Scalar>;
    using L0L2RegressorParam = L0L2Regressor::Param;

    using Strategy = l0l2::linearmodel::leastsquares::Strategy;

    std::cout << "l0l2 library version " << l0l2::metadata::libVersion << "\n";

    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(10000.0);

    std::cout << "\nbeta: " << beta;
    std::cout << "\ndelta: " << delta;

    const std::streamsize streamSize = 6;

    const auto [mat, vect] = l0l2::linearmodel::diabetes<Scalar>();

    const Scalar tolerance = static_cast<Scalar>(1e-6);
    const unsigned int maximumNumberOfIterations = 100000U;

    const Scalar innerEpsilon = static_cast<Scalar>(1e-6);
    const unsigned int innerMaximumNumberOfIterations = 100000U;

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy strategy = Strategy::SequentialFromZeroSolution;

        const auto withIntercept = false;

        L0L2RegressorParam param{delta,         beta,
                                 withIntercept, strategy,
                                 tolerance,     maximumNumberOfIterations,
                                 innerEpsilon,  innerMaximumNumberOfIterations};

        const auto solution = L0L2Regressor{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy strategy = Strategy::SequentialFromL2Solution;

        const auto withIntercept = false;

        L0L2RegressorParam param{delta,         beta,
                                 withIntercept, strategy,
                                 tolerance,     maximumNumberOfIterations,
                                 innerEpsilon,  innerMaximumNumberOfIterations};

        const auto solution = L0L2Regressor{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy strategy = Strategy::Parallel;

        const auto withIntercept = false;

        L0L2RegressorParam param{delta,         beta,
                                 withIntercept, strategy,
                                 tolerance,     maximumNumberOfIterations,
                                 innerEpsilon,  innerMaximumNumberOfIterations};

        const auto solution = L0L2Regressor{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy strategy = Strategy::SequentialFromZeroSolution;

        const auto withIntercept = true;

        L0L2RegressorParam param{delta,         beta,
                                 withIntercept, strategy,
                                 tolerance,     maximumNumberOfIterations,
                                 innerEpsilon,  innerMaximumNumberOfIterations};

        const auto solution = L0L2Regressor{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy strategy = Strategy::SequentialFromL2Solution;

        const auto withIntercept = true;

        L0L2RegressorParam param{delta,         beta,
                                 withIntercept, strategy,
                                 tolerance,     maximumNumberOfIterations,
                                 innerEpsilon,  innerMaximumNumberOfIterations};

        const auto solution = L0L2Regressor{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy strategy = Strategy::Parallel;

        const auto withIntercept = true;

        L0L2RegressorParam param{delta,         beta,
                                 withIntercept, strategy,
                                 tolerance,     maximumNumberOfIterations,
                                 innerEpsilon,  innerMaximumNumberOfIterations};

        const auto solution = L0L2Regressor{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    return 0;
}

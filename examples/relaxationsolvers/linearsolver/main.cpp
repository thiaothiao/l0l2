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

    using LinearSolver =
        l0l2::linearmodel::leastsquares::LinearSolver<Scalar>;
    using LinearSolverParam = LinearSolver::Param;

    using Strategy = l0l2::linearmodel::leastsquares::Strategy;

    std::cout << "l0l2 library version " << l0l2::metadata::libVersion << "\n";

    const Scalar beta = static_cast<Scalar>(0.1);

    std::cout << std::boolalpha;
    std::cout << "\nbeta: " << beta;

    const auto withIntercept = false;

    const std::streamsize streamSize = 6;

    const auto [mat, vect] = l0l2::linearmodel::diabetes<Scalar>();

    const Scalar tolerance = static_cast<Scalar>(1e-6);
    const unsigned int maximumNumberOfIterations = 100000U;

    const Scalar innerEpsilon = static_cast<Scalar>(1e-6);
    const unsigned int innerMaximumNumberOfIterations = 100000U;

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto delta =
            static_cast<Scalar>(-1); // delta <= 0 => fullpath computation

        const Strategy strategy = Strategy::SequentialFromL2Solution;

        LinearSolverParam param{delta, beta, withIntercept, strategy};

        auto results = LinearSolver{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";

        for (auto it = results.begin(); it != results.end(); ++it)
        {
            const auto &result = *it;
            std::cout << l0l2::linearmodel::toStringAll(result, streamSize);
            std::cout << " \n";
        }
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto delta =
            static_cast<Scalar>(-1); // delta <= 0 => fullpath computation

        const Strategy strategy = Strategy::SequentialFromZeroSolution;

        LinearSolverParam param{delta, beta, withIntercept, strategy};

        auto results = LinearSolver{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";

        for (auto it = results.begin(); it != results.end(); ++it)
        {
            const auto &result = *it;
            std::cout << l0l2::linearmodel::toStringAll(result, streamSize);
            std::cout << " \n";
        }
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto delta = static_cast<Scalar>(10000.0);

        const Strategy strategy = Strategy::SequentialFromZeroSolution;

        LinearSolverParam param{delta, beta, withIntercept, strategy};

        auto solutions = LinearSolver{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solutions.front(), streamSize)
                  << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const auto delta = static_cast<Scalar>(10000.0);

        const Strategy strategy = Strategy::SequentialFromL2Solution;

        LinearSolverParam param{delta, beta, withIntercept, strategy};

        auto solutions = LinearSolver{param}.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\njob done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solutions.front(), streamSize)
                  << "\n";
    }

    return 0;
}

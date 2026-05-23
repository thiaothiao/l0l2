#include <chrono>
#include <ios>
#include <iostream>

#include <string_view>

#include <../simu.hpp>
#include <l0l2/core.hpp>
#include <l0l2/version.hpp>

int main()
{
    using Scalar = double;

    using Strategy = l0l2::linearmodel::leastsquares::Strategy;
    using SimpleBranchAndBound =
        l0l2::linearmodel::leastsquares::SimpleBranchAndBound<Scalar>;
    using SimpleBranchAndBoundParam = SimpleBranchAndBound::BBParam;

    std::cout << "l0l2 library version " << l0l2::metadata::libVersion << "\n";

    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(100.0);

    const auto withIntercept = false;

    std::cout << std::boolalpha;
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

        SimpleBranchAndBound::RegressorParam regressorParam{
            delta,         beta,
            withIntercept, strategy,
            tolerance,     maximumNumberOfIterations,
            innerEpsilon,  innerMaximumNumberOfIterations};

        Scalar globalGapEpsilon = static_cast<Scalar>(1e-8);
        Scalar localGapEpsilon = static_cast<Scalar>(1e-8);
        unsigned int maximumNumberOfBranchAndBoundIterations = 1000000U;

        SimpleBranchAndBoundParam param{
            regressorParam, globalGapEpsilon, localGapEpsilon,
            maximumNumberOfBranchAndBoundIterations};

        auto branchAndBound = SimpleBranchAndBound{param};

        const auto solution = branchAndBound.fit(mat, vect);

        const auto stop = std::chrono::high_resolution_clock::now();

        const auto durationUs =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nbranch and bound job done in " << durationUs.count()
                  << " microseconds!\n\n";
        std::cout << "\nobjective value: "
                  << branchAndBound.objectiveValue(mat, vect, solution) << "\n";
        std::cout << "\nsolution\n"
                  << l0l2::linearmodel::toString(solution, streamSize) << "\n";
    }

    return 0;
}

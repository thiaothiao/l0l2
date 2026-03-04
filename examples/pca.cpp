#include <iostream>
#include <ios>
#include <chrono>

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
    using L0L2SPCA = l0l2::linearmodel::L0L2SPCA<Scalar>;
    using Param = L0L2SPCA::Param;
    using FullPathL0L2SPCA = l0l2::linearmodel::FullPathL0L2SPCA<Scalar>;
    using FullPathParam = FullPathL0L2SPCA::Param;

    constexpr std::string version = L0L2_MACRO_STRINGIFY(L0L2_VERSION);
    std::cout << "l0l2 library version " << version << "\n";

    std::cout << "Finding sparse principal components...\n\n";

    const auto CovarianceMatrix = l0l2::linearmodel::pitprops<Scalar>();

    const auto n = static_cast<Index>(CovarianceMatrix.cols());
    const auto m = static_cast<Index>(CovarianceMatrix.rows());

    const auto matrixIsCovariance = true;
    const Scalar beta = static_cast<Scalar>(0.01);
    const auto delta = static_cast<Scalar>(15);

    std::cout << std::boolalpha;
    std::cout << "\nMatrixIsCovariance: " << matrixIsCovariance;
    std::cout << "\nBeta: " << beta;
    std::cout << "\nDelta: " << delta;
    std::cout << "\nFullpath: " << (delta < static_cast<Scalar>(0)) << "\n";

    const auto nbComponents = static_cast<Index>(3);
    const auto nbJobs = 1U;
    const auto epsilon = static_cast<Scalar>(1e-5);
    const auto numberOfTrialsMax = 1000U;
    const auto regressorTolerance = static_cast<Scalar>(1e-6);
    const auto regressorMaximumNumberOfIterations = 100000U;
    const auto regressorInnerEpsilon = static_cast<Scalar>(1e-6);
    const auto intregressorInnerMaximumNumberOfIterations = 100000U;

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy regressorStrategy = Strategy::FromZeroSolution;

        Param param{
            delta,
            beta,
            nbComponents,
            regressorStrategy,
            regressorTolerance,
            regressorMaximumNumberOfIterations,
            regressorInnerEpsilon,
            intregressorInnerMaximumNumberOfIterations };

        auto components = L0L2SPCA{
            param,
            nbJobs,
            epsilon,
            numberOfTrialsMax}.run(CovarianceMatrix, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nFrom Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        std::cout << "Components\n" << components.transpose() << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy regressorStrategy = Strategy::FromL2Solution;

        Param param{
            delta,
            beta,
            nbComponents,
            regressorStrategy,
            regressorTolerance,
            regressorMaximumNumberOfIterations,
            regressorInnerEpsilon,
            intregressorInnerMaximumNumberOfIterations };

        auto components = L0L2SPCA{
            param,
            nbJobs,
            epsilon,
            numberOfTrialsMax }.run(CovarianceMatrix, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nFrom L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        std::cout << "Components\n" << components.transpose() << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy regressorStrategy = Strategy::FromBothSolutions;

        Param param{
            delta,
            beta,
            nbComponents,
            regressorStrategy,
            regressorTolerance,
            regressorMaximumNumberOfIterations,
            regressorInnerEpsilon,
            intregressorInnerMaximumNumberOfIterations };

        auto components = L0L2SPCA{
            param,
            nbJobs,
            epsilon,
            numberOfTrialsMax }.run(CovarianceMatrix, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nFrom Both JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        std::cout << "Components\n" << components.transpose() << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy regressorStrategy = Strategy::FromZeroSolution;

        FullPathParam param{
            delta,
            beta,
            nbComponents,
            regressorStrategy };

        auto components = FullPathL0L2SPCA{
            param,
            nbJobs,
            epsilon,
            2*numberOfTrialsMax }.run(CovarianceMatrix, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nOne solution From Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        std::cout << "Components\n" << components.transpose() << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy regressorStrategy = Strategy::FromL2Solution;

        FullPathParam param{
            delta,
            beta,
            nbComponents,
            regressorStrategy };

        auto components = FullPathL0L2SPCA{
            param,
            nbJobs,
            epsilon,
            numberOfTrialsMax}.run(CovarianceMatrix, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nOne solution From L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        std::cout << "Components\n" << components.transpose() << "\n";
    }

    {
        const auto start = std::chrono::high_resolution_clock::now();

        const Strategy regressorStrategy = Strategy::FromBothSolutions;

        FullPathParam param{
            delta,
            beta,
            nbComponents,
            regressorStrategy };

        auto components = FullPathL0L2SPCA{
            param,
            nbJobs,
            epsilon,
            numberOfTrialsMax}.run(CovarianceMatrix, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nOne solution From both JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        std::cout << "Components\n" << components.transpose() << "\n";
    }

    return 0;
}
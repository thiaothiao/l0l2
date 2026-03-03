#include <iostream>
#include <ios>
#include <chrono>

#include "l0l2/core.hpp"
#include "simu.hpp"

template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<float>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<float>>;
template<>
const float l0l2::Utils<float>::epsilon = 1e-6f;

template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<double>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<double>>;
template<>
const double l0l2::Utils<double>::epsilon = 1e-8;

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

    std::cout << "Finding path...\n\n";
    const auto matrixIsCovariance = false;
    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(0.5); //static_cast<Scalar>(2.5);//static_cast<Scalar>(0.5);// static_cast<Scalar>(0.1);

    std::cout << std::boolalpha;
    std::cout << "\nMatrixIsCovariance: " << matrixIsCovariance;
    std::cout << "\nBeta: " << beta;
    std::cout << "\nDelta: " << delta;
    std::cout << "\nFullpath: " << (delta < static_cast<Scalar>(0)) << "\n";

    const auto [Mat, Vect] = l0l2::linearmodel::simulatedData<Scalar>(matrixIsCovariance);

    //const auto [Mat, Vect] = l0l2::linearmodel::simulatedRandomData<Scalar>(matrixIsCovariance);

    const auto n = static_cast<Index>(Mat.cols());
    const auto m = static_cast<Index>(Mat.rows());

    const auto nbComponents = static_cast<Index>(2);
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
            numberOfTrialsMax}.run(Mat, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        //Eigen::Map<const l0l2::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nFrom Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        l0l2::linearmodel::saveMatrix<Scalar>(components, "SparsePCAs.csv");
        //std::cout << "Objective value " <<
        //    objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
        //std::cout << "\nSolution\n" << solution.toString() << "\n";
        //std::cout << "\nSolution\n" << componentsX << "\n";

        //return 0;
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
            numberOfTrialsMax }.run(Mat, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        //Eigen::Map<const l0l2::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nFrom L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        l0l2::linearmodel::saveMatrix<Scalar>(components, "L2SparsePCAs.csv");
        //std::cout << "Objective value " <<
        //    objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
        //std::cout << "\nSolution\n" << solution.toString() << "\n";
        //std::cout << "\nSolution\n" << componentsX << "\n";

        //return 0;
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
            numberOfTrialsMax }.run(Mat, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        //Eigen::Map<const l0l2::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nFrom Both JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        l0l2::linearmodel::saveMatrix<Scalar>(components, "L02SparsePCAs.csv");
        //std::cout << "Objective value " <<
        //    objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
        //std::cout << "\nSolution\n" << solution.toString() << "\n";
        //std::cout << "\nSolution\n" << componentsX << "\n";

        //return 0;
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
            2*numberOfTrialsMax }.run(Mat, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        //Eigen::Map<const l0l2::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nOne solution From Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        l0l2::linearmodel::saveMatrix<Scalar>(components, "FP0SparsePCAs.csv");
        //std::cout << "Objective value " <<
        //    objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
        //std::cout << "\nSolution\n" << solution.toString() << "\n";
        //std::cout << "\nSolution\n" << componentsX << "\n";

        //return 0;
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
            numberOfTrialsMax}.run(Mat, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        //Eigen::Map<const l0l2::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nOne solution From L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        l0l2::linearmodel::saveMatrix<Scalar>(components, "FP2SparsePCAs.csv");
        //std::cout << "Objective value " <<
        //    objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
        //std::cout << "\nSolution\n" << solution.toString() << "\n";
        //std::cout << "\nSolution\n" << componentsX << "\n";

        //return 0;
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
            numberOfTrialsMax}.run(Mat, matrixIsCovariance);

        const auto stop = std::chrono::high_resolution_clock::now();

        //Eigen::Map<const l0l2::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nOne solution From both JOB DONE in " << duration_us.count() << " microseconds!\n\n";

        l0l2::linearmodel::saveMatrix<Scalar>(components, "FP02SparsePCAs.csv");
        //std::cout << "Objective value " <<
        //    objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
        //std::cout << "\nSolution\n" << solution.toString() << "\n";
        //std::cout << "\nSolution\n" << componentsX << "\n";

        //return 0;
    }

    return 0;
}
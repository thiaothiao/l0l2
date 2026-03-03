#include <iostream>
#include <ios>
#include <chrono>

#include <string>

#include "leastsquares/utils.hpp"
#include "leastsquares/cycliccoordinatedescent/solver.hpp"
#include "leastsquares/fullpath/solver.hpp"
#include "examples/simu.hpp"

template class l0l2::linearmodel::leastsquares::CyclicCoordinateDescent<
    l0l2::linearmodel::leastsquares::L0L2ModelImplementation<float>>;
template class l0l2::linearmodel::leastsquares::FullPathSolver<float>;
template<>
const float l0l2::Utils<float>::epsilon = 1e-6f;

template class l0l2::linearmodel::leastsquares::CyclicCoordinateDescent<
    l0l2::linearmodel::leastsquares::L0L2ModelImplementation<double>>;
template class l0l2::linearmodel::leastsquares::FullPathSolver<double>;
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



    std::cout << "Finding path...\n\n";
    const auto matrixIsCovariance = false;
    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(2.5);//static_cast<Scalar>(-1); //static_cast<Scalar>(0.5);// static_cast<Scalar>(0.1);

    const auto withIntercept = false;

    const auto fullpath = delta < static_cast<Scalar>(0);

    std::cout << std::boolalpha;
    std::cout << "\nMatrixIsCovariance: " << matrixIsCovariance;
    std::cout << "\nBeta: " << beta;
    std::cout << "\nDelta: " << delta;
    std::cout << "\nFullpath: " << fullpath << "\n";

    const std::streamsize streamSize = 6;

    const auto [Mat, Vect] = l0l2::linearmodel::simulatedData<Scalar>(matrixIsCovariance);

    //const auto [Mat, Vect] = l0l2::linearmodel::simulatedRandomData<Scalar>(matrixIsCovariance);

    const auto n = static_cast<Index>(Mat.cols());
    const auto m = static_cast<Index>(Mat.rows());

    const Scalar tolerance = static_cast<Scalar>(1e-6);
    const unsigned int maximumNumberOfIterations = 100000U;

    const Scalar innerEpsilon = static_cast<Scalar>(1e-6);
    const unsigned int innerMaximumNumberOfIterations = 100000U;

    if (fullpath)
    {
        std::string resultsfilename = R"(Results.txt)";

        const auto start = std::chrono::high_resolution_clock::now();

        auto results = FullPathSolver::fitAll(
            Mat, 
            Vect, 
            matrixIsCovariance, 
            beta,
            withIntercept,
            Strategy::FromL2Solution);

        const auto stop = std::chrono::high_resolution_clock::now();

        l0l2::linearmodel::save<Scalar>(beta, results, resultsfilename, streamSize);

        // Calculate the duration and cast to microseconds
        const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "\n\nJOB DONE in " << duration_us.count() << " microseconds!\n\n";
    }
    else
    {
        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromZeroSolution;

            L0L2RegressorParam param{ delta, beta, strategy, tolerance, maximumNumberOfIterations,
            innerEpsilon, innerMaximumNumberOfIterations };

            L0L2Regressor regressor{ param, withIntercept };

            const auto solution = regressor.fit(
                Mat,
                Vect,
                matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\nFrom Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";
            std::cout << "\nSolution\n" << l0l2::linearmodel::toString(solution, streamSize) << "\n";

            //return 0;
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromL2Solution;

            L0L2RegressorParam param{ delta, beta, strategy, tolerance, maximumNumberOfIterations,
            innerEpsilon, innerMaximumNumberOfIterations };

            L0L2Regressor regressor{ param, withIntercept };

            const auto solution = regressor.fit(
                Mat,
                Vect,
                matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\nFrom L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";
            std::cout << "\nSolution\n" << l0l2::linearmodel::toString(solution, streamSize) << "\n";

            //return 0;
        }

        {
            const auto start = std::chrono::high_resolution_clock::now();

            const Strategy strategy = Strategy::FromBothSolutions;

            L0L2RegressorParam param{ delta, beta, strategy, tolerance, maximumNumberOfIterations,
            innerEpsilon, innerMaximumNumberOfIterations };

            L0L2Regressor regressor{ param, withIntercept };

            const auto solution = regressor.fit(
                Mat,
                Vect,
                matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            // Calculate the duration and cast to microseconds
            const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\nFrom Both JOB DONE in " << duration_us.count() << " microseconds!\n\n";
            std::cout << "\nSolution\n" << l0l2::linearmodel::toString(solution, streamSize) << "\n";

            //return 0;
        }

        {
            //std::string OneSolutionfilename = R"(OneSolution.txt)";

            const auto start = std::chrono::high_resolution_clock::now();

            FullPathSolverParam param{ delta, beta, Strategy::FromZeroSolution };

            FullPathSolver regressor{ param , withIntercept };

            auto solution = regressor.fit(Mat, Vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            //l0l2::linearmodel::save<Scalar>(beta, { solution }, OneSolutionfilename, streamSize);

            // Calculate the duration and cast to microseconds
            const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\n0 JOB DONE in " << duration_us.count() << " microseconds!\n\n";
            std::cout << "\nSolution\n" << l0l2::linearmodel::toString(solution, streamSize) << "\n";
            //return 0;
        }

        {
            //std::string OneSolutionfilename = R"(OneSolution2.txt)";

            const auto start = std::chrono::high_resolution_clock::now();

            FullPathSolverParam param{ delta, beta, Strategy::FromL2Solution };

            FullPathSolver regressor{ param , withIntercept };

            auto solution = regressor.fit(Mat, Vect, matrixIsCovariance);

            const auto stop = std::chrono::high_resolution_clock::now();

            //l0l2::linearmodel::save<Scalar>(beta, { solution }, OneSolutionfilename, streamSize);

            // Calculate the duration and cast to microseconds
            const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\nL2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";
            std::cout << "\nSolution\n" << l0l2::linearmodel::toString(solution, streamSize) << "\n";
            return 0;
        }
    }

    return 0;
}
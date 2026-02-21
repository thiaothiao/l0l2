#include <iostream>
#include <ios>
#include <chrono>
#include <random>
#include <list>
#include <fstream>
#include <string>
#include <utility>
#include <execution>
#include <iomanip>
#include <algorithm>
#include <concepts>


#include "SparsePCA.h"
#include "Utils.h"

template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<float>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<float>>;
const std::streamsize l0l2::linearmodel::Solution<float>::streamSize = 7;
const float l0l2::linearmodel::Utils<float>::epsilon = 1e-6f;
const std::streamsize l0l2::linearmodel::CDSolution<float>::streamSize = 7;

template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<double>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<double>>;
const std::streamsize l0l2::linearmodel::Solution<double>::streamSize = 9;
const double l0l2::linearmodel::Utils<double>::epsilon = 1e-8;
const std::streamsize l0l2::linearmodel::CDSolution<double>::streamSize = 9;

namespace
{

    template<std::floating_point Scalar>
    static void save(Scalar beta,
        const std::list<l0l2::linearmodel::Solution<Scalar>>& results,
        const std::string& filename)
    {
        std::ofstream file;
        file.open(filename);

        for (auto it = results.begin(); it != results.end(); ++it)
        {
            const auto& result = *it;

            file << result.toString();

            if (result.isValidFor(beta))
            {
                file << " OK\n";
            }
            else
            {
                file << " NOK! Failed.\n";
            }
        }

        file.close();
    }

    template<std::floating_point Scalar>
    static void saveMatrix(const typename l0l2::linearmodel::Matrix<Scalar>& mat,
        const std::string& filename)
    {
        using Index = l0l2::linearmodel::Index;

        const auto n = static_cast<Index>(mat.cols());
        const auto m = static_cast<Index>(mat.rows());

        std::ofstream file;
        file.open(filename);

        file << std::fixed << std::setprecision(10);

        using Index = typename l0l2::linearmodel::Index;

        for (Index i = 0; i < m; ++i)
        {
            for (Index j = 0; j < n - 1; ++j)
            {
                file << mat.coeff(i, j) << ";";
            }

            file << mat.coeff(i, n - 1) << "\n";
        }

        file.close();
    }

    template<std::floating_point Scalar>
    static void saveVector(const typename l0l2::linearmodel::Vector<Scalar>& vect,
        const std::string& filename)
    {
        std::ofstream file;
        file.open(filename);

        file << std::fixed << std::setprecision(10);

        using Index = typename l0l2::linearmodel::Index;

        const auto n = static_cast<Index>(vect.size());

        for (Index j = 0; j < n; ++j)
        {
            file << vect[j] << "\n";
        }

        //file << vect[n-1] << "\n";

        file.close();
    }

    template<std::floating_point Scalar>
    static auto simulatedData()
    {
        using Matrix = l0l2::linearmodel::Matrix<Scalar>;
        using Vector = l0l2::linearmodel::Vector<Scalar>;
        using Index = l0l2::linearmodel::Index;

        Matrix AData(8, 20);
        AData <<
            -0.20123884, -0.00541008, -0.05698505, 0.33191139, 0.08332175, -0.47482827,
            0.20920801, -0.23443386, -0.23639715, -0.34962213, 0.1838184, 0.31660181,
            -0.16392842, 0.39081651, -0.30187818, -0.46938336, 0.37761492, 0.22743553,
            0.04088092, -0.36854184, -0.08633262, 0.27872878, 0.08390135, -0.31736857,
            0.32608223, -0.39459819, -0.21642333, -0.43443674, -0.4435558, 0.26545584,
            -0.48821196, 0.11194336, -0.16811773, 0.05964839, -0.16450036, -0.08881745,
            -0.42314449, 0.35304302, -0.06001255, -0.37804586, 0.23173463, -0.36121753,
            0.26688004, 0.33198977, -0.19022194, 0.09758228, 0.37239248, 0.48302084,
            -0.03259671, 0.37574452, -0.2039313, -0.36870897, 0.34281796, 0.15903628,
            0.09543961, -0.06364632, -0.14374968, 0.0871309, -0.35052866, -0.3287614,
            -0.10283548, 0.13795155, -0.12748006, -0.49759322, 0.04881638, -0.37302816,
            -0.42020732, -0.26496142, 0.15996492, -0.28504682, -0.29695338, -0.1171349,
            -0.2751272, -0.38535118, 0.08301705, -0.26735896, 0.0565182, 0.02381086,
            -0.14904177, 0.20533162, 0.32070374, -0.365437, 0.10472614, -0.21685171,
            0.0064078, -0.46153957, -0.1675559, -0.49972746, 0.16465545, -0.12255934,
            0.24758196, -0.35942423, -0.11195055, 0.36649162, 0.45685333, -0.01344633,
            -0.39406693, -0.15996271, 0.33557081, -0.03455839, -0.40714389, 0.35469413,
            -0.16274181, 0.10614675, -0.38733542, 0.04392046, 0.22079861, 0.36581534,
            -0.35329887, -0.39661893, -0.00944531, -0.47082043, 0.09486097, -0.29415452,
            -0.29719704, 0.34412587, -0.43843362, -0.00076932, -0.25153303, -0.17285597,
            -0.4970839, -0.20455787, 0.48688912, -0.26166728, 0.23339647, 0.46219689,
            0.31545579, 0.19422698, -0.2893039, 0.01784408, 0.44967002, 0.19597387,
            0.40466475, 0.14789242, -0.15469787, -0.43450215, -0.27937216, -0.36262673,
            0.2442959, -0.15330935, 0.07766426, -0.33164707, -0.46202829, 0.3171767,
            0.31246483, -0.31156433, -0.1269376, 0.24419212, -0.31375223, -0.0862022,
            -0.26899469, 0.15362293, 0.26437473, 0.33487916, -0.28778714, 0.46196657,
            -0.23885965, 0.36377019, -0.37312558, -0.12909779;

        //ContiguousDataContainer bData = { -0.30532599, -0.20403296, -0.88080627, -0.79558313,
        //    -0.11683895, -0.71092457, -0.05647609, -0.09459329 };

        Vector alphaData(20);
        alphaData <<
            0.0000000000, 0.1271706223, -0.6133915186, 0.0000000000, 0.0000000000, -0.0402539149,
            0.0000000000, 0.7919247746, 0.6456801295, 0.4932096303, 0.0000000000, 0.0000000000,
            0.4210028350, 0.0270699188, -0.3920102417, -0.9700308442, -0.8171941042, -0.2710959315,
            -0.7053743005, -0.6682027578;

        return std::pair<Matrix, Vector>
            (std::move(AData), std::move(alphaData));
    }

    template<std::floating_point ScalarType>
    static auto run(const l0l2::linearmodel::Matrix<ScalarType>& AData,//AData[j*numberOfRows + i]
        const l0l2::linearmodel::Vector<ScalarType>& alpha,
        bool matrixIsCovariance,
        ScalarType beta,
        ScalarType delta)
    {
        using Scalar = ScalarType;
        using Matrix = l0l2::linearmodel::Matrix<Scalar>;
        using Vector = l0l2::linearmodel::Vector<Scalar>;
        using Index = l0l2::linearmodel::Index;
        using Utils = l0l2::linearmodel::Utils<Scalar>;
        using Strategy = l0l2::linearmodel::Strategy;
        using L0L2SPCA = l0l2::linearmodel::L0L2SPCA<Scalar>;
        using Param = L0L2SPCA::Param;
        using FullPathL0L2SPCA = l0l2::linearmodel::FullPathL0L2SPCA<Scalar>;
        using FullPathParam = FullPathL0L2SPCA::Param;

        const auto n = static_cast<Index>(AData.cols());
        const auto m = static_cast<Index>(AData.rows());

        //l0l2::FullPathSolver<Scalar>::ContiguousDataContainer b = A * alpha;
        const Vector b = AData * alpha;

        const Vector Vect = matrixIsCovariance ? alpha : b;

        Matrix MatData = matrixIsCovariance
            ? static_cast<Matrix>(AData.transpose() * AData)
            : AData;

        const auto fullpath = delta < static_cast<Scalar>(0);

        const auto nbComponents = static_cast<Index>(2);
        const unsigned int nbJobs = 1U;
        const Scalar epsilon = static_cast<Scalar>(1e-5);
        const unsigned int numberOfTrialsMax = 1000U;
        const Scalar regressorTolerance = static_cast<Scalar>(1e-6);
        const unsigned int regressorMaximumNumberOfIterations = 100000U;

        if (fullpath)
        {
            //std::string resultsfilename = R"(Results.txt)";

            //const auto start = std::chrono::high_resolution_clock::now();

            //auto results = FullPathSolver::fitAll(MatData, m, n, Vect, matrixIsCovariance, beta,
            //    Strategy::FromBothSolutions);

            //const auto stop = std::chrono::high_resolution_clock::now();

            //save<Scalar>(beta, results, resultsfilename);

            //// Calculate the duration and cast to microseconds
            //const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            //std::cout << "\n\nJOB DONE in " << duration_us.count() << " microseconds!\n\n";
        }
        else
        {
            {
                const auto start = std::chrono::high_resolution_clock::now();

                const Strategy regressorStrategy = Strategy::FromZeroSolution;

                Param param{
                    delta,
                    beta,
                    nbComponents,
                    regressorStrategy,
                    regressorTolerance,
                    regressorMaximumNumberOfIterations };

                L0L2SPCA regressor{
                    param,
                    nbJobs,
                    epsilon,
                    numberOfTrialsMax };

                auto components = regressor.run(MatData, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                //Eigen::Map<const l0l2::linearmodel::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nFrom Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

                saveMatrix<Scalar>(components, "SparsePCAs.csv");
                //std::cout << "Objective value " <<
                //    Utils::objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
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
                    regressorMaximumNumberOfIterations };

                L0L2SPCA regressor{
                    param,
                    nbJobs,
                    epsilon,
                    numberOfTrialsMax };

                auto components = regressor.run(MatData, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                //Eigen::Map<const l0l2::linearmodel::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nFrom Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

                saveMatrix<Scalar>(components, "L2SparsePCAs.csv");
                //std::cout << "Objective value " <<
                //    Utils::objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
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
                    regressorMaximumNumberOfIterations };

                L0L2SPCA regressor{
                    param,
                    nbJobs,
                    epsilon,
                    numberOfTrialsMax };

                auto components = regressor.run(MatData, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                //Eigen::Map<const l0l2::linearmodel::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nFrom Both JOB DONE in " << duration_us.count() << " microseconds!\n\n";

                saveMatrix<Scalar>(components, "L02SparsePCAs.csv");
                //std::cout << "Objective value " <<
                //    Utils::objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
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

                FullPathL0L2SPCA regressor{
                    param,
                    nbJobs,
                    epsilon,
                    numberOfTrialsMax };

                auto components = regressor.run(MatData, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                //Eigen::Map<const l0l2::linearmodel::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nOne solution From Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";

                saveMatrix<Scalar>(components, "FP0SparsePCAs.csv");
                //std::cout << "Objective value " <<
                //    Utils::objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
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

                FullPathL0L2SPCA regressor{
                    param,
                    nbJobs,
                    epsilon,
                    numberOfTrialsMax };

                auto components = regressor.run(MatData, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                //Eigen::Map<const l0l2::linearmodel::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nOne solution From L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";

                saveMatrix<Scalar>(components, "FP2SparsePCAs.csv");
                //std::cout << "Objective value " <<
                //    Utils::objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
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

                FullPathL0L2SPCA regressor{
                    param,
                    nbJobs,
                    epsilon,
                    numberOfTrialsMax };

                auto components = regressor.run(MatData, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                //Eigen::Map<const l0l2::linearmodel::Matrix<Scalar>> componentsX(components.data(), n, param.nbComponents);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nOne solution From both JOB DONE in " << duration_us.count() << " microseconds!\n\n";

                saveMatrix<Scalar>(components, "FP02SparsePCAs.csv");
                //std::cout << "Objective value " <<
                //    Utils::objectiveValue(MatData, m, n, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
                //std::cout << "\nSolution\n" << solution.toString() << "\n";
                //std::cout << "\nSolution\n" << componentsX << "\n";

                //return 0;
            }
        }

        return 0;
    }
}

int main()
{
    using Scalar = double;

    using Matrix = l0l2::linearmodel::Matrix<Scalar>;
    using Vector = l0l2::linearmodel::Vector<Scalar>;
    using Index = l0l2::linearmodel::Index;

    const auto [A, alpha] = simulatedData<Scalar>();

    //Index m = 8;
    //Index n = 20;

    //std::default_random_engine engine;//engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
    //engine.seed(10);
    //std::uniform_real_distribution<double> uniform{ static_cast<double>(0), static_cast<double>(1) };

    ////AData[j*numberOfRows + i]    // row major is the default layout
    //typename l0l2::FullPathSolver<double>::ContiguousDataContainer ADouble(m * n);
    //std::generate(ADouble.begin(), ADouble.end(), [&]() { return uniform(engine); });

    //std::for_each(std::execution::par, ADouble.begin(), ADouble.end(), [](double& v) { v += static_cast<double>(-0.5); });

    //typename l0l2::FullPathSolver<double>::ContiguousDataContainer alphaDouble(n);
    //std::generate(alphaDouble.begin(), alphaDouble.end(), [&]() { return uniform(engine); });

    //alphaDouble[0] = static_cast<double>(0);
    //alphaDouble[4] = static_cast<double>(0);
    //alphaDouble[6] = static_cast<double>(0);
    //alphaDouble[3] = static_cast<double>(0);
    //alphaDouble[10] = static_cast<double>(0);
    //alphaDouble[11] = static_cast<double>(0);

    //const auto mn = m * n;
    //ContiguousDataContainer A(m * n);//AData[j*numberOfRows + i]    // row major is the default layout
    //for (Index j = 0; j < mn; ++j)
    //{
    //    A[j] = static_cast<Scalar>(ADouble[j]);
    //}

    //ContiguousDataContainer alpha(n);
    //for (Index j = 0; j < n; ++j)
    //{
    //    alpha[j] = static_cast<Scalar>(alphaDouble[j]);
    //}

    //saveMatrix<Scalar>(A, m, n, "refMatAlpha.csv");
    //saveVector<Scalar>(alpha, "refVectAlpha.csv");

    std::cout << "Finding path...\n\n";
    const auto matrixIsCovariance = false;
    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(0.5);//static_cast<Scalar>(-1);// static_cast<Scalar>(0.1);//

    const auto fullpath = true;

    std::cout << std::boolalpha;
    std::cout << "\nMatrixIsCovariance: " << matrixIsCovariance;
    std::cout << "\nBeta: " << beta;
    std::cout << "\nDelta: " << delta;
    std::cout << "\nFullpath: " << (delta < static_cast<Scalar>(0)) << "\n";

    return run<Scalar>(A,
        alpha,
        matrixIsCovariance,
        beta,
        delta);
}
#include <iostream>
#include <ios>
#include <chrono>
#include <random>
#include <list>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include <utility>
#include <concepts>

#include "Utils.h"

#include "FullPathSolver.h"
#include "CoordinateDescentSolver.h"

template class l0l2::linearmodel::CyclicalCoordinateDescent<l0l2::linearmodel::L0L2ModelImplementation<float>>;
template class l0l2::linearmodel::FullPathSolver<float>;
template<>
const std::streamsize l0l2::linearmodel::Solution<float>::streamSize = 7;
template<>
const float l0l2::linearmodel::Utils<float>::epsilon = 1e-6f;
template<>
const std::streamsize l0l2::linearmodel::CDSolution<float>::streamSize = 7;

template class l0l2::linearmodel::CyclicalCoordinateDescent<l0l2::linearmodel::L0L2ModelImplementation<double>>;
template class l0l2::linearmodel::FullPathSolver<double>;
template<>
const std::streamsize l0l2::linearmodel::Solution<double>::streamSize = 9;
template<>
const double l0l2::linearmodel::Utils<double>::epsilon = 1e-8;
template<>
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
            //for (Index j = 0; j < n - 1; ++j)
                for (Index j = 0; j < n ; ++j)
            {
                file << mat.coeff(i, j) << ",";
            }

            //file << mat.coeff(i, n - 1) << "\n";
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

        Matrix AData(8,20);
        AData <<
        -0.2012388400, -0.2363971500, 0.3776149200, 0.3260822300, -0.1681177300, 0.2317346300,
            -0.0325967100, -0.1437496800, 0.0488163800, -0.2751272000, 0.3207037400, 0.1646554500,
            -0.3940669300, -0.3873354200, 0.0948609700, -0.4970839000, -0.2893039000, -0.2793721600,
            0.3124648300, 0.2643747300, -0.0054100800, -0.3496221300, 0.2274355300, -0.3945981900,
            0.0596483900, -0.3612175300, 0.3757445200, 0.0871309000, -0.3730281600, -0.3853511800,
            -0.3654370000, -0.1225593400, -0.1599627100, 0.0439204600, -0.2941545200, -0.2045578700,
            0.0178440800, -0.3626267300, -0.3115643300, 0.3348791600, -0.0569850500, 0.1838184000,
            0.0408809200, -0.2164233300, -0.1645003600, 0.2668800400, -0.2039313000, -0.3505286600,
            -0.4202073200, 0.0830170500, 0.1047261400, 0.2475819600, 0.3355708100, 0.2207986100,
            -0.2971970400, 0.4868891200, 0.4496700200, 0.2442959000, -0.1269376000, -0.2877871400,
            0.3319113900, 0.3166018100, -0.3685418400, -0.4344367400, -0.0888174500, 0.3319897700,
            -0.3687089700, -0.3287614000, -0.2649614200, -0.2673589600, -0.2168517100, -0.3594242300,
            -0.0345583900, 0.3658153400, 0.3441258700, -0.2616672800, 0.1959738700, -0.1533093500,
            0.2441921200, 0.4619665700, 0.0833217500, -0.1639284200, -0.0863326200, -0.4435558000,
            -0.4231444900, -0.1902219400, 0.3428179600, -0.1028354800, 0.1599649200, 0.0565182000,
            0.0064078000, -0.1119505500, -0.4071438900, -0.3532988700, -0.4384336200, 0.2333964700,
            0.4046647500, 0.0776642600, -0.3137522300, -0.2388596500, -0.4748282700, 0.3908165100,
            0.2787287800, 0.2654558400, 0.3530430200, 0.0975822800, 0.1590362800, 0.1379515500,
            -0.2850468200, 0.0238108600, -0.4615395700, 0.3664916200, 0.3546941300, -0.3966189300,
            -0.0007693200, 0.4621968900, 0.1478924200, -0.3316470700, -0.0862022000, 0.3637701900,
            0.2092080100, -0.3018781800, 0.0839013500, -0.4882119600, -0.0600125500, 0.3723924800,
            0.0954396100, -0.1274800600, -0.2969533800, -0.1490417700, -0.1675559000, 0.4568533300,
            -0.1627418100, -0.0094453100, -0.2515330300, 0.3154557900, -0.1546978700, -0.4620282900,
            -0.2689946900, -0.3731255800, -0.2344338600, -0.4693833600, -0.3173685700, 0.1119433600,
            -0.3780458600, 0.4830208400, -0.0636463200, -0.4975932200, -0.1171349000, 0.2053316200,
            -0.4997274600, -0.0134463300, 0.1061467500, -0.4708204300, -0.1728559700, 0.1942269800,
            -0.4345021500, 0.3171767000, 0.1536229300, -0.1290977900;

        /*Matrix AData0(20, 8);
        AData0 <<
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
            -0.23885965, 0.36377019, -0.37312558, -0.12909779;*/

        //ContiguousDataContainer bData = { -0.30532599, -0.20403296, -0.88080627, -0.79558313,
        //    -0.11683895, -0.71092457, -0.05647609, -0.09459329 };

        Vector alphaData(20);
        alphaData <<
            0.0000000000, 0.1271706223, -0.6133915186, 0.0000000000, 0.0000000000, -0.0402539149,
            0.0000000000, 0.7919247746, 0.6456801295, 0.4932096303, 0.0000000000, 0.0000000000,
            0.4210028350, 0.0270699188, -0.3920102417, -0.9700308442, -0.8171941042, -0.2710959315,
            -0.7053743005, -0.6682027578;

        //Matrix AData = AData0.transpose();
        return std::pair<Matrix, Vector>
            (std::move(AData), std::move(alphaData));
    }

    template<std::floating_point ScalarType>
    static auto run(const l0l2::linearmodel::Matrix<ScalarType>& AData,//AData[j*numberOfRows + i]
        const l0l2::linearmodel::Vector<ScalarType>& alpha,
        bool matrixIsCovariance,
        ScalarType beta,
        ScalarType delta,
        bool withIntercept)
    {
        using Scalar = ScalarType;
        using Matrix = l0l2::linearmodel::Matrix<Scalar>;
        using Vector = l0l2::linearmodel::Vector<Scalar>;
        using Index = l0l2::linearmodel::Index;
        using FullPathSolver = l0l2::linearmodel::FullPathSolver<Scalar>;
        using FullPathSolverParam = FullPathSolver::Param;
        using L0L2Regressor = l0l2::linearmodel::L0L2Regressor<Scalar>;
        using L0L2RegressorParam = L0L2Regressor::Param;
        using Utils = l0l2::linearmodel::Utils<Scalar>;
        using Strategy = l0l2::linearmodel::Strategy;

        const auto n = static_cast<Index>(AData.cols());
        const auto m = static_cast<Index>(AData.rows());

        const Vector b = AData * alpha;

        const Vector Vect = matrixIsCovariance ? alpha : b;

        Matrix MatData = matrixIsCovariance
            ? static_cast<Matrix>(AData.transpose() * AData)
            : AData;

        const auto fullpath = delta < static_cast<Scalar>(0);

        const Scalar tolerance = static_cast<Scalar>(1e-6);
        const unsigned int maximumNumberOfIterations = 100000U;

        const Scalar innerEpsilon = static_cast<Scalar>(1e-6);
        const unsigned int innerMaximumNumberOfIterations = 100000U;

        if (fullpath)
        {
            std::string resultsfilename = R"(Results.txt)";

            const auto start = std::chrono::high_resolution_clock::now();

            auto results = FullPathSolver::fitAll(MatData, Vect, matrixIsCovariance, beta,
                withIntercept,
                Strategy::FromBothSolutions);

            const auto stop = std::chrono::high_resolution_clock::now();

            save<Scalar>(beta, results, resultsfilename);

            // Calculate the duration and cast to microseconds
            const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            std::cout << "\n\nJOB DONE in " << duration_us.count() << " microseconds!\n\n";
        }
        else
        {
            //{
            //    const l0l2::L2Regressor<Scalar> ridge(beta);

            //    const auto start = std::chrono::high_resolution_clock::now();
            //    auto sol = ridge.fit(MatData, m, n, Vect, matrixIsCovariance);
            //    const auto stop = std::chrono::high_resolution_clock::now();

            //    // Calculate the duration and cast to microseconds
            //    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            //    std::cout << "\n\nJOB DONE in " << duration_us.count() << " microseconds!\n\n";

            //    //std::cout << "\nSolution\n" << solution.toString() << "\n";

            //    return 0;
            //}

            {
                const auto start = std::chrono::high_resolution_clock::now();

                const Strategy strategy = Strategy::FromZeroSolution;

                L0L2RegressorParam param{ delta, beta, strategy, tolerance, maximumNumberOfIterations,
                innerEpsilon, innerMaximumNumberOfIterations };

                L0L2Regressor regressor{ param, withIntercept };

                const auto solution = regressor.fit(
                    MatData,
                    Vect,
                    matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nFrom Zero JOB DONE in " << duration_us.count() << " microseconds!\n\n";
                std::cout << "Objective value " <<
                    Utils::objectiveValue(MatData, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
                std::cout << "\nSolution\n" << solution.toString() << "\n";

                //return 0;
            }

            {
                const auto start = std::chrono::high_resolution_clock::now();

                const Strategy strategy = Strategy::FromL2Solution;

                L0L2RegressorParam param{ delta, beta, strategy, tolerance, maximumNumberOfIterations,
                innerEpsilon, innerMaximumNumberOfIterations };

                L0L2Regressor regressor{ param, withIntercept };

                const auto solution = regressor.fit(
                    MatData,
                    Vect,
                    matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nFrom L2 JOB DONE in " << duration_us.count() << " microseconds!\n\n";
                std::cout << "Objective value " <<
                    Utils::objectiveValue(MatData, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
                std::cout << "\nSolution\n" << solution.toString() << "\n";

                //return 0;
            }

            {
                const auto start = std::chrono::high_resolution_clock::now();

                const Strategy strategy = Strategy::FromBothSolutions;

                L0L2RegressorParam param{ delta, beta, strategy, tolerance, maximumNumberOfIterations,
                innerEpsilon, innerMaximumNumberOfIterations };

                L0L2Regressor regressor{ param, withIntercept };

                const auto solution = regressor.fit(
                    MatData,
                    Vect,
                    matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nFrom Both JOB DONE in " << duration_us.count() << " microseconds!\n\n";
                std::cout << "Objective value " <<
                    Utils::objectiveValue(MatData, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
                std::cout << "\nSolution\n" << solution.toString() << "\n";

                //return 0;
            }

            {
                std::string OneSolutionfilename = R"(OneSolution.txt)";

                const auto start = std::chrono::high_resolution_clock::now();

                FullPathSolverParam param{ delta, beta, Strategy::FromZeroSolution };

                FullPathSolver regressor{ param , withIntercept };

                auto solution = regressor.fit(MatData, Vect, matrixIsCovariance);

                const auto stop = std::chrono::high_resolution_clock::now();

                save<Scalar>(beta, { solution }, OneSolutionfilename);

                // Calculate the duration and cast to microseconds
                const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                std::cout << "\n\nJOB DONE in " << duration_us.count() << " microseconds!\n\n";
                std::cout << "Objective value " <<
                    Utils::objectiveValue(MatData, Vect, matrixIsCovariance, beta, delta, solution.x) << "\n";
                return 0;
            }
        }

        return 0;
    }
}

int main()
{
    using Scalar = double;

    using Vector = l0l2::linearmodel::Vector<Scalar>;
    using Matrix = l0l2::linearmodel::Matrix<Scalar>;
    using Index = l0l2::linearmodel::Index;

    const auto [A, alpha] = simulatedData<Scalar>();

    //Index m = 200; //50; //8;
    //Index n = 500;//150; //20;

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

    saveMatrix<Scalar>(A, "MatToto.txt");
    saveVector<Scalar>(alpha, "VectToto.txt");

    std::cout << "Finding path...\n\n";
    const auto matrixIsCovariance = false;
    const Scalar beta = static_cast<Scalar>(0.1);
    const auto delta = static_cast<Scalar>(0.01); //static_cast<Scalar>(2.5);//static_cast<Scalar>(0.5);// static_cast<Scalar>(0.1);

    const auto withIntercept = true;

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
        delta,
        withIntercept);
}
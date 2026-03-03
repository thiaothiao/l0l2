#ifndef L0L2_SIMU_HPP
#define L0L2_SIMU_HPP

#include <ios>
#include <iomanip>
#include <random>
#include <string>
#include <fstream>
#include <sstream>
#include <concepts>
#include <utility>
#include <algorithm>
#include <list>

#include "leastsquares/utils.hpp"

namespace l0l2
{
	namespace linearmodel
	{
        template<std::floating_point Scalar>
        static auto simulatedData(bool matrixIsCovariance = false)
        {
            using Matrix = l0l2::Matrix<Scalar>;
            using Vector = l0l2::Vector<Scalar>;
            using Index = l0l2::Index;

            Matrix AData(8, 20);
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

            Vector alphaData(20);
            alphaData <<
                0.0000000000, 0.1271706223, -0.6133915186, 0.0000000000, 0.0000000000, -0.0402539149,
                0.0000000000, 0.7919247746, 0.6456801295, 0.4932096303, 0.0000000000, 0.0000000000,
                0.4210028350, 0.0270699188, -0.3920102417, -0.9700308442, -0.8171941042, -0.2710959315,
                -0.7053743005, -0.6682027578;


            const auto n = static_cast<Index>(AData.cols());
            const auto m = static_cast<Index>(AData.rows());

            const Vector b = AData * alphaData;

            const Vector Vect = matrixIsCovariance ? alphaData : b;

            Matrix MatData = matrixIsCovariance
                ? static_cast<Matrix>(AData.transpose() * AData)
                : AData;

            return std::pair<Matrix, Vector>
                (std::move(MatData), std::move(Vect));
        }

        template<std::floating_point Scalar>
        static auto simulatedRandomData(bool matrixIsCovariance = false)
        {
            using Matrix = l0l2::Matrix<Scalar>;
            using Vector = l0l2::Vector<Scalar>;
            using Index = l0l2::Index;

            Index m = 8;
            Index n = 20;

            std::default_random_engine engine;//engine.seed(std::chrono::system_clock::now().time_since_epoch().count());
            engine.seed(10);
            std::uniform_real_distribution<double> uniform{ static_cast<double>(0), static_cast<double>(1) };

            // Generate a random matrix using NullaryExpr
            // The lambda function captures the generator and distribution by reference [&]
            Matrix AData = Eigen::MatrixXd::NullaryExpr(m, n, [&]() { return uniform(engine); }).template cast<Scalar>();
            AData.array() -= static_cast<Scalar>(0.5);

            Vector alphaData = Eigen::VectorXd::NullaryExpr(n, [&]() { return uniform(engine); }).template cast<Scalar>();;

            alphaData[0] = static_cast<Scalar>(0);
            alphaData[4] = static_cast<Scalar>(0);
            alphaData[6] = static_cast<Scalar>(0);
            alphaData[3] = static_cast<Scalar>(0);
            alphaData[10] = static_cast<Scalar>(0);
            alphaData[11] = static_cast<Scalar>(0);

            const Vector b = AData * alphaData;

            const Vector Vect = matrixIsCovariance ? alphaData : b;

            Matrix MatData = matrixIsCovariance
                ? static_cast<Matrix>(AData.transpose() * AData)
                : AData;

            return std::pair<Matrix, Vector>
                (std::move(MatData), std::move(Vect));
        }

        template<std::floating_point ScalarType>
        std::string toString(
            const l0l2::linearmodel::leastsquares::Solution<ScalarType>& solution,
            std::streamsize streamSize)
        {
            using Scalar = ScalarType;
            using Utils = Utils<Scalar>;

            std::stringstream out;

            out << std::fixed << std::setprecision(streamSize);

            out << solution.delta << ": [";

            out << Utils::print(streamSize, solution.x);

            out << "\t\t";

            out << Utils::print(streamSize, solution.grad);

            out << "\t\t" << solution.intercept;

            out << "]";

            return out.str();
        }

        template<std::floating_point ScalarType>
        std::string toString(
            const l0l2::linearmodel::leastsquares::CDSolution<ScalarType>& solution,
            std::streamsize streamSize)
        {
            using Scalar = ScalarType;
            using Utils = Utils<Scalar>;

            std::stringstream out;

            out << std::fixed << std::setprecision(streamSize);
            out << "NumberOfIterations: " << solution.numberOfIterations << "\n";
            out << "GlobalChange: " << solution.globalChange << "\n";

            out << Utils::print(streamSize, solution.x);

            out << "\t\t" << solution.intercept;

            return out.str();
        }

        template<std::floating_point Scalar>
        static void save(Scalar beta,
            const std::list<l0l2::linearmodel::leastsquares::Solution<Scalar>>& results,
            const std::string& filename,
            std::streamsize streamSize)
        {
            std::ofstream file;
            file.open(filename);

            for (auto it = results.begin(); it != results.end(); ++it)
            {
                const auto& result = *it;

                file << toString(result, streamSize);

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
        static void saveMatrix(const typename l0l2::Matrix<Scalar>& mat,
            const std::string& filename)
        {
            using Index = l0l2::Index;

            const auto n = static_cast<Index>(mat.cols());
            const auto m = static_cast<Index>(mat.rows());

            std::ofstream file;
            file.open(filename);

            file << std::fixed << std::setprecision(10);

            using Index = typename l0l2::Index;

            for (Index i = 0; i < m; ++i)
            {
                for (Index j = 0; j < n - 1; ++j)
                //for (Index j = 0; j < n; ++j)
                {
                    file << mat.coeff(i, j) << ",";
                }

                file << mat.coeff(i, n - 1) << "\n";
            }

            file.close();
        }

        template<std::floating_point Scalar>
        static void saveVector(const typename l0l2::Vector<Scalar>& vect,
            const std::string& filename)
        {
            std::ofstream file;
            file.open(filename);

            file << std::fixed << std::setprecision(10);

            using Index = typename l0l2::Index;

            const auto n = static_cast<Index>(vect.size());

            for (Index j = 0; j < n; ++j)
            {
                file << vect[j] << "\n";
            }

            //file << vect[n-1] << "\n";

            file.close();
        }
	}
}
#endif //L0L2_SIMU_HPP

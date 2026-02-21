#ifndef L0L2_UTILS_H
#define L0L2_UTILS_H

#include <ios>
#include <string>
#include <vector>
#include <limits>
#include <cstdint>
#include <cmath>
#include <execution>
#include <functional>
#include <sstream>
#include <iomanip>
#include <concepts>



namespace l0l2
{
	namespace linearmodel
	{
		using Index = int;

		template<std::floating_point ScalarType>
		using ContiguousDataContainer = std::vector<ScalarType>;

		enum class CDStatus : std::uint8_t
		{
			Converged = 0U,
			LimitReached,
			Unknown
		};

		enum class Strategy : std::uint8_t
		{
			FromZeroSolution = 0U,
			FromL2Solution,
			FromBothSolutions
		};

		template<std::floating_point ScalarType>
		class Utils final
		{
		public:
			using Scalar = ScalarType;

			const static Scalar epsilon;

			static Scalar sign(Scalar value);

			static Scalar objectiveValue(
				const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
				Index numberOfRows, Index numberOfColumns,
				const ContiguousDataContainer<Scalar>& vectData,
				bool matrixIsCovariance,
				Scalar beta,
				Scalar delta,
				const ContiguousDataContainer<Scalar>& x);

			static std::string print(std::streamsize size,
				const ContiguousDataContainer<Scalar>& other);
		};

		template<std::floating_point ScalarType>
		struct Solution
		{
			using Scalar = ScalarType;

			const static std::streamsize streamSize;

			Solution(Scalar deltaInput = std::numeric_limits<Scalar>::max(),
				const ContiguousDataContainer<Scalar>& xInput = {},
				const ContiguousDataContainer<Scalar>& gradInput = {}) :
				delta{ deltaInput },
				x{ xInput },
				grad{ gradInput }
			{
			}

			Solution(Index n) :
				delta{ std::numeric_limits<Scalar>::max() },
				x{ ContiguousDataContainer<Scalar>(n, static_cast<Scalar>(0)) },
				grad{ ContiguousDataContainer<Scalar>(n, static_cast<Scalar>(0)) }
			{
			}

			Solution(const Solution&) = default;
			Solution& operator=(const Solution&) = default;

			Solution(Solution&&) = default;
			Solution& operator=(Solution&&) = default;

			bool isValidFor(Scalar beta) const;

			std::string toString() const;

			Scalar delta;
			ContiguousDataContainer<Scalar> x;
			ContiguousDataContainer<Scalar> grad;
		};

		template<std::floating_point ScalarType>
		struct CDSolution
		{
			using Scalar = ScalarType;

			const static std::streamsize streamSize;

			CDSolution(Index n = 0) :
				numberOfIterations{ 0 },
				globalChange{ static_cast<Scalar>(0) },
				dualityGap{ static_cast<Scalar>(0) },
				status{ CDStatus::Unknown },
				x{ ContiguousDataContainer<Scalar>(n, static_cast<Scalar>(0)) }
			{
			}

			CDSolution(const CDSolution&) = default;
			CDSolution& operator=(const CDSolution&) = default;

			CDSolution(CDSolution&&) = default;
			CDSolution& operator=(CDSolution&&) = default;

			std::string toString() const;

			unsigned int numberOfIterations;
			Scalar globalChange;
			Scalar dualityGap;
			CDStatus status;
			ContiguousDataContainer<Scalar> x;
		};

		template<std::floating_point ScalarType>
		inline ScalarType squaredNorm(const ContiguousDataContainer<ScalarType>& v)
		{
			return std::transform_reduce(std::execution::par,
				v.cbegin(), v.cend(), static_cast<ScalarType>(0), std::plus{},
				[](auto val) { return val * val; });
		}

		template<std::floating_point ScalarType>
		inline ScalarType norm(const ContiguousDataContainer<ScalarType>& v)
		{
			return std::sqrt(squaredNorm(v));
		}

		template<std::floating_point ScalarType>
		ContiguousDataContainer<ScalarType>
			opposite(const ContiguousDataContainer<ScalarType>& aVec)
		{
			auto result = aVec;

			std::for_each(std::execution::par, result.begin(), result.end(),
				[](ScalarType& v) { v = -v; });

			return result;
		}

		template<std::floating_point ScalarType>
		ScalarType componentwiseAbsMinCoeff(const ContiguousDataContainer<ScalarType>& v)
		{
			// TODO optimize
			const auto min = std::min_element(std::execution::par, // parallel not implemented in c++20
				v.begin(), v.end(),
				[](auto a, auto b) { return std::abs(a) < std::abs(b); });

			if (min != v.end())
			{
				return std::abs(*min);
			}

			// empty vector
			return std::numeric_limits<ScalarType>::max();
		}

		template<std::floating_point ScalarType>
		ScalarType lpNormInfinity(const ContiguousDataContainer<ScalarType>& v)// use anonimuous namespace
		{//TODO optimize
			const auto max = std::max_element(std::execution::par, // parallel not implemented in c++20
				v.begin(), v.end(),
				[](auto a, auto b) { return std::abs(a) < std::abs(b); });

			if (max != v.end())
			{
				return std::abs(*max);
			}

			// empty vector
			return std::numeric_limits<ScalarType>::min();
		}

		template<std::floating_point ScalarType>
		inline Utils<ScalarType>::Scalar Utils<ScalarType>::sign(Scalar value)
		{
			return std::signbit(value) ? static_cast<Scalar>(-1) : static_cast<Scalar>(1);
		}

		template<std::floating_point ScalarType>
		Utils<ScalarType>::Scalar Utils<ScalarType>::objectiveValue(
			const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
			Index numberOfRows, Index numberOfColumns,
			const ContiguousDataContainer<Scalar>& vectData,
			bool matrixIsCovariance,
			Scalar beta,
			Scalar delta,
			const ContiguousDataContainer<Scalar>& x)
		{
			const auto n = numberOfColumns;
			const auto m = numberOfRows;

			const auto APtr = matData.data();
			const auto xPtr = x.data();

			auto bMinusAx = vectData;
			auto bMinusAxPtr = bMinusAx.data();
#pragma omp parallel for
			for (Index i = 0; i < m; ++i)
			{//Not auto vectorized
				for (Index j = 0; j < n; ++j)
				{//Not auto vectorized
					bMinusAxPtr[i] -= APtr[j * m + i] * xPtr[j];
				}
			}

			const auto betaSquaredDelta = beta * delta * delta;
			auto objVal = squaredNorm(bMinusAx);
			for (Index j = 0; j < n; ++j)
			{
				if (std::abs(xPtr[j]) >= delta)
				{
					objVal += beta * xPtr[j] * xPtr[j] + betaSquaredDelta;
				}
				else
				{
					objVal += static_cast<Scalar>(2) * delta * beta * std::abs(xPtr[j]);
				}
			}

			return objVal;
		}

		template<std::floating_point ScalarType>
		std::string Utils<ScalarType>::print(std::streamsize size,
			const ContiguousDataContainer<Scalar>& other)
		{
			std::stringstream out;

			out << std::fixed << std::setprecision(size);

			const auto n = static_cast<Index>(other.size());
			for (Index i = 0; i < n - 1; ++i)
			{//Not auto vectorized
				out << other[i] << "\t";
			}

			if (n - 1 > 0)
			{
				out << other[n - 1];
			}

			return out.str();
		}

		template<std::floating_point ScalarType>
		bool Solution<ScalarType>::isValidFor(Scalar beta) const
		{
			using Utils = Utils<Scalar>;

			const auto n = static_cast<Index>(x.size());

			const auto vPtr = grad.data();
			const auto xPtr = x.data();

			const auto deltaBeta = delta * beta;
			////////#pragma omp parallel for
			for (Index i = 0; i < n; ++i)
			{//Not auto vectorized
				if (std::abs(xPtr[i]) <= Utils::epsilon)
				{
					if (std::abs(vPtr[i]) - deltaBeta > Utils::epsilon)
					{
						return false;
					}
				}
				else if (std::abs(xPtr[i]) < delta)
				{
					const auto err = vPtr[i] - beta * xPtr[i] + deltaBeta * Utils::sign(xPtr[i]);
					if (std::abs(err) > Utils::epsilon)
					{
						return false;
					}
				}
				else //if (std::abs(xPtr[i]) >= delta)
				{
					if (std::abs(vPtr[i]) > Utils::epsilon)
					{
						return false;
					}
				}
			}

			return true;
		}

		template<std::floating_point ScalarType>
		std::string Solution<ScalarType>::toString() const
		{
			using Utils = Utils<Scalar>;

			std::stringstream out;

			out << std::fixed << std::setprecision(Solution::streamSize);

			out << delta << ": [";

			out << Utils::print(Solution::streamSize, x);

			out << "\t\t";

			out << Utils::print(Solution::streamSize, grad);

			out << "]";

			return out.str();
		}

		template<std::floating_point ScalarType>
		std::string CDSolution<ScalarType>::toString() const
		{
			using Utils = Utils<Scalar>;

			std::stringstream out;

			out << std::fixed << std::setprecision(CDSolution::streamSize);
			out << "NumberOfIterations: " << numberOfIterations << "\n";
			out << "GlobalChange: " << globalChange << "\n";

			out << Utils::print(CDSolution::streamSize, x);

			return out.str();
		}

		const std::streamsize Solution<float>::streamSize = 7;
		const float Utils<float>::epsilon = 1e-6f;
		const std::streamsize CDSolution<float>::streamSize = 7;

		const std::streamsize Solution<double>::streamSize = 9;
		const double Utils<double>::epsilon = 1e-8;
		const std::streamsize CDSolution<double>::streamSize = 9;
	}
}

#endif //L0L2_UTILS_H

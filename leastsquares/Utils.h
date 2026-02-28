#ifndef L0L2_UTILS_H
#define L0L2_UTILS_H

#include <ios>
#include <string>
#include <limits>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <concepts>

#include <Eigen/Dense>

namespace l0l2
{
	using Index = Eigen::Index;

	template<std::floating_point ScalarType>
	using Vector = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>;

	template<std::floating_point ScalarType>
	using Matrix = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

	template<std::floating_point ScalarType>
	using RMMatrix = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	template<std::floating_point ScalarType>
	class Utils final
	{
	public:
		using Scalar = ScalarType;

		const static Scalar epsilon;

		static Scalar sign(Scalar value);

		static std::string print(std::streamsize size,
			const Vector<Scalar>& other);
	};

	template<std::floating_point ScalarType>
	inline Utils<ScalarType>::Scalar Utils<ScalarType>::sign(Scalar value)
	{
		return std::signbit(value) ? static_cast<Scalar>(-1) : static_cast<Scalar>(1);
	}

	template<std::floating_point ScalarType>
	std::string Utils<ScalarType>::print(std::streamsize size,
		const Vector<Scalar>& other)
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

	namespace linearmodel
	{
		namespace leastsquares
		{
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
			struct Solution
			{
				using Scalar = ScalarType;

				const static std::streamsize streamSize;

				Solution(Scalar deltaInput = std::numeric_limits<Scalar>::max(),
					const Vector<Scalar>& xInput = {},
					const Vector<Scalar>& gradInput = {}) :
					delta{ deltaInput },
					x{ xInput },
					grad{ gradInput },
					intercept{ static_cast<Scalar>(0) }
				{
				}

				Solution(Index n) :
					delta{ std::numeric_limits<Scalar>::max() },
					x{ Vector<Scalar>::Zero(n) },
					grad{ Vector<Scalar>::Zero(n) },
					intercept{ static_cast<Scalar>(0) }
				{
				}

				Solution(const Solution&) = default;
				Solution& operator=(const Solution&) = default;

				Solution(Solution&&) = default;
				Solution& operator=(Solution&&) = default;

				bool isValidFor(Scalar beta) const;

				std::string toString() const;

				Scalar delta;
				Vector<Scalar> x;
				Vector<Scalar> grad;
				Scalar intercept;
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
					x{ Vector<Scalar>::Zero(n) },
					intercept{ static_cast<Scalar>(0) }
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
				Vector<Scalar> x;
				Scalar intercept;
			};			

			template<std::floating_point ScalarType>
			bool Solution<ScalarType>::isValidFor(Scalar beta) const
			{
				using Utils = Utils<Scalar>;

				const auto n = static_cast<Index>(x.size());

				const auto deltaBeta = delta * beta;

				for (Index i = 0; i < n; ++i)
				{//Not auto vectorized
					if (std::abs(x[i]) <= Utils::epsilon)
					{
						if (std::abs(grad[i]) - deltaBeta > Utils::epsilon)
						{
							return false;
						}
					}
					else if (std::abs(x[i]) < delta)
					{
						const auto err = grad[i] - beta * x[i] + deltaBeta * Utils::sign(x[i]);
						if (std::abs(err) > Utils::epsilon)
						{
							return false;
						}
					}
					else //if (std::abs(x[i]) >= delta)
					{
						if (std::abs(grad[i]) > Utils::epsilon)
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

				out << "\t\t" << intercept;

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

				out << "\t\t" << intercept;

				return out.str();
			}
		}
	}
}

#endif //L0L2_UTILS_H

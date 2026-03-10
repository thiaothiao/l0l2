#ifndef L0L2_UTILS_HPP
#define L0L2_UTILS_HPP

#include <ios>
#include <string>
#include <limits>
#include <cstdint>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <concepts>
#include <type_traits>

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

		static constexpr Scalar epsilon = 
			std::is_same_v<Scalar, float> ? static_cast<Scalar>(1e-6) : static_cast<Scalar>(1e-8);

		static Scalar sign(Scalar value);

		static Scalar solveMaxConcaveQP1D(Scalar a, Scalar b, Scalar c, Scalar s0, Scalar s1);

		static std::string print(std::streamsize size,
			const Vector<Scalar>& other);
	};

	template<std::floating_point ScalarType>
	inline Utils<ScalarType>::Scalar Utils<ScalarType>::sign(Scalar value)
	{
		return std::signbit(value) ? static_cast<Scalar>(-1) : static_cast<Scalar>(1);
	}

	template<std::floating_point ScalarType>
	Utils<ScalarType>::Scalar Utils<ScalarType>::solveMaxConcaveQP1D(
		Scalar a, Scalar b, Scalar c, Scalar s0, Scalar s1)
	{
		if (a == static_cast<Scalar>(0))
		{
			return std::max(b * s0, b * s1) + c;
		}

		auto sol = -static_cast<Scalar>(0.5) * b / a;
		if (sol < s0)
		{
			sol = s0;
		}
		else if (sol > s1)
		{
			sol = s1;
		}

		return  a * sol * sol + b * sol + c;
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

				Solution(Scalar deltaInput = std::numeric_limits<Scalar>::max(),
					const Vector<Scalar>& xInput = {},
					const Vector<Scalar>& gradInput = {}) :
					delta{ deltaInput },
					x{ xInput },
					intercept{ static_cast<Scalar>(0) },
					dualityGap{ static_cast<Scalar>(0) },
					grad{ gradInput }
				{
				}

				Solution(Vector<Scalar>&& xInput, Scalar interceptInput=static_cast<Scalar>(0)) :
					delta{ static_cast<Scalar>(0) },
					x{ std::move(xInput) },// TODO check move
					intercept{ interceptInput },
					dualityGap{ static_cast<Scalar>(0) }
				{
				}

				Solution(Index n) :
					delta{ std::numeric_limits<Scalar>::max() },
					x{ Vector<Scalar>::Zero(n) },
					intercept{ static_cast<Scalar>(0) },
					dualityGap{ static_cast<Scalar>(0) },
					grad{ Vector<Scalar>::Zero(n) }
				{
				}

				Solution(const Solution&) = default;
				Solution& operator=(const Solution&) = default;

				Solution(Solution&&) = default;
				Solution& operator=(Solution&&) = default;

				Scalar delta;
				Vector<Scalar> x;
				Scalar intercept;
				Scalar dualityGap;
				Vector<Scalar> grad;
			};
		}
	}
}

#endif //L0L2_UTILS_HPP

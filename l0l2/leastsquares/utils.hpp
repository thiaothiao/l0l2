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

				Scalar delta;
				Vector<Scalar> x;
				Vector<Scalar> grad;
				Scalar intercept;
			};

			template<std::floating_point ScalarType>
			struct CDSolution
			{
				using Scalar = ScalarType;

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

				unsigned int numberOfIterations;
				Scalar globalChange;
				Scalar dualityGap;
				CDStatus status;
				Vector<Scalar> x;
				Scalar intercept;
			};
		}
	}
}

#endif //L0L2_UTILS_HPP

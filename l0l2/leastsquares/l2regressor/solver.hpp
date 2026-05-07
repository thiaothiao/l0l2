#pragma once

#include <concepts>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
	namespace linearmodel
	{
        namespace leastsquares
        {
            template <class ModelImplementationType>
            concept NoInterceptDirectL2RegressorLike = requires(ModelImplementationType impl)
            {
                {
                    std::as_const(impl).fitNoIntercept(
                        Matrix<typename ModelImplementationType::Scalar>{},
                        Vector<typename ModelImplementationType::Scalar>{},
                        bool{},
                        CoordinateStates{})
                } ->std::convertible_to<Solution<typename ModelImplementationType::Scalar>>;
            };

            template <class ModelImplementationType>
            concept NoInterceptIterativeL2RegressorLike = requires(ModelImplementationType impl)
            {
                {
                    std::as_const(impl).fitNoIntercept(
                        Matrix<typename ModelImplementationType::Scalar>{},
                        Vector<typename ModelImplementationType::Scalar>{},
                        bool{},
                        typename ModelImplementationType::Scalar{},
                        0U,
                        Solution<typename ModelImplementationType::Scalar>{},
                        CoordinateStates{})
                } ->std::convertible_to<Solution<typename ModelImplementationType::Scalar>>;
            };

            template <NoInterceptDirectL2RegressorLike ModelImplementationType>
            class L2RegressorDirect final
            {
            public:
                using ModelImplementation = ModelImplementationType;
                using Scalar = typename ModelImplementation::Scalar;

                L2RegressorDirect(Scalar beta, bool hasIntercept) :
                    m_Beta{ beta },
                    m_HasIntercept{ hasIntercept }
                {
                }

                Solution<Scalar> fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const CoordinateStates& coordinateStates = {}) const;

            private:
                const Scalar m_Beta;
                const bool m_HasIntercept;
            };

            template <NoInterceptDirectL2RegressorLike ModelImplementationType>
            Solution<typename L2RegressorDirect<ModelImplementationType>::Scalar>
                L2RegressorDirect<ModelImplementationType>::fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const CoordinateStates& coordinateStates) const
            {
                const ModelImplementation modelImplementation{ m_Beta };

                if (m_HasIntercept && !matrixIsCovariance)
                {
                    auto solution = modelImplementation.fitNoIntercept(
                        matData.rowwise() - matData.colwise().mean(),
                        vectData.array() - vectData.mean(),
                        matrixIsCovariance,
                        coordinateStates);

                    solution.intercept = (vectData - matData * solution.x).mean();

                    return solution;
                }

                return modelImplementation.fitNoIntercept(matData, vectData, matrixIsCovariance, coordinateStates);
            }

            template <NoInterceptIterativeL2RegressorLike ModelImplementationType>
            class L2RegressorIterative final
            {
            public:
                using ModelImplementation = ModelImplementationType;
                using Scalar = typename ModelImplementation::Scalar;

                L2RegressorIterative(Scalar beta, bool hasIntercept) :
                    m_Beta{ beta },
                    m_HasIntercept{ hasIntercept }
                {
                }

                Solution<Scalar> fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar epsilon,
                    unsigned int maxNumberOfIterations,
                    const Solution<Scalar>& guess = {},
                    const CoordinateStates& coordinateStates = {}) const;

            private:
                const Scalar m_Beta;
                const bool m_HasIntercept;
            };

            template <NoInterceptIterativeL2RegressorLike ModelImplementationType>
            Solution<typename L2RegressorIterative<ModelImplementationType>::Scalar>
                L2RegressorIterative<ModelImplementationType>::fit(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar epsilon,
                    unsigned int maxNumberOfIterations,
                    const Solution<Scalar>& guess,
                    const CoordinateStates& coordinateStates) const
            {
                const ModelImplementation modelImplementation{ m_Beta };

                if (m_HasIntercept && !matrixIsCovariance)
                {
                    auto solution = modelImplementation.fitNoIntercept(
                        matData.rowwise() - matData.colwise().mean(),
                        vectData.array() - vectData.mean(),
                        matrixIsCovariance,
                        epsilon,
                        maxNumberOfIterations,
                        guess,
                        coordinateStates);

                    solution.intercept = (vectData - matData * solution.x).mean();

                    return solution;
                }

                return modelImplementation.fitNoIntercept(matData, vectData, matrixIsCovariance,
                    epsilon, maxNumberOfIterations, guess, coordinateStates);
            }

            template<std::floating_point ScalarType>
            class LDLTModelImplementation final
            {
            public:
                using Scalar = ScalarType;

                LDLTModelImplementation(Scalar beta) :
                    m_Beta{ beta }
                {
                }

                Solution<Scalar> fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const CoordinateStates& coordinateStates) const;

            private:
                const Scalar m_Beta;
            };

            template<std::floating_point ScalarType>
            Solution<typename LDLTModelImplementation<ScalarType>::Scalar>
                LDLTModelImplementation<ScalarType>::fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    [[maybe_unused]] const CoordinateStates& coordinateStates) const
            {
                using Vector = Vector<Scalar>;
                using Matrix = Matrix<Scalar>;

                Matrix ATA;
                if (matrixIsCovariance)
                {
                    ATA = matData;
                }
                else
                {
                    ATA = matData.transpose() * matData;
                }

                Vector ATb;// Avoiding ternary operator as suggested by lib eigen c++
                if (matrixIsCovariance)
                {
                    ATb = matData * vectData;
                }
                else
                {
                    ATb = matData.transpose() * vectData;
                }

                ATA.diagonal().array() += m_Beta;

                return Solution<Scalar>(ATA.ldlt().solve(ATb));
            }

            template<std::floating_point ScalarType>
            class QRModelImplementation final
            {
            public:
                using Scalar = ScalarType;

                QRModelImplementation(Scalar beta) :
                    m_Beta{ beta }
                {
                }

                Solution<Scalar> fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    const CoordinateStates& coordinateStates) const;

            private:
                const Scalar m_Beta;
            };

            template<std::floating_point ScalarType>
            Solution<typename QRModelImplementation<ScalarType>::Scalar>
                QRModelImplementation<ScalarType>::fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    [[maybe_unused]] const CoordinateStates& coordinateStates) const
            {
                using Vector = Vector<Scalar>;
                using Matrix = Matrix<Scalar>;

                Matrix ATA;
                if (matrixIsCovariance)
                {
                    ATA = matData;
                }
                else
                {
                    ATA = matData.transpose() * matData;
                }

                Vector ATb;// Avoiding ternary operator as suggested by lib eigen c++
                if (matrixIsCovariance)
                {
                    ATb = matData * vectData;
                }
                else
                {
                    ATb = matData.transpose() * vectData;
                }

                ATA.diagonal().array() += m_Beta;

                return Solution<Scalar>(ATA.householderQr().solve(ATb));
            }

            template<std::floating_point ScalarType>
            using LDLTL2Regressor = L2RegressorDirect<LDLTModelImplementation<ScalarType>>;

            template<std::floating_point ScalarType>
            using QRL2Regressor = L2RegressorDirect<QRModelImplementation<ScalarType>>;
        }
	}
}

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            template<class MatrixType>
            class ATAPlusBetaIMatrix;
        }
    }
}

namespace Eigen
{
    namespace internal
    {
        template<class MatrixType>
        struct traits<l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType>> : 
            public traits<SparseMatrix<typename MatrixType::Scalar> > {};
    }
}

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            template<class MatrixType_>
            struct ATAPlusBetaIMatrix final : public Eigen::EigenBase<ATAPlusBetaIMatrix<MatrixType_>>
            {
            public:
                using MatrixType = MatrixType_;
                using Scalar = MatrixType::Scalar;
                using RealScalar = MatrixType::RealScalar;
                using StorageIndex = MatrixType::StorageIndex;

                enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic, IsRowMajor = false };

                auto rows() const { return m_A.get().cols(); }
                auto cols() const { return rows(); }

                template <typename Rhs_>
                auto operator*(const Eigen::MatrixBase<Rhs_>& x) const
                {
                    return Eigen::Product<ATAPlusBetaIMatrix, Rhs_, Eigen::AliasFreeProduct>(*this, x.derived());
                }

                explicit ATAPlusBetaIMatrix(const MatrixType& A, Scalar beta) :
                    m_A{ A }, m_beta{ beta }
                {
                }

                const std::reference_wrapper<const MatrixType> m_A;
                const Scalar m_beta;
            };
        }
    }
}

// Implementation of ATAPlusBetaIMatrix * Eigen::DenseVector 
// through a specialization of generic_product_impl
namespace Eigen
{
    namespace internal
    {
        template <class MatrixType_, class Rhs_>
        struct generic_product_impl<
            l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType_>, Rhs_, SparseShape, DenseShape,
            GemvProduct>  // GEMV stands for matrix-vector
            : generic_product_impl_base<l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType_>, Rhs_,
            generic_product_impl<l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType_>, Rhs_> >
        {
            using Scalar = typename Product<l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType_>, Rhs_, AliasFreeProduct>::Scalar;

            template <class Dest_>
            static void scaleAndAddTo(Dest_& dst, const l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType_>& lhs, const Rhs_& rhs, const Scalar& alpha)
            {
                // This method should implement "dst += alpha * lhs * rhs" inplace,
                // however, for iterative solvers, alpha is always equal to 1, so let's not bother about it.
                eigen_assert(alpha == static_cast<Scalar>(1) && "scaling is not implemented");
                EIGEN_ONLY_USED_FOR_DEBUG(alpha);

                const auto& A = lhs.m_A.get();

                dst = A.transpose() * (A * rhs) + lhs.m_beta * rhs;
            }
        };
    }
}

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            template<class MatType_>
            class ATAPlusBetaIDiagonalPreconditioner final : public Eigen::DiagonalPreconditioner<typename MatType_::Scalar>
            {
            public:
                using MatType = MatType_;
                using Scalar = MatType::Scalar;
                using RealScalar = MatType::RealScalar;// Eigen::NumTraits<Scalar>::Real;
                using Base = Eigen::DiagonalPreconditioner<Scalar>;
                using Base::m_invdiag;
                using Base::m_isInitialized;

                using Vector = decltype(m_invdiag);


                ATAPlusBetaIDiagonalPreconditioner() : Base()
                {
                }

                explicit ATAPlusBetaIDiagonalPreconditioner(const MatType& mat) : Base()
                {
                    compute(mat);
                }

                auto& analyzePattern(const MatType&)
                {
                    return *this;
                }

                auto& factorize(const MatType& mat)
                {
                    const auto& A = mat.m_A.get();

                    m_invdiag = (A.colwise().squaredNorm().transpose().array() + mat.m_beta).matrix().cwiseInverse();

                    m_isInitialized = true;

                    return *this;
                }

                auto& compute(const MatType& mat)
                {
                    return factorize(mat);
                }

                Eigen::ComputationInfo info()
                {
                    return Eigen::Success;
                }
            };

            template<std::floating_point ScalarType>
            class PCGModelImplementation final
            {
            public:
                using Scalar = ScalarType;

                PCGModelImplementation(Scalar beta)
                    : m_Beta{ beta }
                {
                }

                Solution<Scalar> fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar epsilon,
                    unsigned int maxNumberOfIterations,
                    const Solution<Scalar>& guess,
                    const CoordinateStates& coordinateStates) const;

            private:
                const Scalar m_Beta;
            };

            template<std::floating_point ScalarType>
            Solution<typename PCGModelImplementation<ScalarType>::Scalar>
                PCGModelImplementation<ScalarType>::fitNoIntercept(
                    const Matrix<Scalar>& matData,
                    const Vector<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar epsilon,
                    unsigned int maxNumberOfIterations,
                    [[maybe_unused]] const Solution<Scalar>& guess,
                    [[maybe_unused]] const CoordinateStates& coordinateStates) const
            {
                using Matrix = Matrix<Scalar>;
                using Vector = Vector<Scalar>;

                if (matrixIsCovariance)
                {
                    Matrix qPlusBetaI = matData;
                    qPlusBetaI.diagonal().array() += m_Beta;

                    Eigen::ConjugateGradient<Matrix, Eigen::Lower | Eigen::Upper> solver;

                    solver.setTolerance(epsilon);
                    solver.setMaxIterations(maxNumberOfIterations);

                    solver.compute(qPlusBetaI);

                    return Solution<Scalar>(solver.solve((matData * vectData).eval()));
                }
                else
                {
                    using ATAPlusBetaIMatrix = ATAPlusBetaIMatrix<Matrix>;

                    const ATAPlusBetaIMatrix aTAPlusBetaI{ matData, m_Beta };

                    Eigen::ConjugateGradient<ATAPlusBetaIMatrix, Eigen::Lower | Eigen::Upper,
                        ATAPlusBetaIDiagonalPreconditioner<ATAPlusBetaIMatrix>> solver;

                    solver.setTolerance(epsilon);
                    solver.setMaxIterations(maxNumberOfIterations);

                    solver.compute(aTAPlusBetaI);

                    return Solution<Scalar>(solver.solve((matData.transpose() * vectData).eval()));
                }
            }

            template<std::floating_point ScalarType>
            using PCGL2Regressor = L2RegressorIterative<PCGModelImplementation<ScalarType>>;
        }
    }
} // namespace l0l2

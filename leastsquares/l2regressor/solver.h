#ifndef L0L2_L2_REGRESSORS_H
#define L0L2_L2_REGRESSORS_H

#include <concepts>

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

namespace l0l2
{
	namespace linearmodel
	{
        template<std::floating_point ScalarType>
        class L2Regressor final
        {
        public:
            using Scalar = ScalarType;

            L2Regressor(Scalar beta) : m_Beta{ beta }
            {
            }

            Vector<Scalar> fitNoIntercept(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance) const;

        private:
            const Scalar m_Beta;
        };

        template<std::floating_point ScalarType>
        Vector<typename L2Regressor<ScalarType>::Scalar>
            L2Regressor<ScalarType>::fitNoIntercept(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance) const
        {
            using Vector = Vector<Scalar>;
            using Matrix = Matrix<Scalar>;

            const auto n = static_cast<Index>(matData.cols());

            auto ATA = matrixIsCovariance
                ? static_cast<Matrix>(matData)
                : static_cast<Matrix>(matData.transpose() * matData);

            auto ATb = matrixIsCovariance
                ? static_cast<Vector>(matData * vectData)
                : static_cast<Vector>(matData.transpose() * vectData);

            ATA.diagonal().array() += m_Beta;

            return ATA.ldlt().solve(ATb);
            //return ATA.householderQr().solve(ATb);
        }
	}
}

namespace l0l2
{
    namespace linearmodel
    {
        template<class MatrixType>
        class ATAPlusBetaIMatrix;
    }
}

namespace Eigen
{
    namespace internal
    {
        template<class MatrixType>
        struct traits<l0l2::linearmodel::ATAPlusBetaIMatrix<MatrixType>> : public traits<SparseMatrix<typename MatrixType::Scalar> > {};
    }
}

namespace l0l2
{
    namespace linearmodel
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

// Implementation of ATAPlusBetaIMatrix * Eigen::DenseVector 
// through a specialization of generic_product_impl
namespace Eigen
{
    namespace internal
    {
        template <class MatrixType_, class Rhs_>
        struct generic_product_impl<l0l2::linearmodel::ATAPlusBetaIMatrix<MatrixType_>, Rhs_, SparseShape, DenseShape,
            GemvProduct>  // GEMV stands for matrix-vector
            : generic_product_impl_base<l0l2::linearmodel::ATAPlusBetaIMatrix<MatrixType_>, Rhs_,
            generic_product_impl<l0l2::linearmodel::ATAPlusBetaIMatrix<MatrixType_>, Rhs_> >
        {
            using Scalar = typename Product<l0l2::linearmodel::ATAPlusBetaIMatrix<MatrixType_>, Rhs_, AliasFreeProduct>::Scalar;

            template <class Dest_>
            static void scaleAndAddTo(Dest_& dst, const l0l2::linearmodel::ATAPlusBetaIMatrix<MatrixType_>& lhs, const Rhs_& rhs, const Scalar& alpha)
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
        class L2RegressorPCG final
        {
        public:
            using Scalar = ScalarType;

            L2RegressorPCG(Scalar beta)
                : m_Beta{ beta }
            {
            }

            Vector<Scalar> fitNoIntercept(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar epsilon,
                unsigned int maxNumberOfIterations) const;

        private:
            Scalar m_Beta;
        };

        template<std::floating_point ScalarType>
        Vector<typename L2RegressorPCG<ScalarType>::Scalar>
            L2RegressorPCG<ScalarType>::fitNoIntercept(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar epsilon,
                unsigned int maxNumberOfIterations) const
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

                return solver.solve((matData * vectData).eval());
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

                return solver.solve((matData.transpose() * vectData).eval());
            }
        }
    }
}
#endif //L0L2_L2_REGRESSORS_H

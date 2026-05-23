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
            template <class ImplementationType>
            concept NoInterceptDirectL2RegressorLike =
                requires(ImplementationType impl) {
                    {
                        std::as_const(impl).fitNoIntercept(
                            Matrix<typename ImplementationType::Scalar>{},
                            Vector<typename ImplementationType::Scalar>{},
                            CoordinateStates{})
                    } -> std::convertible_to<
                        Solution<typename ImplementationType::Scalar>>;
                };

            template <class ImplementationType>
            concept NoInterceptIterativeL2RegressorLike =
                requires(ImplementationType impl) {
                    {
                        std::as_const(impl).fitNoIntercept(
                            Matrix<typename ImplementationType::Scalar>{},
                            Vector<typename ImplementationType::Scalar>{},
                            typename ImplementationType::Scalar{}, 0U,
                            Solution<typename ImplementationType::Scalar>{},
                            CoordinateStates{})
                    } -> std::convertible_to<
                        Solution<typename ImplementationType::Scalar>>;
                };

            template <NoInterceptDirectL2RegressorLike ImplementationType>
            class L2RegressorDirect final
            {
              public:
                using Implementation = ImplementationType;
                using Scalar = typename Implementation::Scalar;

                L2RegressorDirect(Scalar beta, bool hasIntercept)
                    : m_Beta{beta}, m_HasIntercept{hasIntercept}
                {
                }

                Solution<Scalar>
                fit(const Matrix<Scalar> &matData,
                    const Vector<Scalar> &vectData,
                    const CoordinateStates &coordinateStates = {}) const;

              private:
                const Scalar m_Beta;
                const bool m_HasIntercept;
            };

            template <NoInterceptDirectL2RegressorLike ImplementationType>
            Solution<typename L2RegressorDirect<ImplementationType>::Scalar>
            L2RegressorDirect<ImplementationType>::fit(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                const CoordinateStates &coordinateStates) const
            {
                const Implementation implementation{m_Beta};

                if (m_HasIntercept)
                {
                    auto solution = implementation.fitNoIntercept(
                        matData.rowwise() - matData.colwise().mean(),
                        vectData.array() - vectData.mean(), coordinateStates);

                    solution.intercept =
                        (vectData - matData * solution.x).mean();

                    return solution;
                }

                return implementation.fitNoIntercept(matData, vectData,
                                                          coordinateStates);
            }

            template <NoInterceptIterativeL2RegressorLike ImplementationType>
            class L2RegressorIterative final
            {
              public:
                using Implementation = ImplementationType;
                using Scalar = typename Implementation::Scalar;

                L2RegressorIterative(Scalar beta, bool hasIntercept)
                    : m_Beta{beta}, m_HasIntercept{hasIntercept}
                {
                }

                Solution<Scalar>
                fit(const Matrix<Scalar> &matData,
                    const Vector<Scalar> &vectData, Scalar epsilon,
                    unsigned int maxNumberOfIterations,
                    const Solution<Scalar> &guess = {},
                    const CoordinateStates &coordinateStates = {}) const;

              private:
                const Scalar m_Beta;
                const bool m_HasIntercept;
            };

            template <NoInterceptIterativeL2RegressorLike ImplementationType>
            Solution<typename L2RegressorIterative<ImplementationType>::Scalar>
            L2RegressorIterative<ImplementationType>::fit(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                Scalar epsilon, unsigned int maxNumberOfIterations,
                const Solution<Scalar> &guess,
                const CoordinateStates &coordinateStates) const
            {
                const Implementation implementation{m_Beta};

                if (m_HasIntercept)
                {
                    auto solution = implementation.fitNoIntercept(
                        matData.rowwise() - matData.colwise().mean(),
                        vectData.array() - vectData.mean(), epsilon,
                        maxNumberOfIterations, guess, coordinateStates);

                    solution.intercept =
                        (vectData - matData * solution.x).mean();

                    return solution;
                }

                return implementation.fitNoIntercept(
                    matData, vectData, epsilon, maxNumberOfIterations, guess,
                    coordinateStates);
            }

            template <std::floating_point ScalarType>
            class LDLTImplementation final
            {
              public:
                using Scalar = ScalarType;

                LDLTImplementation(Scalar beta) : m_Beta{beta} {}

                Solution<Scalar>
                fitNoIntercept(const Matrix<Scalar> &matData,
                               const Vector<Scalar> &vectData,
                               const CoordinateStates &coordinateStates) const;

              private:
                const Scalar m_Beta;
            };

            template <std::floating_point ScalarType>
            Solution<typename LDLTImplementation<ScalarType>::Scalar>
            LDLTImplementation<ScalarType>::fitNoIntercept(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                [[maybe_unused]] const CoordinateStates &coordinateStates) const
            {
                using Vector = Vector<Scalar>;
                using Matrix = Matrix<Scalar>;

                Matrix ATA = matData.transpose() * matData;
                ATA.diagonal().array() += m_Beta;

                const Vector ATb = matData.transpose() * vectData;

                return Solution<Scalar>(ATA.ldlt().solve(ATb));
            }

            template <std::floating_point ScalarType>
            class QRImplementation final
            {
              public:
                using Scalar = ScalarType;

                QRImplementation(Scalar beta) : m_Beta{beta} {}

                Solution<Scalar>
                fitNoIntercept(const Matrix<Scalar> &matData,
                               const Vector<Scalar> &vectData,
                               const CoordinateStates &coordinateStates) const;

              private:
                const Scalar m_Beta;
            };

            template <std::floating_point ScalarType>
            Solution<typename QRImplementation<ScalarType>::Scalar>
            QRImplementation<ScalarType>::fitNoIntercept(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                [[maybe_unused]] const CoordinateStates &coordinateStates) const
            {
                using Vector = Vector<Scalar>;
                using Matrix = Matrix<Scalar>;

                Matrix ATA = matData.transpose() * matData;
                ATA.diagonal().array() += m_Beta;

                const Vector ATb = matData.transpose() * vectData;

                return Solution<Scalar>(ATA.householderQr().solve(ATb));
            }

            template <std::floating_point ScalarType>
            using LDLTL2Regressor =
                L2RegressorDirect<LDLTImplementation<ScalarType>>;

            template <std::floating_point ScalarType>
            using QRL2Regressor =
                L2RegressorDirect<QRImplementation<ScalarType>>;
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            template <class MatrixType> class ATAPlusBetaIMatrix;
        }
    } // namespace linearmodel
} // namespace l0l2

namespace Eigen
{
    namespace internal
    {
        template <class MatrixType>
        struct traits<
            l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType>>
            : public traits<SparseMatrix<typename MatrixType::Scalar>>
        {
        };
    } // namespace internal
} // namespace Eigen

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            template <class MatrixType_>
            struct ATAPlusBetaIMatrix final
                : public Eigen::EigenBase<ATAPlusBetaIMatrix<MatrixType_>>
            {
              public:
                using MatrixType = MatrixType_;
                using Scalar = MatrixType::Scalar;
                using RealScalar = MatrixType::RealScalar;
                using StorageIndex = MatrixType::StorageIndex;

                enum
                {
                    ColsAtCompileTime = Eigen::Dynamic,
                    MaxColsAtCompileTime = Eigen::Dynamic,
                    IsRowMajor = false
                };

                auto rows() const { return m_A.get().cols(); }
                auto cols() const { return rows(); }

                template <typename Rhs_>
                auto operator*(const Eigen::MatrixBase<Rhs_> &x) const
                {
                    return Eigen::Product<ATAPlusBetaIMatrix, Rhs_,
                                          Eigen::AliasFreeProduct>(*this,
                                                                   x.derived());
                }

                explicit ATAPlusBetaIMatrix(const MatrixType &A, Scalar beta)
                    : m_A{A}, m_beta{beta}
                {
                }

                const std::reference_wrapper<const MatrixType> m_A;
                const Scalar m_beta;
            };
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

// Implementation of ATAPlusBetaIMatrix * Eigen::DenseVector
// through a specialization of generic_product_impl
namespace Eigen
{
    namespace internal
    {
        template <class MatrixType_, class Rhs_>
        struct generic_product_impl<
            l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<MatrixType_>,
            Rhs_, SparseShape, DenseShape,
            GemvProduct> // GEMV stands for matrix-vector
            : generic_product_impl_base<
                  l0l2::linearmodel::leastsquares::ATAPlusBetaIMatrix<
                      MatrixType_>,
                  Rhs_,
                  generic_product_impl<l0l2::linearmodel::leastsquares::
                                           ATAPlusBetaIMatrix<MatrixType_>,
                                       Rhs_>>
        {
            using Scalar = typename Product<l0l2::linearmodel::leastsquares::
                                                ATAPlusBetaIMatrix<MatrixType_>,
                                            Rhs_, AliasFreeProduct>::Scalar;

            template <class Dest_>
            static void scaleAndAddTo(Dest_ &dst,
                                      const l0l2::linearmodel::leastsquares::
                                          ATAPlusBetaIMatrix<MatrixType_> &lhs,
                                      const Rhs_ &rhs, const Scalar &alpha)
            {
                // This method should implement "dst += alpha * lhs * rhs"
                // inplace, however, for iterative solvers, alpha is always
                // equal to 1, so let's not bother about it.
                eigen_assert(alpha == static_cast<Scalar>(1) &&
                             "scaling is not implemented");
                EIGEN_ONLY_USED_FOR_DEBUG(alpha);

                const auto &A = lhs.m_A.get();

                dst = A.transpose() * (A * rhs) + lhs.m_beta * rhs;
            }
        };
    } // namespace internal
} // namespace Eigen

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            template <class MatType_>
            class ATAPlusBetaIDiagonalPreconditioner final
                : public Eigen::DiagonalPreconditioner<
                      typename MatType_::Scalar>
            {
              public:
                using MatType = MatType_;
                using Scalar = MatType::Scalar;
                using RealScalar =
                    MatType::RealScalar; // Eigen::NumTraits<Scalar>::Real;
                using Base = Eigen::DiagonalPreconditioner<Scalar>;
                using Base::m_invdiag;
                using Base::m_isInitialized;

                using Vector = decltype(m_invdiag);

                ATAPlusBetaIDiagonalPreconditioner() : Base() {}

                explicit ATAPlusBetaIDiagonalPreconditioner(const MatType &mat)
                    : Base()
                {
                    compute(mat);
                }

                auto &analyzePattern(const MatType &) { return *this; }

                auto &factorize(const MatType &mat)
                {
                    const auto &A = mat.m_A.get();

                    m_invdiag = (A.colwise().squaredNorm().transpose().array() +
                                 mat.m_beta)
                                    .matrix()
                                    .cwiseInverse();

                    m_isInitialized = true;

                    return *this;
                }

                auto &compute(const MatType &mat) { return factorize(mat); }

                Eigen::ComputationInfo info() { return Eigen::Success; }
            };

            template <std::floating_point ScalarType>
            class PCGImplementation final
            {
              public:
                using Scalar = ScalarType;

                PCGImplementation(Scalar beta) : m_Beta{beta} {}

                Solution<Scalar>
                fitNoIntercept(const Matrix<Scalar> &matData,
                               const Vector<Scalar> &vectData, Scalar epsilon,
                               unsigned int maxNumberOfIterations,
                               const Solution<Scalar> &guess,
                               const CoordinateStates &coordinateStates) const;

              private:
                const Scalar m_Beta;
            };

            template <std::floating_point ScalarType>
            Solution<typename PCGImplementation<ScalarType>::Scalar>
            PCGImplementation<ScalarType>::fitNoIntercept(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData,
                Scalar epsilon, unsigned int maxNumberOfIterations,
                [[maybe_unused]] const Solution<Scalar> &guess,
                [[maybe_unused]] const CoordinateStates &coordinateStates) const
            {
                using Matrix = Matrix<Scalar>;
                using Vector = Vector<Scalar>;

                using ATAPlusBetaIMatrix = ATAPlusBetaIMatrix<Matrix>;

                const ATAPlusBetaIMatrix aTAPlusBetaI{matData, m_Beta};

                Eigen::ConjugateGradient<
                    ATAPlusBetaIMatrix, Eigen::Lower | Eigen::Upper,
                    ATAPlusBetaIDiagonalPreconditioner<ATAPlusBetaIMatrix>>
                    solver;

                solver.setTolerance(epsilon);
                solver.setMaxIterations(maxNumberOfIterations);

                solver.compute(aTAPlusBetaI);

                return Solution<Scalar>(
                    solver.solve((matData.transpose() * vectData).eval()));
            }

            template <std::floating_point ScalarType>
            using PCGL2Regressor =
                L2RegressorIterative<PCGImplementation<ScalarType>>;
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

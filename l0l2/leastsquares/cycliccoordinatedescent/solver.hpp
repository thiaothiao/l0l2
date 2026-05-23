#pragma once

#include <l0l2/leastsquares/cycliccoordinatedescent/generic.hpp>
#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            /*! \brief Cyclical Coordinate Descent Regressor class.
             *
             * It produces solutions using cyclical coordinate descent
             * algorithm.
             */
            template <
                CyclicCoordinateDescentImplementationLike ImplementationType>
            class CyclicCoordinateDescent final
                : public CyclicCoordinateDescentDetails<ImplementationType>
            {
              public:
                using Base = CyclicCoordinateDescentDetails<ImplementationType>;
                using Implementation =
                    typename Base::Implementation;  /*!< Alias for the
                                                       implementation type*/
                using Param = typename Base::Param; /*!< Alias for the used
                                                       parameter type */
                using Scalar =
                    typename Implementation::Scalar; /*!< Alias for the used
                                                        scalar type */

                /*! \brief A cyclic coordinate descent solver object
                  constructor.
                  \param param regularization parameters.
                  \param withIntercept boolean indicating with intercept or not.
                  Default is false.
                */
                CyclicCoordinateDescent(const Param &param) : Base{param} {}

                /*! \brief Fit model.
                   \param matData contiguous data container representing matrix
                   in column major layout.
                   \param vectData contiguous data container representing target
                   vector.
                   \return a solution in the format Solution.
                 */
                Solution<Scalar> fit(const Matrix<Scalar> &matData,
                                     const Vector<Scalar> &vectData);
            };

            template <
                CyclicCoordinateDescentImplementationLike ImplementationType>
            Solution<
                typename CyclicCoordinateDescent<ImplementationType>::Scalar>
            CyclicCoordinateDescent<ImplementationType>::fit(
                const Matrix<Scalar> &matData, const Vector<Scalar> &vectData)
            {
                return Base::fit(matData, vectData, {});
            }
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

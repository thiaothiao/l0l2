#pragma once

#include <l0l2/leastsquares/cycliccoordinatedescent/details.hpp>
#include <l0l2/leastsquares/utils.hpp>

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            /*! \brief Generic Cyclical Coordinate Descent class.
             *
             * It produces solutions using cyclical coordinate descent
             * algorithm.
             */
            template <
                CyclicCoordinateDescentImplementationLike ImplementationType>
            class GenericCyclicCoordinateDescent final
                : public CyclicCoordinateDescentDetails<ImplementationType>
            {
              public:
                using Base = CyclicCoordinateDescentDetails<ImplementationType>;
                using Implementation =
                    typename Base::Implementation;    /*!< Alias for the
                                                         implementation type*/
                using Param = typename Base::Param;   /*!< Alias for the used
                                                         parameter type */
                using Scalar = typename Base::Scalar; /*!< Alias for the used
                                                         scalar type */

                /*! \brief A cyclic coordinate descent solver object
                  constructor.
                  \param param underlying implregularization parameters.
                */
                GenericCyclicCoordinateDescent(const Param &param) : Base{param} {}

                /*! \brief Fit model.
                   \param matData contiguous data container representing matrix
                   in column major layout.
                   \param vectData contiguous data container representing target
                   vector.
                   \return a solution in the format Solution.
                 */
                auto fit(const Matrix<Scalar> &matData,
                         const Vector<Scalar> &vectData)
                {
                    return Base::fit(matData, vectData, {});
                }
            };
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

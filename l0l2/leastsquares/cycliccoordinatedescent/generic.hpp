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
                  \param param underlying implementation parameters.
                */
                GenericCyclicCoordinateDescent(const Param &param) : Base{param} {}

                /*! \brief Fit model.
                   \param matData matrix containing the features data, #columns
                   = #features, #rows = #samples.
                   \param vectData vector containing the targets.
                   \return a result containing computed solution informations.
                 */
                Solution<Scalar> fit(const Matrix<Scalar> &matData,
                                     const Vector<Scalar> &vectData)
                {
                    return Base::fit(matData, vectData, {});
                }
            };
        } // namespace leastsquares
    } // namespace linearmodel
} // namespace l0l2

#pragma once

#include <concepts>

namespace l0l2
{
    namespace linearmodel
    {
        namespace leastsquares
        {
            namespace param
            {
                template<std::floating_point ScalarType>
                inline ScalarType getGamma(ScalarType beta, ScalarType delta)
                {
                    return static_cast<ScalarType>(2) * delta * beta;
                }

                template<std::floating_point ScalarType>
                inline ScalarType getDelta(ScalarType beta, ScalarType gamma)
                {
                    return gamma / (static_cast<ScalarType>(2) * beta);
                }

                template<std::floating_point ScalarType>
                inline ScalarType getBeta(ScalarType delta, ScalarType gamma)
                {
                    return gamma / (static_cast<ScalarType>(2) * delta);
                }
            }
        }
    }
} // namespace l0l2

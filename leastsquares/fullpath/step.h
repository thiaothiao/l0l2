#ifndef L0L2_FULL_PATH_STEP_H
#define L0L2_FULL_PATH_STEP_H

#include <iostream>
#include <concepts>
#include <cmath>
#include <cstdint>
#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <execution>

#include "leastsquares/utils.h"

namespace l0l2
{
    namespace linearmodel
    {
        template<std::floating_point ScalarType>
        class FullPathStep final
        {
        public:
            using Scalar = ScalarType;

            FullPathStep(Scalar beta) : m_Beta{ beta }
            {
            }

            Solution<Scalar> run(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                const Solution<Scalar>& solutionYaay,
                const Solution<Scalar>& solutionMaam,
                bool matrixIsCovariance,
                Scalar tau) const;

        private:
            const Scalar m_Beta;
        };

        enum class ConstraintsType : std::uint8_t
        {
            K0 = 0U,
            K1,
            K2,
            K3,
            K4,
            K5,
            K6,
            K7
        };

        template<std::floating_point ScalarType>
        auto updateSizesAndSigns(ScalarType beta,
            const Solution<ScalarType>& solutionYaay,
            const Solution<ScalarType>& solutionMaam,
            Index numberOfColumns)
        {
            using Scalar = ScalarType;
            using Utils = Utils<Scalar>;
            using Vector = Vector<Scalar>;

            const auto n = numberOfColumns;

            const auto& solutionMaamx = solutionMaam.x;
            const auto solutionMaamdelta = solutionMaam.delta;

            const auto& solutionYaayx = solutionYaay.x;
            const auto& solutionYaaygrad = solutionYaay.grad;
            const auto solutionYaaydelta = solutionYaay.delta;

            std::vector<ConstraintsType> indices(n);
            auto indicesPtr = indices.data();

            Vector xSigns(n);

#pragma omp parallel for
            for (Index i = 0; i < n; ++i)
            {
                if (std::abs(solutionYaayx[i]) <= Utils::epsilon)
                {
                    const auto gradi = solutionYaaygrad[i];
                    if (std::abs(gradi) > Utils::epsilon)
                    {
                        xSigns[i] = -Utils::sign(gradi);
                    }
                    else
                    {
                        xSigns[i] = static_cast<Scalar>(1);
                    }
                }
                else
                {
                    xSigns[i] = Utils::sign(solutionYaayx[i]);
                }
            }

            const auto deltaYaayBeta = solutionYaaydelta * beta;
#pragma omp parallel for
            for (Index i = 0; i < n; ++i)
            {
                const auto absXMaami = std::abs(solutionMaamx[i]);
                const auto absXYaayi = std::abs(solutionYaayx[i]);

                if (std::abs(absXYaayi - solutionYaaydelta) <= Utils::epsilon)
                {
                    if (std::abs(absXMaami - solutionMaamdelta) <= Utils::epsilon)
                    {
                        indicesPtr[i] = ConstraintsType::K1;
                    }
                    else if (absXMaami > solutionMaamdelta)
                    {
                        indicesPtr[i] = ConstraintsType::K2;
                    }
                    else//if (absXMaami < solutionMaamdelta)
                    {
                        indicesPtr[i] = ConstraintsType::K1;
                    }
                }
                else if (absXYaayi > solutionYaaydelta)
                {
                    indicesPtr[i] = ConstraintsType::K0;
                }
                else if (Utils::epsilon < absXYaayi && absXYaayi < solutionYaaydelta)
                {
                    indicesPtr[i] = ConstraintsType::K3;
                }
                else// absXYaayi <= Utils::epsilon case
                {
                    const auto absGradi = std::abs(solutionYaaygrad[i]);

                    if (std::abs(absGradi - deltaYaayBeta) <= Utils::epsilon)
                    {
                        if (absXMaami <= Utils::epsilon)
                        {
                            indicesPtr[i] = ConstraintsType::K4;
                        }
                        else
                        {
                            indicesPtr[i] = ConstraintsType::K5;
                        }
                    }
                    else if (absGradi <= Utils::epsilon)
                    {
                        indicesPtr[i] = ConstraintsType::K7;
                    }
                    else //if (Utils::epsilon < absATAxMinusBi && absATAxMinusBi < deltaYaayBeta)
                    {
                        indicesPtr[i] = ConstraintsType::K6;
                    }
                }
            }

            return std::pair<std::vector<ConstraintsType>, Vector>
            {std::move(indices), std::move(xSigns)};
        }

        auto computeNbWs(const std::vector<ConstraintsType>& indices)
        {
            auto nbWs = static_cast<Index>(0);
            for (auto idx : indices)
            {
                if (idx == ConstraintsType::K5
                    || idx == ConstraintsType::K6
                    || idx == ConstraintsType::K7)
                {
                    ++nbWs;
                }
            }

            return nbWs;
        }

        template<std::floating_point ScalarType>
        Solution<typename FullPathStep<ScalarType>::Scalar>
            FullPathStep<ScalarType>::run(
                const Matrix<Scalar>& matData,
                const Vector<Scalar>& vectData,
                const Solution<Scalar>& solutionYaay,
                const Solution<Scalar>& solutionMaam,
                bool matrixIsCovariance,
                Scalar tau) const
        {
            using Solution = Solution<Scalar>;
            using Utils = Utils<Scalar>;
            using Matrix = Matrix<Scalar>;
            using RMMatrix = RMMatrix<Scalar>;
            using Vector = Vector<Scalar>;

            const auto n = static_cast<Index>(matData.cols());

            auto [indices, xSigns] = updateSizesAndSigns(m_Beta, solutionYaay, solutionMaam, n);

            const auto& VectData = vectData;

            auto indicesPtr = indices.data();

            RMMatrix AData;

            const auto numberOfConstraints = 2 * n;

            auto nbWs = computeNbWs(indices);
            auto nbTs = n - nbWs;

            Vector b = Vector::Zero(numberOfConstraints);

            Vector gamma = Vector::Zero(numberOfConstraints);

            const auto solutionYaaydelta = solutionYaay.delta;

            const auto deltaYaayBeta = solutionYaaydelta * m_Beta;

            std::unordered_map<Index, Index> indicesMap;

            std::vector<Index> tIndicesMap(n, static_cast<Index>(-1));

            std::vector<Index> wIndicesMap(n, static_cast<Index>(-1));

            std::vector<Index> pivots(n, static_cast<Index>(-1));

            auto presolveDone = false;

            while (!presolveDone)
            {
                const auto jStartSlacksT = n;
                const auto jStartSlacksW = jStartSlacksT + nbTs;
                const auto jStartSlacksS = jStartSlacksW + nbWs;

                indicesMap.clear();

                std::fill(std::execution::par,
                    pivots.begin(), pivots.end(), static_cast<Index>(-1));

                {
                    Index idx = 0;
                    for (Index j = 0; j < n; ++j)
                    {
                        const auto idxType = indicesPtr[j];

                        if (idxType == ConstraintsType::K0
                            || idxType == ConstraintsType::K1
                            || idxType == ConstraintsType::K2
                            || idxType == ConstraintsType::K3
                            || idxType == ConstraintsType::K4)
                        {
                            indicesMap[j] = idx++;
                        }
                    }
                }

                {
                    const auto nA = static_cast<Index>(indicesMap.size());
                    const auto mA = n;

                    AData = RMMatrix(mA, nA);

                    for (const auto& [j, idx] : indicesMap)
                    {
                        AData.col(idx) = matrixIsCovariance ?
                            static_cast<Vector>(matData.col(j))
                            : static_cast<Vector>(matData.transpose() * matData.col(j));

                        AData.coeffRef(j, idx) += m_Beta;

                        AData.col(idx) *= xSigns[j];
                    }
                }

                gamma.setZero();

                b.setZero();

                b.head(n) = VectData;// ATb or Qalpha

                Index jT = 0;
                Index jW = 0;
                Index jS = 0;

                for (Index i = 0; i < n; ++i)
                {
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gamma[tIndex] = tau;

                        b[tIndex] = solutionYaaydelta;

                        tIndicesMap[i] = tIndex;

                        ++jT;

                        pivots[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gamma[tIndex] = tau;

                        b[tIndex] = solutionYaaydelta;

                        tIndicesMap[i] = tIndex;

                        ++jT;

                        pivots[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gamma[tIndex] = tau;

                        b[tIndex] = solutionYaaydelta;

                        tIndicesMap[i] = tIndex;

                        ++jT;

                        pivots[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gamma[tIndex] = tau;

                        b[tIndex] = solutionYaaydelta;

                        tIndicesMap[i] = tIndex;

                        ++jT;

                        pivots[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gamma[tIndex] = tau;

                        b[tIndex] = solutionYaaydelta;

                        tIndicesMap[i] = tIndex;

                        ++jT;

                        pivots[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        const auto sIndex = jStartSlacksS + jS;
                        const auto wIndex = jStartSlacksW + jW;

                        gamma[wIndex] = tau;

                        b[wIndex] = solutionYaaydelta;

                        //sIndicesMap[i] = sIndex;
                        wIndicesMap[i] = wIndex;

                        ++jS;
                        ++jW;

                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        const auto sIndex = jStartSlacksS + jS;
                        const auto wIndex = jStartSlacksW + jW;

                        gamma[wIndex] = tau;

                        b[wIndex] = solutionYaaydelta;

                        //sIndicesMap[i] = sIndex;
                        wIndicesMap[i] = wIndex;

                        ++jS;
                        ++jW;

                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        const auto sIndex = jStartSlacksS + jS;
                        const auto wIndex = jStartSlacksW + jW;

                        gamma[wIndex] = tau;

                        b[wIndex] = solutionYaaydelta;

                        //sIndicesMap[i] = sIndex;
                        wIndicesMap[i] = wIndex;

                        ++jS;
                        ++jW;

                        break;
                    }
                    default:
                        std::cout << "\nNOT POSSIBLE\n";
                        break;
                    }
                }

#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = tIndicesMap[i];

                        const auto betaTimesSigni = m_Beta * xSigns[i];

                        AData.coeffRef(i, indicesMap.at(i)) -= betaTimesSigni;

                        gamma[i] -= betaTimesSigni * gamma[tIndex];

                        b[i] -= betaTimesSigni * b[tIndex];

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        const auto tIndex = tIndicesMap[i];

                        const auto betaTimesSigni = m_Beta * xSigns[i];

                        AData.coeffRef(i, indicesMap.at(i)) -= betaTimesSigni;

                        gamma[i] -= betaTimesSigni * gamma[tIndex];

                        b[i] -= betaTimesSigni * b[tIndex];

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        const auto tIndex = tIndicesMap[i];

                        const auto betaTimesSigni = m_Beta * xSigns[i];

                        AData.coeffRef(i, indicesMap.at(i)) -= betaTimesSigni;

                        gamma[i] -= betaTimesSigni * gamma[tIndex];

                        b[i] -= betaTimesSigni * b[tIndex];

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        const auto wIndex = wIndicesMap[i];

                        const auto betaTimesSigni = m_Beta * xSigns[i];

                        gamma[i] -= betaTimesSigni * gamma[wIndex];

                        b[i] -= betaTimesSigni * b[wIndex];

                        break;
                    }
                    default:
                        break;
                    }
                }

                // do not parallelize           
                for (Index i = 0; i < n; ++i)
                {
                    const auto pivotRowIndex = i;
                    const auto pivotColumnIndex = pivots[i];

                    if (pivotColumnIndex == static_cast<Index>(-1))
                    {
                        continue;
                    }

                    const auto pivotCoeff = AData.coeff(pivotRowIndex, pivotColumnIndex);

                    AData.row(pivotRowIndex) /= pivotCoeff;

                    gamma[pivotRowIndex] /= pivotCoeff;

                    b[pivotRowIndex] /= pivotCoeff;

#pragma omp parallel for
                    for (Index rowIndex = 0; rowIndex < n; ++rowIndex)
                    {
                        if (pivotRowIndex != rowIndex)
                        {
                            const auto value = AData.coeff(rowIndex, pivotColumnIndex);
                            if (std::abs(value) > Utils::epsilon)
                            {
                                AData.row(rowIndex) -= value * AData.row(pivotRowIndex);

                                gamma[rowIndex] -= value * gamma[pivotRowIndex];

                                b[rowIndex] -= value * b[pivotRowIndex];
                            }
                        }
                    }
                }

#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        const auto tIndex = tIndicesMap[i];

                        gamma[tIndex] -= gamma[i];

                        b[tIndex] -= b[i];

                        gamma[tIndex] *= static_cast<Scalar>(-1);

                        b[tIndex] *= static_cast<Scalar>(-1);

                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        const auto tIndex = tIndicesMap[i];

                        gamma[tIndex] -= gamma[i];

                        b[tIndex] -= b[i];

                        gamma[tIndex] *= static_cast<Scalar>(-1);

                        b[tIndex] *= static_cast<Scalar>(-1);

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = tIndicesMap[i];

                        gamma[tIndex] -= gamma[i];

                        b[tIndex] -= b[i];

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        const auto tIndex = tIndicesMap[i];

                        gamma[tIndex] -= gamma[i];

                        b[tIndex] -= b[i];

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        const auto tIndex = tIndicesMap[i];

                        gamma[tIndex] -= gamma[i];

                        b[tIndex] -= b[i];

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        const auto wIndex = wIndicesMap[i];

                        const auto signiOverBeta = xSigns[i] / m_Beta;

                        gamma[i] *= signiOverBeta;

                        b[i] *= signiOverBeta;

                        gamma[wIndex] -= gamma[i];

                        b[wIndex] -= b[i];

                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        const auto wIndex = wIndicesMap[i];

                        const auto signiOverBeta = xSigns[i] / m_Beta;

                        gamma[i] *= signiOverBeta;

                        b[i] *= signiOverBeta;

                        gamma[wIndex] -= gamma[i];

                        b[wIndex] -= b[i];

                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        const auto wIndex = wIndicesMap[i];

                        const auto minusSigniOver2Beta = -xSigns[i] / (static_cast<Scalar>(2) * m_Beta);

                        gamma[i] *= minusSigniOver2Beta;

                        b[i] *= minusSigniOver2Beta;

                        gamma[wIndex] -= gamma[i];

                        b[wIndex] -= b[i];

                        break;
                    }
                    default:
                        break;
                    }
                }

                bool bHasZeros = false;
                for (Index i = 0; i < n; ++i)
                {
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K1:
                    {
                        const auto tIndex = tIndicesMap[i];

                        if (b[tIndex] <= Utils::epsilon && gamma[tIndex] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K2;

                            bHasZeros = true;
                        }

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = tIndicesMap[i];

                        if (b[tIndex] <= Utils::epsilon && gamma[tIndex] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K1;

                            bHasZeros = true;
                        }

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        if (b[i] <= Utils::epsilon && gamma[i] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K5;

                            bHasZeros = true;
                        }

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        const auto wIndex = wIndicesMap[i];

                        if (b[wIndex] <= Utils::epsilon && gamma[wIndex] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K4;

                            bHasZeros = true;
                        }

                        break;
                    }
                    default:
                        break;
                    }
                }

                presolveDone = !bHasZeros;

                nbWs = computeNbWs(indices);
                nbTs = n - nbWs;
            }

            Solution solutionNew{ n };

            // extract solution
            auto minimum = std::numeric_limits<Scalar>::max();
            auto aPivotRowIndex = static_cast<Index>(-1);
            {
                for (Index i = 0; i < numberOfConstraints; ++i)
                {
                    const auto value = gamma[i];
                    if (value > Utils::epsilon)
                    {
                        const auto itemValue = b[i] / value;
                        if (itemValue < minimum)
                        {
                            minimum = itemValue;

                            aPivotRowIndex = i;
                        }
                    }
                }
            }

            if (static_cast<Index>(-1) != aPivotRowIndex)
            {
                const auto gammaSol = b[aPivotRowIndex] / gamma[aPivotRowIndex];

                solutionNew.delta = solutionYaaydelta - tau * gammaSol;

                auto& solutionNewx = solutionNew.x;
                auto& solutionNewgrad = solutionNew.grad;
#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewx[i] = b[i] - gamma[i] * gammaSol;
                        }

                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewx[i] = b[i] - gamma[i] * gammaSol;
                        }

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewx[i] = b[i] - gamma[i] * gammaSol;
                        }

                        const auto tIndex = tIndicesMap[i];
                        if (tIndex != aPivotRowIndex)
                        {
                            const auto Ti = b[tIndex] - gamma[tIndex] * gammaSol;

                            solutionNewgrad[i] = -m_Beta * xSigns[i] * Ti;
                        }

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewx[i] = b[i] - gamma[i] * gammaSol;
                        }

                        const auto tIndex = tIndicesMap[i];
                        if (tIndex != aPivotRowIndex)
                        {
                            const auto Ti = b[tIndex] - gamma[tIndex] * gammaSol;

                            solutionNewgrad[i] = -m_Beta * xSigns[i] * Ti;
                        }

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewx[i] = b[i] - gamma[i] * gammaSol;
                        }

                        const auto tIndex = tIndicesMap[i];
                        if (tIndex != aPivotRowIndex)
                        {
                            const auto Ti = b[tIndex] - gamma[tIndex] * gammaSol;

                            solutionNewgrad[i] = -m_Beta * xSigns[i] * Ti;
                        }

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        if (i != aPivotRowIndex)
                        {
                            const auto Si = b[i] - gamma[i] * gammaSol;

                            solutionNewgrad[i] = -m_Beta * xSigns[i] * Si;
                        }

                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        if (i != aPivotRowIndex)
                        {
                            const auto Si = b[i] - gamma[i] * gammaSol;

                            solutionNewgrad[i] = -m_Beta * xSigns[i] * Si;
                        }

                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        if (i != aPivotRowIndex)
                        {
                            const auto Si = b[i] - gamma[i] * gammaSol;

                            solutionNewgrad[i] += m_Beta * xSigns[i] * Si;
                        }

                        const auto wIndex = wIndicesMap[i];
                        if (wIndex != aPivotRowIndex)
                        {
                            const auto Wi = b[wIndex] - gamma[wIndex] * gammaSol;

                            solutionNewgrad[i] -= m_Beta * xSigns[i] * Wi;
                        }

                        break;
                    }
                    default:
                        break;
                    }
                }

                // update signs
                solutionNewx.array() *= xSigns.array();

                return solutionNew;
            }
            else
            {
                std::cout << "\nEXceptional Case: unbounded LP.\n";
            }

            return Solution{};
        }
    }
}
#endif //L0L2_FULL_PATH_STEP_H
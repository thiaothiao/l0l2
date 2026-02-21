#ifndef L0L2_FULL_PATH_SOLVER_H
#define L0L2_FULL_PATH_SOLVER_H

#include <list>
#include <limits>
#include <mutex>
#include <iostream>
#include <ios>
#include <iomanip>
#include <future>
#include <utility>
#include <cmath>
#include <execution>
#include <functional>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <concepts>

#include "Utils.h"

namespace l0l2
{
    namespace linearmodel
    {
        /* \brief Full path solver class.
        *
        * It produces solutions from partial path or full path computations.
        */
        template<std::floating_point ScalarType>
        class FullPathSolver
        {
        public:
            using Scalar = ScalarType; /*!< Alias for the used scalar type */

            struct Param final
            {
                Param(
                    Scalar deltaInput = static_cast<Scalar>(0),
                    Scalar betaInput = static_cast<Scalar>(1),
                    Strategy strategyInput = Strategy::FromZeroSolution)
                    :delta{ deltaInput },
                    beta{ betaInput },
                    strategy{ strategyInput }
                    //to optimize
                {
                }

                const Scalar delta;
                const Scalar beta;
                const Strategy strategy;
            };

            /*! \brief A solver object constructor.
              \param delta sparsity regularization parameter.
              \param beta l2 regularization parameter.
              \param strategy an enum indicating a strategy: from zero, or l2 or both solutions.
            */
            /*FullPathSolver(
                Scalar delta,
                Scalar beta,
                Strategy strategy = Strategy::FromZeroSolution) :
                m_Param{ delta, beta,  strategy },
                m_DeltaFromZeroSolution{ std::numeric_limits<Scalar>::max() },
                m_DeltaFromL2Solution{ static_cast<Scalar>(0) }
            {
            }*/

            FullPathSolver(const Param& param) :
                m_Param{ param },
                m_DeltaFromZeroSolution{ std::numeric_limits<Scalar>::max() },
                m_DeltaFromL2Solution{ static_cast<Scalar>(0) }
            {
            }

            /*! \brief Fit full path solutions.
               \param matData contiguous data container representing matrix in column major layout.
               \param numberOfRows matrix number of rows.
               \param numberOfColumns matrix number of columns.
               \param vectData contiguous data container representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \param beta l2 regularization parameter.
               \param strategy an enum indicating a strategy: from zero, or l2 or both solutions.
               \return a list of solutions generating entire piecewise linear path solutions
             */
            static std::list<Solution<Scalar>>
                fitAll(const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                    Index numberOfRows, Index numberOfColumns,
                    const ContiguousDataContainer<Scalar>& vectData,
                    bool matrixIsCovariance,
                    Scalar beta,
                    Strategy strategy = Strategy::FromZeroSolution);

            /*! \brief Fit full path solutions.
               \param matData contiguous data container representing matrix in column major layout.
               \param numberOfRows matrix number of rows.
               \param numberOfColumns matrix number of columns.
               \param vectData contiguous data pointer representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \param beta l2 regularization parameter.
               \param strategy an enum indicating a strategy: from zero, or l2 or both solutions.
               \return a list of solutions generating entire piecewise linear path solutions
             */
            static std::list<Solution<Scalar>>
                fittAll(const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                    Index numberOfRows, Index numberOfColumns,
                    const Scalar* vectDataPtr,
                    bool matrixIsCovariance,
                    Scalar beta,
                    Strategy strategy = Strategy::FromZeroSolution);

            /*! \brief Fit one solution.
               \param matData contiguous data container representing matrix in column major layout.
               \param numberOfRows matrix number of rows.
               \param numberOfColumns matrix number of columns.
               \param vectData contiguous data container representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \return a solution corresponding to the regularization parameters.
             */
            Solution<Scalar>  fit(const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
                bool matrixIsCovariance);

            /*! \brief Fit one solution.
               \param matData contiguous data container representing matrix in column major layout.
               \param numberOfRows matrix number of rows.
               \param numberOfColumns matrix number of columns.
               \param vectDataPtr contiguous data pointer representing target vector.
               \param matrixIsCovariance a boolean indicating if matrix is covariance or not.
               \return a solution corresponding to the regularization parameters.
             */
            Solution<Scalar>  fitt(const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance);

        private:
            std::list<Solution<Scalar>> solve(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance, bool fromZeroSolution);

            const Param m_Param;

            volatile Scalar m_DeltaFromZeroSolution;
            volatile Scalar m_DeltaFromL2Solution;

            std::mutex m_DeltasMutex;
        };

        template<std::floating_point ScalarType>
        class L2RegressorGauss
        {
        public:
            using Scalar = ScalarType;

            L2RegressorGauss(Scalar beta) : m_Beta{ beta }
            {
            }

            ContiguousDataContainer<Scalar> fit(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance) const;

        private:
            const Scalar m_Beta;
        };

        template<std::floating_point ScalarType>
        ContiguousDataContainer<typename L2RegressorGauss<ScalarType>::Scalar>
            L2RegressorGauss<ScalarType>::fit(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance) const
        {// TODO optimize
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using Utils = Utils<Scalar>;
            using Index = Index;

            const auto n = numberOfColumns;
            const auto m = numberOfRows;

            const auto& MatData = matData;// MatDataPtr[j*numberOfRows + i]
            const auto MatDataPtr = MatData.data();

            ContiguousDataContainer ATA(n * n);// ATAPtr[i*numberOfColumns + j]
            auto ATAPtr = ATA.data();

            ContiguousDataContainer ATb(n);
            auto ATbPtr = ATb.data();

            if (matrixIsCovariance)
            {
                const auto alphaPtr = vectDataPtr;// vectData.data();

#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto MatDataPtrColi = MatDataPtr + i * n;
                    auto ATAPtrRowi = ATAPtr + i * n;

                    auto total = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index j = 0; j < n; ++j)
                    {//Vectorized
                        //qij == qji
                        total += MatDataPtrColi[j] * alphaPtr[j];

                        ATAPtrRowi[j] = MatDataPtrColi[j];// TODO exploit matrix symmetry
                    }

                    ATbPtr[i] = total;
                }
            }
            else
            {
                const auto bPtr = vectDataPtr;//vectData.data();
#pragma omp parallel for
                for (Index j = 0; j < n; ++j)
                {//Not auto vectorized
                    const auto MatDataPtrColj = MatDataPtr + j * m;
                    auto total = static_cast<Scalar>(0);
#pragma omp simd
                    for (Index i = 0; i < m; ++i)
                    {//Vectorized
                        total += bPtr[i] * MatDataPtrColj[i];
                    }

                    ATbPtr[j] = total;
                }

                // A.transpose(A)
#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto MatDataPtrColi = MatDataPtr + i * m;
                    auto ATAPtrRowi = ATAPtr + i * n;
                    for (Index j = i; j < n; ++j)
                    {//Not auto vectorized
                        const auto MatDataPtrColj = MatDataPtr + j * m;
                        auto coeffij = static_cast<Scalar>(0);
#pragma omp simd
                        for (Index k = 0; k < m; ++k)
                        {//Vectorized
                            coeffij += MatDataPtrColi[k] * MatDataPtrColj[k];
                        }

                        ATAPtrRowi[j] = coeffij;
                        ATAPtr[j * n + i] = coeffij;
                    }
                }
            }

#pragma omp parallel for
            for (Index i = 0; i < n; ++i)
            {//Not auto vectorized
                ATAPtr[i * n + i] += m_Beta;
            }

            // do not parallelize
            for (Index i = 0; i < n; ++i)
            {//Not auto vectorized
                const auto pivotRowIndex = i;
                const auto pivotColumnIndex = i;

                auto ATAPtrpivotRowIndex = ATAPtr + pivotRowIndex * n;

                const auto pivotCoeff = ATAPtrpivotRowIndex[pivotColumnIndex];
#pragma omp simd
                for (Index j = 0; j < n; ++j)
                {//Vectorized
                    ATAPtrpivotRowIndex[j] /= pivotCoeff;
                }

                ATbPtr[pivotRowIndex] /= pivotCoeff;
#pragma omp parallel for
                for (Index rowIndex = 0; rowIndex < n; ++rowIndex)
                {//Not auto vectorized
                    if (pivotRowIndex != rowIndex)
                    {
                        const auto value = ATAPtr[rowIndex * n + pivotColumnIndex];
                        if (std::abs(value) > Utils::epsilon)
                        {
                            auto ATAPtrrowIndex = ATAPtr + rowIndex * n;
#pragma omp simd
                            for (Index j = 0; j < n; ++j)
                            {//Vectorized
                                ATAPtrrowIndex[j] -= value * ATAPtrpivotRowIndex[j];
                            }

                            ATbPtr[rowIndex] -= value * ATbPtr[pivotRowIndex];
                        }
                    }
                }
            }

            return ATb;
        }


        template<std::floating_point ScalarType>
        class FullPathStep
        {
        public:
            using Scalar = ScalarType;

            FullPathStep(Scalar beta) : m_Beta{ beta }
            {
            }

            Solution<Scalar> run(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
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
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;

            const auto solutionMaamxPtr = solutionMaam.x.data();
            const auto solutionMaamdelta = solutionMaam.delta;

            const auto solutionYaayxPtr = solutionYaay.x.data();
            const auto solutionYaaygradPtr = solutionYaay.grad.data();
            const auto solutionYaaydelta = solutionYaay.delta;

            auto indices = std::vector<ConstraintsType>(numberOfColumns, ConstraintsType::K0);
            auto xSigns = ContiguousDataContainer(numberOfColumns);

            auto xSignsPtr = xSigns.data();

            auto indicesPtr = indices.data();

            const auto n = numberOfColumns;
#pragma omp parallel for
            for (Index i = 0; i < n; ++i)
            {//Not auto vectorized
                // TODO optimize
                if (std::abs(solutionYaayxPtr[i]) <= Utils::epsilon)
                {
                    const auto gradi = solutionYaaygradPtr[i];
                    if (std::abs(gradi) > Utils::epsilon)
                    {
                        xSignsPtr[i] = -Utils::sign(gradi);
                    }
                    else
                    {
                        xSignsPtr[i] = static_cast<Scalar>(1);
                    }
                }
                else
                {
                    xSignsPtr[i] = Utils::sign(solutionYaayxPtr[i]);
                }
            }

            const auto deltaYaayBeta = solutionYaaydelta * beta;
#pragma omp parallel for
            for (Index i = 0; i < n; ++i)
            {//Not auto vectorized
                const auto absXMaami = std::abs(solutionMaamxPtr[i]);
                const auto absXYaayi = std::abs(solutionYaayxPtr[i]);

                if (std::abs(absXYaayi - solutionYaaydelta) <= Utils::epsilon)
                {
                    if (std::abs(absXMaami - solutionMaamdelta) <= Utils::epsilon)
                    {// crtical
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
                    const auto absGradi = std::abs(solutionYaaygradPtr[i]);

                    if (std::abs(absGradi - deltaYaayBeta) <= Utils::epsilon)
                    {//critical
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

            return std::pair<std::vector <ConstraintsType>, ContiguousDataContainer>
            {std::move(indices), std::move(xSigns)};
        }

        auto computeNbWs(Index numberOfColumns, const std::vector<ConstraintsType>& indices)
        {
            auto nbWs = static_cast<Index>(0);
            for (auto idx : indices)
            {//Not auto vectorized
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
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
                const Solution<Scalar>& solutionYaay,
                const Solution<Scalar>& solutionMaam,
                bool matrixIsCovariance,
                Scalar tau) const
        {
            using Solution = Solution<Scalar>;
            using Utils = Utils<Scalar>;
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using Index = Index;

            auto [indices, xSigns] = updateSizesAndSigns(m_Beta, solutionYaay, solutionMaam, numberOfColumns);

            const auto& MatData = matData; //MatData[j*numberOfRows + i]
            const auto MatDataPtr = MatData.data();

            const auto& VectData = vectData;

            const auto xSignsPtr = xSigns.data();

            auto indicesPtr = indices.data();

            const auto n = numberOfColumns;
            const auto m = numberOfRows;

            ContiguousDataContainer AData;

            const auto m_NumberOfConstraints = 2 * n;

            auto m_NbWs = computeNbWs(numberOfColumns, indices);
            auto m_NbTs = n - m_NbWs;

            ContiguousDataContainer b(m_NumberOfConstraints, static_cast<Scalar>(0));
            auto bPtr = b.data();

            ContiguousDataContainer gamma(m_NumberOfConstraints, static_cast<Scalar>(0));
            auto gammaPtr = gamma.data();

            const auto solutionYaaydelta = solutionYaay.delta;

            const auto deltaYaayBeta = solutionYaaydelta * m_Beta;

            std::unordered_map<Index, Index> indicesMap;

            std::vector<Index> tIndicesMap(n, static_cast<Index>(-1));
            auto tIndicesMapPtr = tIndicesMap.data();

            std::vector<Index> wIndicesMap(n, static_cast<Index>(-1));
            auto wIndicesMapPtr = wIndicesMap.data();

            std::vector<Index> pivots(n, static_cast<Index>(-1));
            auto pivotsPtr = pivots.data();

            auto presolveDone = false;

            while (!presolveDone)
            {
                const auto jStartSlacksT = n;
                const auto jStartSlacksW = jStartSlacksT + m_NbTs;
                const auto jStartSlacksS = jStartSlacksW + m_NbWs;

                indicesMap.clear();// TODO is it necessary to clear all

                //tIndicesMapPtr reset values to -1;// TODO is it necessary to clear all
                //wIndicesMapPtr reset values to -1;
                //sIndicesMap reset values to -1;
                //pivots reset values to -1;
                std::fill(std::execution::par,
                    pivots.begin(), pivots.end(), static_cast<Index>(-1));

                {
                    Index idx = 0;
                    for (Index j = 0; j < n; ++j)
                    {//Not auto vectorized
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

                const auto nA = static_cast<Index>(indicesMap.size());
                const auto mA = n;

                AData = ContiguousDataContainer(mA * nA);//ADataPtr[i*nA + j]
                auto ADataPtr = AData.data();

                if (matrixIsCovariance)
                {
                    // TODO parallelize map for loop
                    for (const auto& [j, idx] : indicesMap)
                    {//Not auto vectorized
                        const auto MatDataPtrColj = MatDataPtr + j * m;
                        for (Index k = 0; k < mA; ++k)
                        {//Not auto vectorized
                            ADataPtr[k * nA + idx] = MatDataPtrColj[k];
                        }

                        ADataPtr[j * nA + idx] += m_Beta;

                        const auto xsignsj = xSignsPtr[j];

                        for (Index i = 0; i < mA; ++i)
                        {//Not auto vectorized
                            ADataPtr[i * nA + idx] *= xsignsj;
                        }
                    }
                }
                else
                {// TODO parallelize map for loop
                    for (const auto& [j, idx] : indicesMap)
                    {//Not auto vectorized
                        const auto MatDataPtrj = MatDataPtr + j * m;

                        for (Index k = 0; k < n; ++k)
                        {//Not auto vectorized
                            const auto MatDataPtrk = MatDataPtr + k * m;
                            auto value = static_cast<Scalar>(0);
                            for (Index l = 0; l < m; ++l)
                            {//Vectorized
                                value += MatDataPtrk[l] * MatDataPtrj[l];
                            }

                            ADataPtr[k * nA + idx] = value;
                        }

                        ADataPtr[j * nA + idx] += m_Beta;

                        const auto xsignsj = xSignsPtr[j];

                        for (Index i = 0; i < mA; ++i)
                        {//Not auto vectorized
                            ADataPtr[i * nA + idx] *= xsignsj;
                        }
                    }
                }

                //gamma.setZero();
                std::fill(std::execution::par,
                    gamma.begin(), gamma.end(), static_cast<Scalar>(0));// TODO optimize

                //b.setZero();
                std::fill(std::execution::par,
                    b.begin(), b.end(), static_cast<Scalar>(0));// TODO optimize

                //b.head(n) = VectData;// ATb or Qalpha
                std::copy(std::execution::par,
                    VectData.cbegin(), VectData.cend(), b.begin());// TODO optimize

                Index jT = 0;
                Index jW = 0;
                Index jS = 0;

                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gammaPtr[tIndex] = tau;

                        bPtr[tIndex] = solutionYaaydelta;

                        tIndicesMapPtr[i] = tIndex;

                        ++jT;

                        pivotsPtr[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gammaPtr[tIndex] = tau;

                        bPtr[tIndex] = solutionYaaydelta;

                        tIndicesMapPtr[i] = tIndex;

                        ++jT;

                        pivotsPtr[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gammaPtr[tIndex] = tau;

                        bPtr[tIndex] = solutionYaaydelta;

                        tIndicesMapPtr[i] = tIndex;

                        ++jT;

                        pivotsPtr[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gammaPtr[tIndex] = tau;

                        bPtr[tIndex] = solutionYaaydelta;

                        tIndicesMapPtr[i] = tIndex;

                        ++jT;

                        pivotsPtr[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        const auto tIndex = jStartSlacksT + jT;

                        gammaPtr[tIndex] = tau;

                        bPtr[tIndex] = solutionYaaydelta;

                        tIndicesMapPtr[i] = tIndex;

                        ++jT;

                        pivotsPtr[i] = indicesMap.at(i);

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        const auto sIndex = jStartSlacksS + jS;
                        const auto wIndex = jStartSlacksW + jW;

                        gammaPtr[wIndex] = tau;

                        bPtr[wIndex] = solutionYaaydelta;

                        //sIndicesMap[i] = sIndex;
                        wIndicesMapPtr[i] = wIndex;

                        ++jS;
                        ++jW;

                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        const auto sIndex = jStartSlacksS + jS;
                        const auto wIndex = jStartSlacksW + jW;

                        gammaPtr[wIndex] = tau;

                        bPtr[wIndex] = solutionYaaydelta;

                        //sIndicesMap[i] = sIndex;
                        wIndicesMapPtr[i] = wIndex;

                        ++jS;
                        ++jW;

                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        const auto sIndex = jStartSlacksS + jS;
                        const auto wIndex = jStartSlacksW + jW;

                        gammaPtr[wIndex] = tau;

                        bPtr[wIndex] = solutionYaaydelta;

                        //sIndicesMap[i] = sIndex;
                        wIndicesMapPtr[i] = wIndex;

                        ++jS;
                        ++jW;

                        break;
                    }
                    default:
                        std::cout << "\nNOT POSSIBLE\n";
                        break;
                    }
                }

#pragma omp parallel for // TODO is it usefull??
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
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
                        const auto tIndex = tIndicesMapPtr[i];

                        const auto betaTimesSigni = m_Beta * xSignsPtr[i];

                        ADataPtr[i * nA + indicesMap.at(i)] -= betaTimesSigni;

                        gammaPtr[i] -= betaTimesSigni * gammaPtr[tIndex];

                        bPtr[i] -= betaTimesSigni * bPtr[tIndex];

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        const auto betaTimesSigni = m_Beta * xSignsPtr[i];

                        ADataPtr[i * nA + indicesMap.at(i)] -= betaTimesSigni;

                        gammaPtr[i] -= betaTimesSigni * gammaPtr[tIndex];

                        bPtr[i] -= betaTimesSigni * bPtr[tIndex];

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        const auto betaTimesSigni = m_Beta * xSignsPtr[i];

                        ADataPtr[i * nA + indicesMap.at(i)] -= betaTimesSigni;

                        gammaPtr[i] -= betaTimesSigni * gammaPtr[tIndex];

                        bPtr[i] -= betaTimesSigni * bPtr[tIndex];

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
                        const auto wIndex = wIndicesMapPtr[i];

                        const auto betaTimesSigni = m_Beta * xSignsPtr[i];

                        gammaPtr[i] -= betaTimesSigni * gammaPtr[wIndex];

                        bPtr[i] -= betaTimesSigni * bPtr[wIndex];

                        break;
                    }
                    default:
                        break;
                    }
                }

                // do not parallelize           
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto pivotRowIndex = i;
                    const auto pivotColumnIndex = pivotsPtr[i];

                    if (pivotColumnIndex == static_cast<Index>(-1))
                    {
                        continue;
                    }

                    auto ADataPtrpivotRowIndex = ADataPtr + pivotRowIndex * nA;
                    const auto pivotCoeff = ADataPtrpivotRowIndex[pivotColumnIndex];
#pragma omp simd
                    for (Index j = 0; j < nA; ++j)
                    {//Vectorized
                        ADataPtrpivotRowIndex[j] /= pivotCoeff;
                    }

                    gammaPtr[pivotRowIndex] /= pivotCoeff;

                    bPtr[pivotRowIndex] /= pivotCoeff;

#pragma omp parallel for
                    for (Index rowIndex = 0; rowIndex < n; ++rowIndex)
                    {//Not auto vectorized
                        if (pivotRowIndex != rowIndex)
                        {
                            auto ADataPtrrowIndex = ADataPtr + rowIndex * nA;
                            const auto value = ADataPtrrowIndex[pivotColumnIndex];
                            if (std::abs(value) > Utils::epsilon)
                            {
#pragma omp simd
                                for (Index j = 0; j < nA; ++j)
                                {//Vectorized
                                    ADataPtrrowIndex[j] -= value * ADataPtrpivotRowIndex[j];
                                }

                                gammaPtr[rowIndex] -= value * gammaPtr[pivotRowIndex];

                                bPtr[rowIndex] -= value * bPtr[pivotRowIndex];
                            }
                        }
                    }
                }

#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        gammaPtr[tIndex] -= gammaPtr[i];

                        bPtr[tIndex] -= bPtr[i];

                        gammaPtr[tIndex] *= static_cast<Scalar>(-1);

                        bPtr[tIndex] *= static_cast<Scalar>(-1);

                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        gammaPtr[tIndex] -= gammaPtr[i];

                        bPtr[tIndex] -= bPtr[i];

                        gammaPtr[tIndex] *= static_cast<Scalar>(-1);

                        bPtr[tIndex] *= static_cast<Scalar>(-1);

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        gammaPtr[tIndex] -= gammaPtr[i];

                        bPtr[tIndex] -= bPtr[i];

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        gammaPtr[tIndex] -= gammaPtr[i];

                        bPtr[tIndex] -= bPtr[i];

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        gammaPtr[tIndex] -= gammaPtr[i];

                        bPtr[tIndex] -= bPtr[i];

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        const auto wIndex = wIndicesMapPtr[i];

                        const auto signiOverBeta = xSignsPtr[i] / m_Beta;

                        gammaPtr[i] *= signiOverBeta;

                        bPtr[i] *= signiOverBeta;

                        gammaPtr[wIndex] -= gammaPtr[i];

                        bPtr[wIndex] -= bPtr[i];

                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        const auto wIndex = wIndicesMapPtr[i];

                        const auto signiOverBeta = xSignsPtr[i] / m_Beta;

                        gammaPtr[i] *= signiOverBeta;

                        bPtr[i] *= signiOverBeta;

                        gammaPtr[wIndex] -= gammaPtr[i];

                        bPtr[wIndex] -= bPtr[i];

                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        const auto wIndex = wIndicesMapPtr[i];

                        const auto minusSigniOver2Beta = -xSignsPtr[i] / (static_cast<Scalar>(2) * m_Beta);

                        gammaPtr[i] *= minusSigniOver2Beta;

                        bPtr[i] *= minusSigniOver2Beta;

                        gammaPtr[wIndex] -= gammaPtr[i];

                        bPtr[wIndex] -= bPtr[i];

                        break;
                    }
                    default:
                        break;
                    }
                }

                bool bHasZeros = false;
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K1:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        if (bPtr[tIndex] <= Utils::epsilon && gammaPtr[tIndex] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K2;

                            bHasZeros = true;
                        }

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        const auto tIndex = tIndicesMapPtr[i];

                        if (bPtr[tIndex] <= Utils::epsilon && gammaPtr[tIndex] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K1;

                            bHasZeros = true;
                        }

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        if (bPtr[i] <= Utils::epsilon && gammaPtr[i] > static_cast<Scalar>(0))
                        {
                            indicesPtr[i] = ConstraintsType::K5;

                            bHasZeros = true;
                        }

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        const auto wIndex = wIndicesMapPtr[i];

                        if (bPtr[wIndex] <= Utils::epsilon && gammaPtr[wIndex] > static_cast<Scalar>(0))
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

                m_NbWs = computeNbWs(numberOfColumns, indices);
                m_NbTs = n - m_NbWs;
            }

            //[m_NumberOfConstraints, m_NbTs, m_NbWs] = computeNbWs(numberOfColumns, indices);

            Solution solutionNew{ n };

            // extract solution
            auto minimum = std::numeric_limits<Scalar>::max();
            auto aPivotRowIndex = static_cast<Index>(-1);
            {
                for (Index i = 0; i < m_NumberOfConstraints; ++i)
                {//Not auto vectorized
                    const auto value = gammaPtr[i];// m_PositiveValues[i];
                    if (value > Utils::epsilon)//&& std::abs(m_B.get()[i]) > Utils::epsilon)
                    {
                        const auto itemValue = bPtr[i] / value;
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
                const auto gammaSol = bPtr[aPivotRowIndex] / gammaPtr[aPivotRowIndex];

                solutionNew.delta = solutionYaaydelta - tau * gammaSol;

                auto solutionNewxPtr = solutionNew.x.data();
                auto solutionNewgradPtr = solutionNew.grad.data();
#pragma omp parallel for
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    const auto idxType = indicesPtr[i];

                    switch (idxType)
                    {
                    case ConstraintsType::K0:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewxPtr[i] = bPtr[i] - gammaPtr[i] * gammaSol;
                        }

                        break;
                    }
                    case ConstraintsType::K1:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewxPtr[i] = bPtr[i] - gammaPtr[i] * gammaSol;
                        }

                        break;
                    }
                    case ConstraintsType::K2:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewxPtr[i] = bPtr[i] - gammaPtr[i] * gammaSol;
                        }

                        const auto tIndex = tIndicesMapPtr[i];
                        if (tIndex != aPivotRowIndex)
                        {
                            const auto Ti = bPtr[tIndex] - gammaPtr[tIndex] * gammaSol;

                            solutionNewgradPtr[i] = -m_Beta * xSignsPtr[i] * Ti;
                        }

                        break;
                    }
                    case ConstraintsType::K3:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewxPtr[i] = bPtr[i] - gammaPtr[i] * gammaSol;
                        }

                        const auto tIndex = tIndicesMapPtr[i];
                        if (tIndex != aPivotRowIndex)
                        {
                            const auto Ti = bPtr[tIndex] - gammaPtr[tIndex] * gammaSol;

                            solutionNewgradPtr[i] = -m_Beta * xSignsPtr[i] * Ti;
                        }

                        break;
                    }
                    case ConstraintsType::K4:
                    {
                        if (i != aPivotRowIndex)
                        {
                            solutionNewxPtr[i] = bPtr[i] - gammaPtr[i] * gammaSol;
                        }

                        const auto tIndex = tIndicesMapPtr[i];
                        if (tIndex != aPivotRowIndex)
                        {
                            const auto Ti = bPtr[tIndex] - gammaPtr[tIndex] * gammaSol;

                            solutionNewgradPtr[i] = -m_Beta * xSignsPtr[i] * Ti;
                        }

                        break;
                    }
                    case ConstraintsType::K5:
                    {
                        if (i != aPivotRowIndex)
                        {
                            //const auto sIndex = sIndicesMap.at(i);
                            const auto Si = bPtr[i] - gammaPtr[i] * gammaSol;

                            solutionNewgradPtr[i] = -m_Beta * xSignsPtr[i] * Si;
                        }

                        break;
                    }
                    case ConstraintsType::K6:
                    {
                        if (i != aPivotRowIndex)
                        {
                            //const auto sIndex = sIndicesMap.at(i);
                            const auto Si = bPtr[i] - gammaPtr[i] * gammaSol;

                            solutionNewgradPtr[i] = -m_Beta * xSignsPtr[i] * Si;
                        }

                        break;
                    }
                    case ConstraintsType::K7:
                    {
                        if (i != aPivotRowIndex)
                        {
                            //const auto sIndex = sIndicesMap.at(i);
                            const auto Si = bPtr[i] - gammaPtr[i] * gammaSol;

                            solutionNewgradPtr[i] += m_Beta * xSignsPtr[i] * Si;
                        }

                        const auto wIndex = wIndicesMapPtr[i];
                        if (wIndex != aPivotRowIndex)
                        {
                            const auto Wi = bPtr[wIndex] - gammaPtr[wIndex] * gammaSol;

                            solutionNewgradPtr[i] -= m_Beta * xSignsPtr[i] * Wi;
                        }

                        break;
                    }
                    default:
                        break;
                    }
                }

                // update signs
#pragma omp simd
                for (Index i = 0; i < n; ++i)
                {//Vectorized
                    solutionNewxPtr[i] *= xSignsPtr[i];
                }

                return solutionNew;
            }
            else
            {
                std::cout << "\nEXceptional Case: unbounded LP.\n";
            }

            return Solution{};
        }

        template<std::floating_point ScalarType>
        inline std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
            FullPathSolver<ScalarType>::fitAll(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
                bool matrixIsCovariance,
                Scalar beta,
                Strategy strategy)
        {
            return fittAll(matData, numberOfRows, numberOfColumns,
                vectData.data(), matrixIsCovariance, beta, strategy);
        }

        template<std::floating_point ScalarType>
        std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
            FullPathSolver<ScalarType>::fittAll(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance,
                Scalar beta,
                Strategy strategy)
        {
            using Solution = Solution<Scalar>;
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using Utils = Utils<Scalar>;

            FullPathSolver regressor{ Param{static_cast<Scalar>(-1), beta, strategy} };

            if (strategy != Strategy::FromBothSolutions)
            {
                return regressor.solve(
                    matData, numberOfRows, numberOfColumns,
                    vectDataPtr,
                    matrixIsCovariance,
                    strategy == Strategy::FromZeroSolution);
            }
            else
            {
                auto fromZeroFuture =
                    std::async(std::launch::async, &FullPathSolver::solve, &regressor,
                        matData, numberOfRows, numberOfColumns,
                        vectDataPtr,
                        matrixIsCovariance,
                        true);

                auto fromL2SolutionResults = regressor.solve(
                    matData, numberOfRows, numberOfColumns,
                    vectDataPtr,
                    matrixIsCovariance,
                    false);

                auto fromZeroSolutionResults = fromZeroFuture.get();

                for (auto it = fromL2SolutionResults.begin(); it != fromL2SolutionResults.end(); ++it)
                {//Not auto vectorized
                    if (it->delta < fromZeroSolutionResults.back().delta - Utils::epsilon)
                    {
                        fromZeroSolutionResults.push_back(std::move(*it));
                    }
                }

                return fromZeroSolutionResults;// move
            }
        }

        template<std::floating_point ScalarType>
        inline Solution<typename FullPathSolver<ScalarType>::Scalar>
            FullPathSolver<ScalarType>::fit(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const ContiguousDataContainer<Scalar>& vectData,
                bool matrixIsCovariance)
        {
            return fitt(matData, numberOfRows, numberOfColumns, vectData.data(), matrixIsCovariance);
        }

        template<std::floating_point ScalarType>
        Solution<typename FullPathSolver<ScalarType>::Scalar>
            FullPathSolver<ScalarType>::fitt(
                const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance)
        {
            using Solution = Solution<Scalar>;
            using Utils = Utils<Scalar>;

            if (m_Param.delta < static_cast<Scalar>(0))
            {
                // Should throw an exception!!!
                std::cout << "\nNegative deltas not allowed: delta == " << m_Param.delta << "\n";
                return Solution{};
            }

            Solution minBoundSolution{};
            Solution maxBoundSolution{};

            if (m_Param.strategy != Strategy::FromBothSolutions)
            {
                auto results = solve(matData, numberOfRows, numberOfColumns, vectDataPtr, matrixIsCovariance,
                    m_Param.strategy == Strategy::FromZeroSolution);
                if (!results.empty())
                {
                    auto minBoundIt = results.begin();

                    minBoundSolution = std::move(*minBoundIt);

                    maxBoundSolution = std::move(*++minBoundIt);
                }
            }
            else
            {
                auto fromZeroFuture =
                    std::async(std::launch::async, &FullPathSolver::solve, this,
                        matData, numberOfRows, numberOfColumns,
                        vectDataPtr,
                        matrixIsCovariance,
                        true);

                auto fromL2SolutionResults = solve(
                    matData, numberOfRows, numberOfColumns,
                    vectDataPtr,
                    matrixIsCovariance,
                    false);

                auto fromZeroSolutionResults = fromZeroFuture.get();

                if (!fromZeroSolutionResults.empty() || !fromL2SolutionResults.empty())
                {
                    auto minBoundIt = fromZeroSolutionResults.empty()
                        ? fromL2SolutionResults.begin()
                        : fromZeroSolutionResults.begin();

                    minBoundSolution = std::move(*minBoundIt);

                    maxBoundSolution = std::move(*++minBoundIt);
                }
            }

            if (!minBoundSolution.x.empty() && !maxBoundSolution.x.empty())
            {
                // interval found. minBoundSolution.delta <= delta <= maxBoundSolution.delta
                auto& deltaMin = minBoundSolution.delta;
                const auto deltaMax = maxBoundSolution.delta;

                const auto maxBoundSolutionxPtr = maxBoundSolution.x.data();
                const auto maxBoundSolutiongradPtr = maxBoundSolution.grad.data();

                auto minBoundSolutionxPtr = minBoundSolution.x.data();
                auto minBoundSolutiongradPtr = minBoundSolution.grad.data();

                const auto diff = deltaMax - deltaMin;
                const auto alpha = (diff > Utils::epsilon)
                    ? (m_Param.delta - deltaMin) / diff : static_cast<Scalar>(0);

                const auto oneMinusAlpha = static_cast<Scalar>(1) - alpha;

                const auto n = static_cast<Index>(minBoundSolution.x.size());

                deltaMin *= oneMinusAlpha;
                deltaMin += alpha * deltaMax;
#pragma omp simd
                for (Index i = 0; i < n; ++i)
                {//Vectorized
                    minBoundSolutionxPtr[i] = oneMinusAlpha * minBoundSolutionxPtr[i]
                        + alpha * maxBoundSolutionxPtr[i];
                    minBoundSolutiongradPtr[i] = oneMinusAlpha * minBoundSolutiongradPtr[i]
                        + alpha * maxBoundSolutiongradPtr[i];
                }

                return minBoundSolution;
            }

            // Should throw an exception!!!
            std::cout << "\nBad situation: delta == " << m_Param.delta << "\n";
            return Solution{};
        }

        template<std::floating_point ScalarType>
        std::list<Solution<typename FullPathSolver<ScalarType>::Scalar>>
            FullPathSolver<ScalarType>::solve(const ContiguousDataContainer<Scalar>& matData/*colmajor*/,
                Index numberOfRows, Index numberOfColumns,
                const Scalar* vectDataPtr,
                bool matrixIsCovariance, bool fromZeroSolution)
        {
            using ContiguousDataContainer = ContiguousDataContainer<Scalar>;
            using Solution = Solution<Scalar>;
            using Utils = Utils<Scalar>;
            using L2Regressor = L2RegressorGauss<Scalar>;
            using FullPathStep = FullPathStep<Scalar>;

            const auto n = numberOfColumns;
            const auto m = numberOfRows;

            const auto tau = fromZeroSolution ? static_cast<Scalar>(1) : static_cast<Scalar>(-1);

            std::list<Solution> results;

            const auto ADataPtr = matData.data();

            const auto bDataPtr = vectDataPtr;// vectData.data();

            //const auto ATb = m_Mat.transpose(vectData);// same as Q\alpha
            ContiguousDataContainer ATb(n);
            auto ATbPtr = ATb.data();
#pragma omp parallel for
            for (Index j = 0; j < n; ++j)
            {//Not auto vectorized
                const auto ADataPtrColj = ADataPtr + j * m;
                //ATbPtr[j] = static_cast<Scalar>(0);
                auto total = static_cast<Scalar>(0);
#pragma omp simd
                for (Index i = 0; i < m; ++i)
                {//Vectorized
                    //ATbPtr[j] += bDataPtr[i] * ADataPtrColj[i];
                    total += bDataPtr[i] * ADataPtrColj[i];
                }

                ATbPtr[j] = total;
            }

            if (fromZeroSolution)
            {
                const auto deltaZero = lpNormInfinity(ATb) / m_Param.beta;//ATb.lpNormInfinity() / m_Beta;

                results.emplace_back(std::numeric_limits<Scalar>::max(),//static_cast<Scalar>(2) * deltaZero,
                    ContiguousDataContainer(n, static_cast<Scalar>(0)),
                    opposite(ATb));

                results.emplace_back(deltaZero,
                    results.back().x,
                    results.back().grad);

                bool leave = false;
                bool iamOnTarget = false;
                bool otherOnTarget = false;

                if (m_Param.strategy == Strategy::FromBothSolutions)
                {
                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                    m_DeltaFromZeroSolution = deltaZero;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                        {
                            iamOnTarget = true;
                        }
                        else if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                        {
                            otherOnTarget = true;
                        }
                    }

                    if (!iamOnTarget && !otherOnTarget
                        && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)// TODO maybe use std option?
                    {
                        leave = true;
                    }
                }
                else
                {
                    m_DeltaFromZeroSolution = deltaZero;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                        {
                            iamOnTarget = true;
                        }
                    }

                    if (!iamOnTarget && m_DeltaFromZeroSolution <= Utils::epsilon)// TODO maybe use std option?
                    {
                        leave = true;
                    }
                }

                if (iamOnTarget)
                {
                    auto minBoundIt = results.crbegin();

                    const auto& minBoundSolution = *minBoundIt;

                    const auto& maxBoundSolution = *++minBoundIt;


                    return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                }
                else if (otherOnTarget)
                {
                    return std::list<Solution>{};
                }
                else if (leave)
                {
                    return results;
                }
            }
            else
            {
                Solution solutionBar{ n };

                solutionBar.x = L2Regressor(m_Param.beta).fit(matData, m, n, vectDataPtr, matrixIsCovariance);

                solutionBar.delta = std::numeric_limits<Scalar>::max();
                for (Index i = 0; i < n; ++i)
                {//Not auto vectorized
                    // TODO optimize
                    const auto absWeight = std::abs(solutionBar.x[i]);
                    if (Utils::epsilon < absWeight && absWeight < solutionBar.delta)
                    {
                        solutionBar.delta = absWeight;
                    }
                }

                results.emplace_front(static_cast<Scalar>(0),// solutionBar.delta / static_cast<Scalar>(2),
                    solutionBar.x,
                    solutionBar.grad);

                results.push_front(std::move(solutionBar));

                bool leave = false;
                bool iamOnTarget = false;
                bool otherOnTarget = false;
                if (m_Param.strategy == Strategy::FromBothSolutions)
                {
                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                    m_DeltaFromL2Solution = solutionBar.delta;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                        {
                            iamOnTarget = true;
                        }
                        else if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                        {
                            otherOnTarget = true;
                        }
                    }

                    if (!iamOnTarget && !otherOnTarget
                        && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)// TODO maybe use std option?
                    {
                        leave = true;
                    }
                }
                else
                {
                    m_DeltaFromL2Solution = solutionBar.delta;
                    if (m_Param.delta > static_cast<Scalar>(0))
                    {
                        if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                        {
                            iamOnTarget = true;
                        }
                    }

                    if (!iamOnTarget &&
                        std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution)// TODO use delta of zero
                    {
                        leave = true;
                    }
                }

                if (iamOnTarget)
                {
                    auto maxBoundIt = results.cbegin();

                    const auto& maxBoundSolution = *maxBoundIt;

                    const auto& minBoundSolution = *++maxBoundIt;

                    return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                }
                else if (otherOnTarget)
                {
                    return std::list<Solution>{};
                }
                else if (leave)
                {
                    return results;
                }
            }

            unsigned int numberIters = 0;

            while (true)
            {
                ++numberIters;

                {
                    const auto& solutionMaam = fromZeroSolution ? *(++results.crbegin()) : *(++results.cbegin());

                    const auto& solutionYaay = fromZeroSolution ? results.back() : results.front();

                    const auto step = FullPathStep(m_Param.beta);

                    auto solutionNew = step.run(matData, m, n,
                        ATb,
                        solutionYaay,
                        solutionMaam,
                        matrixIsCovariance,
                        tau);

                    const auto gammaNew = tau * (solutionYaay.delta - solutionNew.delta);

                    if (gammaNew <= Utils::epsilon)
                    {
                        std::cout << "\nSTART CYCLING!\n";

                        std::cout << std::fixed << std::setprecision(15);
                        std::cout << "\nGamma: " << gammaNew
                            << " is less than " << Utils::epsilon << "\n";

                        break;
                    }

                    bool leave = false;
                    bool iamOnTarget = false;
                    bool otherOnTarget = false;
                    if (m_Param.strategy == Strategy::FromBothSolutions)
                    {
                        const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                        if (fromZeroSolution)
                        {
                            m_DeltaFromZeroSolution = solutionNew.delta;
                        }
                        else
                        {
                            m_DeltaFromL2Solution = solutionNew.delta;
                        }

                        if (m_Param.delta > static_cast<Scalar>(0))
                        {
                            if (fromZeroSolution)
                            {
                                if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                                {
                                    iamOnTarget = true;
                                }
                                else if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                {
                                    otherOnTarget = true;
                                }
                            }
                            else
                            {
                                if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                {
                                    iamOnTarget = true;
                                }
                                else if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                                {
                                    otherOnTarget = true;
                                }
                            }
                        }

                        if (!iamOnTarget && !otherOnTarget
                            && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)// TODO maybe use std option?
                        {
                            leave = true;
                        }
                    }
                    else
                    {
                        if (fromZeroSolution)
                        {
                            m_DeltaFromZeroSolution = solutionNew.delta;
                        }
                        else
                        {
                            m_DeltaFromL2Solution = solutionNew.delta;
                        }

                        if (m_Param.delta > static_cast<Scalar>(0))
                        {
                            if (fromZeroSolution)
                            {
                                if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                                {
                                    iamOnTarget = true;
                                }
                            }
                            else
                            {
                                if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                {
                                    iamOnTarget = true;
                                }
                            }
                        }

                        if (fromZeroSolution)
                        {
                            if (!iamOnTarget && m_DeltaFromZeroSolution <= Utils::epsilon)
                            {
                                leave = true;
                            }
                        }
                        else
                        {
                            if (!iamOnTarget
                                && std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution)
                            {
                                leave = true;
                            }
                        }
                    }

                    if (fromZeroSolution)
                    {
                        results.push_back(std::move(solutionNew));

                        if (iamOnTarget)
                        {
                            auto minBoundIt = results.crbegin();

                            const auto& minBoundSolution = *minBoundIt;

                            const auto& maxBoundSolution = *++minBoundIt;


                            return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                        }
                        else if (otherOnTarget)
                        {
                            return std::list<Solution>{};
                        }
                    }
                    else
                    {
                        results.push_front(std::move(solutionNew));

                        if (iamOnTarget)
                        {
                            auto maxBoundIt = results.cbegin();

                            const auto& maxBoundSolution = *maxBoundIt;

                            const auto& minBoundSolution = *++maxBoundIt;

                            return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                        }
                        else if (otherOnTarget)
                        {
                            return std::list<Solution>{};
                        }
                    }

                    if (leave)
                    {
                        return results;
                    }
                }

                const auto& solutionLast = fromZeroSolution ? results.back() : results.front();

                {
                    const auto fullPathDone = (!fromZeroSolution) && norm(solutionLast.x) <= Utils::epsilon
                        || fromZeroSolution && componentwiseAbsMinCoeff(solutionLast.x) >= solutionLast.delta - Utils::epsilon;

                    if (fullPathDone)
                    {
                        return results;
                    }
                }

                {
                    auto normMax = std::numeric_limits<Scalar>::max();
                    for (Index i = 0; i < n; ++i)
                    {//Not auto vectorized
                        // TODO optimize
                        const auto absXi = std::abs(solutionLast.x[i]);
                        if (Utils::epsilon < absXi && absXi < normMax)
                        {
                            normMax = absXi;
                        }
                    }

                    if (normMax < std::numeric_limits<Scalar>::max())
                    {
                        if (fromZeroSolution && normMax >= solutionLast.delta - Utils::epsilon)
                        {// constant for delta in [deltaCandidate, delta]. improve delta

                            auto absGradMax = std::numeric_limits<Scalar>::min();
                            for (Index i = 0; i < n; ++i)
                            {//Not auto vectorized
                                // TODO optimize
                                const auto absXi = std::abs(solutionLast.x[i]);
                                if (absXi <= Utils::epsilon)
                                {
                                    const auto absGradi = std::abs(solutionLast.grad[i]);
                                    if (absGradi > absGradMax)
                                    {
                                        absGradMax = absGradi;
                                    }
                                }
                            }

                            if (absGradMax > std::numeric_limits<Scalar>::min()
                                && absGradMax / m_Param.beta < solutionLast.delta - Utils::epsilon)
                            {
                                const auto deltaCandidate = absGradMax / m_Param.beta;

                                bool leave = false;
                                bool iamOnTarget = false;
                                bool otherOnTarget = false;
                                if (m_Param.strategy == Strategy::FromBothSolutions)
                                {
                                    const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                                    m_DeltaFromZeroSolution = deltaCandidate;

                                    if (m_Param.delta > static_cast<Scalar>(0))
                                    {
                                        if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                                        {
                                            iamOnTarget = true;
                                        }
                                        else if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                        {
                                            otherOnTarget = true;
                                        }
                                    }

                                    if (!iamOnTarget && !otherOnTarget
                                        && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                    {
                                        leave = true;
                                    }
                                }
                                else
                                {
                                    m_DeltaFromZeroSolution = deltaCandidate;

                                    if (m_Param.delta > static_cast<Scalar>(0))
                                    {
                                        if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                                        {
                                            iamOnTarget = true;
                                        }
                                    }

                                    if (!iamOnTarget
                                        && m_DeltaFromZeroSolution <= Utils::epsilon)// TODO maybe use std option?
                                    {
                                        leave = true;
                                    }
                                }

                                results.emplace_back(deltaCandidate, solutionLast.x, solutionLast.grad);

                                if (iamOnTarget)
                                {
                                    auto minBoundIt = results.crbegin();

                                    const auto& minBoundSolution = *minBoundIt;

                                    const auto& maxBoundSolution = *++minBoundIt;


                                    return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                                }
                                else if (otherOnTarget)
                                {
                                    return std::list<Solution>{};
                                }
                                else if (leave)
                                {
                                    return results;
                                }

                                // TODO try to optimize using minmax and avoid abs
                                const auto fullPathDone = componentwiseAbsMinCoeff(results.back().x)// cwiseAbs().minCoeff()
                                    >= results.back().delta - Utils::epsilon;

                                if (fullPathDone)
                                {
                                    return results;
                                }
                            }
                        }
                        else if (!fromZeroSolution && normMax > solutionLast.delta + Utils::epsilon)
                        {// constant for delta in [delta, normMax]. improve delta

                            const auto deltaCandidate = normMax;

                            bool leave = false;
                            bool iamOnTarget = false;
                            bool otherOnTarget = false;

                            if (m_Param.strategy == Strategy::FromBothSolutions)
                            {
                                const std::lock_guard<std::mutex> lock(m_DeltasMutex);
                                m_DeltaFromL2Solution = deltaCandidate;

                                if (m_Param.delta > static_cast<Scalar>(0))
                                {
                                    if (m_Param.delta <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                    {
                                        iamOnTarget = true;
                                    }
                                    else if (m_DeltaFromZeroSolution <= m_Param.delta)// TODO maybe use std option?
                                    {
                                        otherOnTarget = true;
                                    }
                                }

                                if (!iamOnTarget && !otherOnTarget
                                    && m_DeltaFromZeroSolution <= m_DeltaFromL2Solution)// TODO maybe use std option?
                                {
                                    leave = true;
                                }
                            }
                            else
                            {
                                m_DeltaFromL2Solution = deltaCandidate;

                                if (m_Param.delta > static_cast<Scalar>(0))
                                {
                                    if (m_Param.delta <= m_DeltaFromL2Solution)
                                    {
                                        iamOnTarget = true;
                                    }
                                }

                                if (!iamOnTarget
                                    && std::numeric_limits<Scalar>::max() <= m_DeltaFromL2Solution)
                                {
                                    leave = true;
                                }
                            }

                            results.emplace_front(deltaCandidate, solutionLast.x, solutionLast.grad);

                            if (iamOnTarget)
                            {
                                auto maxBoundIt = results.cbegin();

                                const auto& maxBoundSolution = *maxBoundIt;

                                const auto& minBoundSolution = *++maxBoundIt;

                                return std::list<Solution>{ minBoundSolution, maxBoundSolution };
                            }
                            else if (otherOnTarget)
                            {
                                return std::list<Solution>{};
                            }
                            else if (leave)
                            {
                                return results;
                            }

                            const auto fullPathDone = norm(results.front().x) <= Utils::epsilon;

                            if (fullPathDone)
                            {
                                return results;
                            }
                        }
                    }
                }

                // TODO to remove. just for debug
                //if (numberIters == 1)
                //{
                //   break;
                //}
            }

            return results;
        }
    }
}
#endif //L0L2_FULL_PATH_SOLVER_H
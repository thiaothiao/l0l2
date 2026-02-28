#include <ios>

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/attr.h>
#include <pybind11/stl.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>

#include "Utils.h"
#include "FullPathSolver.h"
#include "CoordinateDescentSolver.h"
#include "SparsePCA.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

template class l0l2::linearmodel::CyclicalCoordinateDescent<l0l2::linearmodel::L0L2ModelImplementation<float>>;
template class l0l2::linearmodel::FullPathSolver<float>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<float>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<float>>;
template<>
const std::streamsize l0l2::linearmodel::Solution<float>::streamSize = 7;
template<>
const float l0l2::linearmodel::Utils<float>::epsilon = 1e-6f;
template<>
const std::streamsize l0l2::linearmodel::CDSolution<float>::streamSize = 7;

template class l0l2::linearmodel::CyclicalCoordinateDescent<l0l2::linearmodel::L0L2ModelImplementation<double>>;
template class l0l2::linearmodel::FullPathSolver<double>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<double>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<double>>;
template<>
const std::streamsize l0l2::linearmodel::Solution<double>::streamSize = 9;
template<>
const double l0l2::linearmodel::Utils<double>::epsilon = 1e-8;
template<>
const std::streamsize l0l2::linearmodel::CDSolution<double>::streamSize = 9;

namespace
{
    using CDStatus = l0l2::linearmodel::CDStatus;
    using Strategy = l0l2::linearmodel::Strategy;
    using Index = l0l2::linearmodel::Index;

    using FullPathSolverf = l0l2::linearmodel::FullPathSolver<float>;
    using FullPathSolverParamf = FullPathSolverf::Param;
    using Matrixf = l0l2::linearmodel::Matrix<float>;
    using Vectorf = l0l2::linearmodel::Vector<float>;
    using Solutionf = l0l2::linearmodel::Solution<float>;
    using L0L2Regressorf = l0l2::linearmodel::L0L2Regressor<float>;
    using L0L2RegressorParamf = L0L2Regressorf::Param;
    using CDSolutionf = l0l2::linearmodel::CDSolution<float>;
    using Utilsf = l0l2::linearmodel::Utils<float>;
    using L0L2SPCAf = l0l2::linearmodel::L0L2SPCA<float>;
    using L0L2SPCAParamf = L0L2SPCAf::Param;
    using FullPathL0L2SPCAf = l0l2::linearmodel::FullPathL0L2SPCA<float>;
    using FullPathL0L2SPCAParamf = FullPathL0L2SPCAf::Param;

    using FullPathSolverd = l0l2::linearmodel::FullPathSolver<double>;
    using FullPathSolverParamd = FullPathSolverd::Param;
    using Matrixd = l0l2::linearmodel::Matrix<double>;
    using Vectord = l0l2::linearmodel::Vector<double>;
    using Solutiond = l0l2::linearmodel::Solution<double>;
    using L0L2Regressord = l0l2::linearmodel::L0L2Regressor<double>;
    using L0L2RegressorParamd = L0L2Regressord::Param;
    using CDSolutiond = l0l2::linearmodel::CDSolution<double>;
    using Utilsd = l0l2::linearmodel::Utils<double>;
    using L0L2SPCAd = l0l2::linearmodel::L0L2SPCA<double>;
    using L0L2SPCAParamd = L0L2SPCAd::Param;
    using FullPathL0L2SPCAd = l0l2::linearmodel::FullPathL0L2SPCA<double>;
    using FullPathL0L2SPCAParamd = FullPathL0L2SPCAd::Param;
}

PYBIND11_MODULE(l0l2, mainmodule)
{
    mainmodule.doc() = "l0l2 sparse modeling module.";

    auto m = mainmodule.def_submodule("linearmodel", "Linear model module.");

    pybind11::enum_<CDStatus>(m, "CDStatus", pybind11::arithmetic(), "Convergence status")
        .value("Converged", CDStatus::Converged, "Converged status")
        .value("LimitReached", CDStatus::LimitReached, "LimitReached status")
        .value("Unknown", CDStatus::Unknown, "Notfitted status")
        .export_values();

    pybind11::enum_<Strategy>(m, "Strategy", pybind11::arithmetic(), "Choosen strategy")
        .value("FromZeroSolution", Strategy::FromZeroSolution, "From zero solution")
        .value("FromL2Solution", Strategy::FromL2Solution, "From ridge solution")
        .value("FromBothSolutions", Strategy::FromBothSolutions, "From both sides")
        .export_values();

    pybind11::class_<FullPathSolverParamf>(m, "FullPathSolverParamf")
        .def(pybind11::init<float, float, Strategy>(),
            pybind11::arg("delta") = 0.f,
            pybind11::arg("beta") = 1.f,
            pybind11::arg("strategy") = Strategy::FromZeroSolution)
        .def_readonly("delta", &FullPathSolverParamf::delta)
        .def_readonly("beta", &FullPathSolverParamf::beta)
        .def_readonly("strategy", &FullPathSolverParamf::strategy);

    pybind11::class_<FullPathSolverf>(m, "FullPathSolverf")
        .def(pybind11::init<FullPathSolverParamf, bool>(), 
            pybind11::arg("param"), 
            pybind11::arg("withintercept") = false)
        .def("fit", &FullPathSolverf::fit, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data on one delta value.")
        .def_static("fitAll", &FullPathSolverf::fitAll, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("beta"),
            pybind11::arg("withintercept") = false,
            pybind11::arg("strategy") = Strategy::FromZeroSolution, "A member function that fits data on all path.");

    pybind11::class_<Solutionf>(m, "Solutionf")
        .def(pybind11::init<Index>())
        .def(pybind11::init<float, Vectorf, Vectorf>())
        .def_readwrite("delta", &Solutionf::delta)
        .def_readwrite("x", &Solutionf::x)
        .def_readwrite("grad", &Solutionf::grad)
        .def_readwrite("intercept", &Solutionf::intercept)
        .def("isValidFor", &Solutionf::isValidFor,
            pybind11::arg("beta"), "A member function that checks solution validity.")
        .def("__str__", &Solutionf::toString);

    pybind11::class_<L0L2RegressorParamf>(m, "L0L2RegressorParamf")
        .def(pybind11::init<float, float, Strategy, float, unsigned int, float, unsigned int>(),
            pybind11::arg("delta") = 0.f,
            pybind11::arg("beta") = 1.f,
            pybind11::arg("strategy") = Strategy::FromZeroSolution,
            pybind11::arg("tolerance") = 1e-4f,
            pybind11::arg("maximumNumberOfIterations") = 10000U, 
            pybind11::arg("innerEpsilon") = 1e-6f,
            pybind11::arg("innerMaximumNumberOfIterations") = 100000U)
        .def_readonly("delta", &L0L2RegressorParamf::delta)
        .def_readonly("beta", &L0L2RegressorParamf::beta)
        .def_readonly("strategy", &L0L2RegressorParamf::strategy)
        .def_readonly("tolerance", &L0L2RegressorParamf::tolerance)
        .def_readonly("maximumNumberOfIterations", &L0L2RegressorParamf::maximumNumberOfIterations)
        .def_readonly("innerEpsilon", &L0L2RegressorParamf::innerEpsilon)
        .def_readonly("innerMaximumNumberOfIterations", &L0L2RegressorParamf::innerMaximumNumberOfIterations);

    pybind11::class_<L0L2Regressorf>(m, "L0L2Regressorf")
        .def(pybind11::init<L0L2RegressorParamf, bool>(), 
            pybind11::arg("param"),
            pybind11::arg("withintercept") = false)
        .def("fit", &L0L2Regressorf::fit,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data on one delta value.");

    pybind11::class_<CDSolutionf>(m, "CDSolutionf")
        .def(pybind11::init<Index>(), pybind11::arg("n") = static_cast<Index>(0))
        .def_readwrite("numberOfIterations", &CDSolutionf::numberOfIterations)
        .def_readwrite("globalChange", &CDSolutionf::globalChange)
        .def_readwrite("dualityGap", &CDSolutionf::dualityGap)
        .def_readwrite("status", &CDSolutionf::status)
        .def_readwrite("x", &CDSolutionf::x)
        .def_readwrite("intercept", &CDSolutionf::intercept)
        .def("__str__", &CDSolutionf::toString);

    pybind11::class_<Utilsf>(m, "Utilsf")
        //.def(pybind11::init())
        .def_static("objectiveValue", &Utilsf::objectiveValue, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("beta"),
            pybind11::arg("delta"),
            pybind11::arg("x"), "A member function that fits data on all path.");

    pybind11::class_<L0L2SPCAParamf>(m, "L0L2SPCAParamf")
        .def(pybind11::init<float, float, Index, Strategy, float, unsigned int, float, unsigned int>(),
            pybind11::arg("regressorDelta") = 0.f,
            pybind11::arg("regressorBeta") = 1.f,
            pybind11::arg("nbComponents") = static_cast<Index>(2),
            pybind11::arg("regressorStrategy") = Strategy::FromZeroSolution,
            pybind11::arg("regressorTolerance") = 1e-4f,
            pybind11::arg("regressorMaximumNumberOfIterations") = 10000U,
            pybind11::arg("regressorInnerEpsilon") = 1e-6f,
            pybind11::arg("regressorInnerMaximumNumberOfIterations") = 100000U)
        .def_readonly("regressorParam", &L0L2SPCAParamf::regressorParam)
        .def_readonly("nbComponents", &L0L2SPCAParamf::nbComponents);

    pybind11::class_<L0L2SPCAf>(m, "L0L2SPCAf")
        .def(pybind11::init<L0L2SPCAParamf, unsigned int, float, unsigned int>(),
            pybind11::arg("param"),
            pybind11::arg("nbJobs") = 1U,
            pybind11::arg("epsilon") = 1e-5f,
            pybind11::arg("numberOfTrialsMax") = 10000U)
        .def("run", &L0L2SPCAf::run,
            pybind11::arg("matData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data.");

    pybind11::class_<FullPathL0L2SPCAParamf>(m, "FullPathL0L2SPCAParamf")
        .def(pybind11::init<float, float, Index, Strategy>(),
            pybind11::arg("regressorDelta") = 0.f,
            pybind11::arg("regressorBeta") = 1.f,
            pybind11::arg("nbComponents") = static_cast<Index>(2),
            pybind11::arg("regressorStrategy") = Strategy::FromZeroSolution)
        .def_readonly("regressorParam", &FullPathL0L2SPCAParamf::regressorParam)
        .def_readonly("nbComponents", &FullPathL0L2SPCAParamf::nbComponents);

    pybind11::class_<FullPathL0L2SPCAf>(m, "FullPathL0L2SPCAf")
        .def(pybind11::init<FullPathL0L2SPCAParamf, unsigned int, float, unsigned int>(),
            pybind11::arg("param"),
            pybind11::arg("nbJobs") = 1U,
            pybind11::arg("epsilon") = 1e-5f,
            pybind11::arg("numberOfTrialsMax") = 10000U)
        .def("run", &FullPathL0L2SPCAf::run,
            pybind11::arg("matData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data.");


    pybind11::class_<FullPathSolverParamd>(m, "FullPathSolverParamd")
        .def(pybind11::init<double, double, Strategy>(),
            pybind11::arg("delta") = 0.0,
            pybind11::arg("beta") = 1.0,
            pybind11::arg("strategy") = Strategy::FromZeroSolution)
        .def_readonly("delta", &FullPathSolverParamd::delta)
        .def_readonly("beta", &FullPathSolverParamd::beta)
        .def_readonly("strategy", &FullPathSolverParamd::strategy);

    pybind11::class_<FullPathSolverd>(m, "FullPathSolverd")
        .def(pybind11::init<FullPathSolverParamd, bool>(), 
            pybind11::arg("param"),
            pybind11::arg("withintercept") = false)
        .def("fit", &FullPathSolverd::fit, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data on one delta value.")
        .def_static("fitAll", &FullPathSolverd::fitAll, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("beta"),
            pybind11::arg("withintercept") = false,
            pybind11::arg("strategy") = Strategy::FromZeroSolution, "A member function that fits data on all path.");

    pybind11::class_<Solutiond>(m, "Solutiond")
        .def(pybind11::init<Index>())
        .def(pybind11::init<double, Vectord, Vectord>())
        .def_readwrite("delta", &Solutiond::delta)
        .def_readwrite("x", &Solutiond::x)
        .def_readwrite("grad", &Solutiond::grad)
        .def_readwrite("intercept", &Solutiond::intercept)
        .def("isValidFor", &Solutiond::isValidFor,
            pybind11::arg("beta"), "A member function that checks solution validity.")
        .def("__str__", &Solutiond::toString);

    pybind11::class_<L0L2RegressorParamd>(m, "L0L2RegressorParamd")
        .def(pybind11::init<double, double, Strategy, double, unsigned int, double, unsigned int>(),
            pybind11::arg("delta") = 0.0,
            pybind11::arg("beta") = 1.0,
            pybind11::arg("strategy") = Strategy::FromZeroSolution,
            pybind11::arg("tolerance") = 1e-4,
            pybind11::arg("maximumNumberOfIterations") = 10000U,
            pybind11::arg("innerEpsilon") = 1e-6,
            pybind11::arg("innerMaximumNumberOfIterations") = 100000U)
        .def_readonly("delta", &L0L2RegressorParamd::delta)
        .def_readonly("beta", &L0L2RegressorParamd::beta)
        .def_readonly("strategy", &L0L2RegressorParamd::strategy)
        .def_readonly("tolerance", &L0L2RegressorParamd::tolerance)
        .def_readonly("maximumNumberOfIterations", &L0L2RegressorParamd::maximumNumberOfIterations)
        .def_readonly("innerEpsilon", &L0L2RegressorParamd::innerEpsilon)
        .def_readonly("innerMaximumNumberOfIterations", &L0L2RegressorParamd::innerMaximumNumberOfIterations);

    pybind11::class_<L0L2Regressord>(m, "L0L2Regressord")
        .def(pybind11::init<L0L2RegressorParamd, bool>(), 
            pybind11::arg("param"),
            pybind11::arg("withintercept") = false)
        .def("fit", &L0L2Regressord::fit, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data on one delta value.");

    pybind11::class_<CDSolutiond>(m, "CDSolutiond")
        .def(pybind11::init<Index>(), pybind11::arg("n") = static_cast<Index>(0))
        .def_readwrite("numberOfIterations", &CDSolutiond::numberOfIterations)
        .def_readwrite("globalChange", &CDSolutiond::globalChange)
        .def_readwrite("dualityGap", &CDSolutiond::dualityGap)
        .def_readwrite("status", &CDSolutiond::status)
        .def_readwrite("x", &CDSolutiond::x)
        .def_readwrite("intercept", &CDSolutiond::intercept)
        .def("__str__", &CDSolutiond::toString);

    pybind11::class_<Utilsd>(m, "Utilsd")
        .def_static("objectiveValue", &Utilsd::objectiveValue,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("beta"),
            pybind11::arg("delta"),
            pybind11::arg("x"), "A member function that fits data on all path.");

    pybind11::class_<L0L2SPCAParamd>(m, "L0L2SPCAParamd")
        .def(pybind11::init<double, double, Index, Strategy, double, unsigned int, double, unsigned int>(),
            pybind11::arg("regressorDelta") = 0.0,
            pybind11::arg("regressorBeta") = 1.0,
            pybind11::arg("nbComponents") = static_cast<Index>(2),
            pybind11::arg("regressorStrategy") = Strategy::FromZeroSolution,
            pybind11::arg("regressorTolerance") = 1e-4,
            pybind11::arg("regressorMaximumNumberOfIterations") = 10000U,
            pybind11::arg("regressorInnerEpsilon") = 1e-6,
            pybind11::arg("regressorInnerMaximumNumberOfIterations") = 100000U)
        .def_readonly("regressorParam", &L0L2SPCAParamd::regressorParam)
        .def_readonly("nbComponents", &L0L2SPCAParamd::nbComponents);

    pybind11::class_<L0L2SPCAd>(m, "L0L2SPCAd")
        .def(pybind11::init<L0L2SPCAParamd, unsigned int, double, unsigned int>(),
            pybind11::arg("param"),
            pybind11::arg("nbJobs") = 1U,
            pybind11::arg("epsilon") = 1e-5,
            pybind11::arg("numberOfTrialsMax") = 10000U)
        .def("run", &L0L2SPCAd::run, 
            pybind11::arg("matData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data.");

    pybind11::class_<FullPathL0L2SPCAParamd>(m, "FullPathL0L2SPCAParamd")
        .def(pybind11::init<double, double, Index, Strategy>(),
            pybind11::arg("regressorDelta") = 0.0,
            pybind11::arg("regressorBeta") = 1.0,
            pybind11::arg("nbComponents") = static_cast<Index>(2),
            pybind11::arg("regressorStrategy") = Strategy::FromZeroSolution)
        .def_readonly("regressorParam", &FullPathL0L2SPCAParamd::regressorParam)
        .def_readonly("nbComponents", &FullPathL0L2SPCAParamd::nbComponents);

    pybind11::class_<FullPathL0L2SPCAd>(m, "FullPathL0L2SPCAd")
        .def(pybind11::init<FullPathL0L2SPCAParamd, unsigned int, double, unsigned int>(),
            pybind11::arg("param"),
            pybind11::arg("nbJobs") = 1U,
            pybind11::arg("epsilon") = 1e-5,
            pybind11::arg("numberOfTrialsMax") = 10000U)
        .def("run", &FullPathL0L2SPCAd::run,
            pybind11::arg("matData"),
            pybind11::arg("matrixIsCovariance"), "A member function that fits data.");

#ifdef VERSION_INFO
    mainmodule.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    mainmodule.attr("__version__") = "dev";
#endif
}

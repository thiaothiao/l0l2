#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/attr.h>
#include <pybind11/stl.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>

#include "l0l2/version.hpp"
#include "l0l2/core.hpp"

template class l0l2::linearmodel::leastsquares::CyclicCoordinateDescent<
    l0l2::linearmodel::leastsquares::L0L2ModelImplementation<float>>;
template class l0l2::linearmodel::leastsquares::FullPathSolver<float>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<float>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<float>>;

template class l0l2::linearmodel::leastsquares::CyclicCoordinateDescent<
    l0l2::linearmodel::leastsquares::L0L2ModelImplementation<double>>;
template class l0l2::linearmodel::leastsquares::FullPathSolver<double>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::L0L2SPCAModelImplementation<double>>;
template class l0l2::linearmodel::SPCA<l0l2::linearmodel::FullPathL0L2SPCAModelImplementation<double>>;

namespace
{
    using CoordinateState = l0l2::linearmodel::leastsquares::CoordinateState;
    using CoordinateStates = l0l2::linearmodel::leastsquares::CoordinateStates;
    using Strategy = l0l2::linearmodel::leastsquares::Strategy;
    using Index = l0l2::Index;

    using FullPathSolverf = l0l2::linearmodel::leastsquares::FullPathSolver<float>;
    using FullPathSolverParamf = FullPathSolverf::Param;
    using Matrixf = l0l2::Matrix<float>;
    using Vectorf = l0l2::Vector<float>;
    using Solutionf = l0l2::linearmodel::leastsquares::Solution<float>;
    using L0L2Regressorf = l0l2::linearmodel::leastsquares::L0L2Regressor<float>;
    using L0L2RegressorParamf = L0L2Regressorf::Param;
    using Utilsf = l0l2::Utils<float>;
    using L0L2SPCAf = l0l2::linearmodel::L0L2SPCA<float>;
    using L0L2SPCAParamf = L0L2SPCAf::Param;
    using FullPathL0L2SPCAf = l0l2::linearmodel::FullPathL0L2SPCA<float>;
    using FullPathL0L2SPCAParamf = FullPathL0L2SPCAf::Param;

    using FullPathSolverd = l0l2::linearmodel::leastsquares::FullPathSolver<double>;
    using FullPathSolverParamd = FullPathSolverd::Param;
    using Matrixd = l0l2::Matrix<double>;
    using Vectord = l0l2::Vector<double>;
    using Solutiond = l0l2::linearmodel::leastsquares::Solution<double>;
    using L0L2Regressord = l0l2::linearmodel::leastsquares::L0L2Regressor<double>;
    using L0L2RegressorParamd = L0L2Regressord::Param;
    using Utilsd = l0l2::Utils<double>;
    using L0L2SPCAd = l0l2::linearmodel::L0L2SPCA<double>;
    using L0L2SPCAParamd = L0L2SPCAd::Param;
    using FullPathL0L2SPCAd = l0l2::linearmodel::FullPathL0L2SPCA<double>;
    using FullPathL0L2SPCAParamd = FullPathL0L2SPCAd::Param;
}

PYBIND11_MODULE(l0l2, mainmodule)
{
    mainmodule.doc() = "l0l2 sparse modeling module.";

    auto m = mainmodule.def_submodule("linearmodel", "Linear model module.");

    pybind11::enum_<CoordinateState>(m, "CoordinateState", pybind11::arithmetic(), "State of a coordinate")
        .value("L0", CoordinateState::L0, "Concerned by l0")
        .value("FREE", CoordinateState::FREE, "Freed from l0")
        .value("ZERO", CoordinateState::ZERO, "Forced to zero")
        .export_values();

    pybind11::enum_<Strategy>(m, "Strategy", pybind11::arithmetic(), "Choosen strategy")
        .value("FromZeroSolution", Strategy::FromZeroSolution, "From zero solution")
        .value("FromL2Solution", Strategy::FromL2Solution, "From l2 solution")
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
            pybind11::arg("matrixIsCovariance"))
        .def_static("fitAll", &FullPathSolverf::fitAll, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("beta"),
            pybind11::arg("withintercept") = false,
            pybind11::arg("strategy") = Strategy::FromZeroSolution);

    pybind11::class_<Solutionf>(m, "Solutionf")
        .def(pybind11::init<Index>())
        .def(pybind11::init<float, Vectorf, Vectorf>())
        .def_readwrite("delta", &Solutionf::delta)
        .def_readwrite("x", &Solutionf::x)
        .def_readwrite("grad", &Solutionf::grad)
        .def_readwrite("intercept", &Solutionf::intercept)
        .def_readwrite("dualityGap", &Solutionf::dualityGap);

    pybind11::class_<L0L2RegressorParamf>(m, "L0L2RegressorParamf")
        .def(pybind11::init<float, float, bool, Strategy, float, unsigned int, float, unsigned int>(),
            pybind11::arg("delta") = 0.f,
            pybind11::arg("beta") = 1.f,
            pybind11::arg("hasIntercept") = false,
            pybind11::arg("strategy") = Strategy::FromZeroSolution,
            pybind11::arg("tolerance") = 1e-4f,
            pybind11::arg("maximumNumberOfIterations") = 10000U, 
            pybind11::arg("innerEpsilon") = 1e-6f,
            pybind11::arg("innerMaximumNumberOfIterations") = 100000U)
        .def_readonly("delta", &L0L2RegressorParamf::delta)
        .def_readonly("beta", &L0L2RegressorParamf::beta)
        .def_readonly("hasIntercept", &L0L2RegressorParamf::hasIntercept)
        .def_readonly("strategy", &L0L2RegressorParamf::strategy)
        .def_readonly("tolerance", &L0L2RegressorParamf::tolerance)
        .def_readonly("maximumNumberOfIterations", &L0L2RegressorParamf::maximumNumberOfIterations)
        .def_readonly("innerEpsilon", &L0L2RegressorParamf::innerEpsilon)
        .def_readonly("innerMaximumNumberOfIterations", &L0L2RegressorParamf::innerMaximumNumberOfIterations);

    pybind11::class_<L0L2Regressorf>(m, "L0L2Regressorf")
        .def(pybind11::init<L0L2RegressorParamf>(), 
            pybind11::arg("param"))
        .def("fit", &L0L2Regressorf::fit,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("coordinateStates") = CoordinateStates{});

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
            pybind11::arg("matrixIsCovariance"));

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
            pybind11::arg("matrixIsCovariance"));

    using SimpleBranchAndBoundf = l0l2::linearmodel::leastsquares::SimpleBranchAndBound<float>;
    using SimpleBranchAndBoundParamf = SimpleBranchAndBoundf::Param;

    pybind11::class_<SimpleBranchAndBoundParamf>(m, "SimpleBranchAndBoundParamf")
        .def(pybind11::init<L0L2RegressorParamf, float, float, unsigned int>(),
            pybind11::arg("regressorParam"),
            pybind11::arg("globalEpsilon") = 1e-8f,
            pybind11::arg("localEpsilon") = 1e-8f,
            pybind11::arg("maximumNumberOfIterations") = 1000000U)
        .def_readonly("regressorParam", &SimpleBranchAndBoundParamf::regressorParam)
        .def_readonly("globalEpsilon", &SimpleBranchAndBoundParamf::globalEpsilon)
        .def_readonly("localEpsilon", &SimpleBranchAndBoundParamf::localEpsilon)
        .def_readonly("maximumNumberOfIterations", &SimpleBranchAndBoundParamf::maximumNumberOfIterations);    
    
    pybind11::class_<SimpleBranchAndBoundf>(m, "SimpleBranchAndBoundf")
        .def(pybind11::init<SimpleBranchAndBoundParamf>(),
            pybind11::arg("param"))
        .def("fit", &SimpleBranchAndBoundf::fit,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"))
        .def("objectiveValue", &SimpleBranchAndBoundf::objectiveValue,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("solution"));

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
            pybind11::arg("matrixIsCovariance"))
        .def_static("fitAll", &FullPathSolverd::fitAll, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("beta"),
            pybind11::arg("withintercept") = false,
            pybind11::arg("strategy") = Strategy::FromZeroSolution);

    pybind11::class_<Solutiond>(m, "Solutiond")
        .def(pybind11::init<Index>())
        .def(pybind11::init<double, Vectord, Vectord>())
        .def_readwrite("delta", &Solutiond::delta)
        .def_readwrite("x", &Solutiond::x)
        .def_readwrite("grad", &Solutiond::grad)
        .def_readwrite("intercept", &Solutiond::intercept)
        .def_readwrite("dualityGap", &Solutiond::dualityGap);

    pybind11::class_<L0L2RegressorParamd>(m, "L0L2RegressorParamd")
        .def(pybind11::init<double, double, bool, Strategy, double, unsigned int, double, unsigned int>(),
            pybind11::arg("delta") = 0.0,
            pybind11::arg("beta") = 1.0,
            pybind11::arg("hasIntercept") = false,
            pybind11::arg("strategy") = Strategy::FromZeroSolution,
            pybind11::arg("tolerance") = 1e-4,
            pybind11::arg("maximumNumberOfIterations") = 10000U,
            pybind11::arg("innerEpsilon") = 1e-6,
            pybind11::arg("innerMaximumNumberOfIterations") = 100000U)
        .def_readonly("delta", &L0L2RegressorParamd::delta)
        .def_readonly("beta", &L0L2RegressorParamd::beta)
        .def_readonly("hasIntercept", &L0L2RegressorParamd::hasIntercept)
        .def_readonly("strategy", &L0L2RegressorParamd::strategy)
        .def_readonly("tolerance", &L0L2RegressorParamd::tolerance)
        .def_readonly("maximumNumberOfIterations", &L0L2RegressorParamd::maximumNumberOfIterations)
        .def_readonly("innerEpsilon", &L0L2RegressorParamd::innerEpsilon)
        .def_readonly("innerMaximumNumberOfIterations", &L0L2RegressorParamd::innerMaximumNumberOfIterations);

    pybind11::class_<L0L2Regressord>(m, "L0L2Regressord")
        .def(pybind11::init<L0L2RegressorParamd>(), 
            pybind11::arg("param"))
        .def("fit", &L0L2Regressord::fit, 
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("coordinateStates") = CoordinateStates{});

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
            pybind11::arg("matrixIsCovariance"));

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
            pybind11::arg("matrixIsCovariance"));

    using SimpleBranchAndBoundd = l0l2::linearmodel::leastsquares::SimpleBranchAndBound<double>;
    using SimpleBranchAndBoundParamd = SimpleBranchAndBoundd::Param;

    pybind11::class_<SimpleBranchAndBoundParamd>(m, "SimpleBranchAndBoundParamd")
        .def(pybind11::init<L0L2RegressorParamd, double, double, unsigned int>(),
            pybind11::arg("regressorParam"),
            pybind11::arg("globalEpsilon") = 1e-8,
            pybind11::arg("localEpsilon") = 1e-8,
            pybind11::arg("maximumNumberOfIterations") = 1000000U)
        .def_readonly("regressorParam", &SimpleBranchAndBoundParamd::regressorParam)
        .def_readonly("globalEpsilon", &SimpleBranchAndBoundParamd::globalEpsilon)
        .def_readonly("localEpsilon", &SimpleBranchAndBoundParamd::localEpsilon)
        .def_readonly("maximumNumberOfIterations", &SimpleBranchAndBoundParamd::maximumNumberOfIterations);

    pybind11::class_<SimpleBranchAndBoundd>(m, "SimpleBranchAndBoundd")
        .def(pybind11::init<SimpleBranchAndBoundParamd>(),
            pybind11::arg("param"))
        .def("fit", &SimpleBranchAndBoundd::fit,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"))
        .def("objectiveValue", &SimpleBranchAndBoundd::objectiveValue,
            pybind11::arg("matData"),
            pybind11::arg("vectData"),
            pybind11::arg("matrixIsCovariance"),
            pybind11::arg("solution"));

#ifdef L0L2_VERSION
    mainmodule.attr("__version__") = L0L2_MACRO_STRINGIFY(L0L2_VERSION);
#else
    mainmodule.attr("__version__") = "dev";
#endif
}
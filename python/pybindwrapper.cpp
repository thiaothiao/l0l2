#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/attr.h>
#include <pybind11/stl.h>
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>

#include <l0l2/core.hpp>
#include <l0l2/version.hpp>

PYBIND11_MODULE(l0l2, mainmodule)
{
    mainmodule.doc() = "l0l2 sparse modeling module.";

    auto m = mainmodule.def_submodule("linearmodel", "Linear model module.");

    using CoordinateState = l0l2::linearmodel::leastsquares::CoordinateState;
    pybind11::enum_<CoordinateState>(m, "CoordinateState", pybind11::arithmetic(), "State of a coordinate")
        .value("Unknown", CoordinateState::Unknown, "Concerned by l0")
        .value("Nonzero", CoordinateState::Nonzero, "Freed from l0")
        .value("Zero", CoordinateState::Zero, "Forced to zero")
        .export_values();

    using Strategy = l0l2::linearmodel::leastsquares::Strategy;
    pybind11::enum_<Strategy>(m, "Strategy", pybind11::arithmetic(), "Choosen strategy")
        .value("SequentialFromZeroSolution", Strategy::SequentialFromZeroSolution, "From zero solution")
        .value("SequentialFromL2Solution", Strategy::SequentialFromL2Solution, "From l2 solution")
        .value("Parallel", Strategy::Parallel, "From both sides")
        .export_values();

    using FullPathSolverf = l0l2::linearmodel::leastsquares::FullPathSolver<float>;
    using FullPathSolverParamf = FullPathSolverf::Param;
    pybind11::class_<FullPathSolverParamf>(m, "FullPathSolverParamf")
        .def(pybind11::init<float, float, Strategy>(),
            pybind11::arg("delta") = 0.f,
            pybind11::arg("beta") = 1.f,
            pybind11::arg("strategy") = Strategy::SequentialFromZeroSolution)
        .def_readonly("delta", &FullPathSolverParamf::delta)
        .def_readonly("beta", &FullPathSolverParamf::beta)
        .def_readonly("strategy", &FullPathSolverParamf::strategy);

    pybind11::class_<FullPathSolverf>(m, "FullPathSolverf")
        .def(pybind11::init<FullPathSolverParamf, bool>(),
             pybind11::arg("param"), pybind11::arg("withintercept") = false)
        .def("fit", &FullPathSolverf::fit, pybind11::arg("matData"),
             pybind11::arg("vectData"))
        .def_static("fitAll", &FullPathSolverf::fitAll,
                    pybind11::arg("matData"), pybind11::arg("vectData"),
                    pybind11::arg("beta"),
                    pybind11::arg("withintercept") = false,
                    pybind11::arg("strategy") = Strategy::SequentialFromZeroSolution);

    using Index = l0l2::Index;
    using Vectorf = l0l2::Vector<float>;
    using Matrixf = l0l2::Matrix<float>;

    using Solutionf = l0l2::linearmodel::leastsquares::Solution<float>;
    pybind11::class_<Solutionf>(m, "Solutionf")
        .def(pybind11::init<Index>())
        .def(pybind11::init<float, Vectorf, Vectorf>())
        .def_readwrite("delta", &Solutionf::delta)
        .def_readwrite("x", &Solutionf::x)
        .def_readwrite("grad", &Solutionf::grad)
        .def_readwrite("intercept", &Solutionf::intercept)
        .def_readwrite("dualityGap", &Solutionf::dualityGap);

    using L0L2Regressorf = l0l2::linearmodel::leastsquares::L0L2Regressor<float>;
    using L0L2RegressorParamf = L0L2Regressorf::Param;
    pybind11::class_<L0L2RegressorParamf>(m, "L0L2RegressorParamf")
        .def(pybind11::init<float, float, bool, Strategy, float, unsigned int, float, unsigned int>(),
            pybind11::arg("delta") = 0.f,
            pybind11::arg("beta") = 1.f,
            pybind11::arg("hasIntercept") = false,
            pybind11::arg("strategy") = Strategy::SequentialFromZeroSolution,
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

    using CoordinateStates = l0l2::linearmodel::leastsquares::CoordinateStates;

    pybind11::class_<L0L2Regressorf>(m, "L0L2Regressorf")
        .def(pybind11::init<L0L2RegressorParamf>(), pybind11::arg("param"))
        .def("fit",
             pybind11::overload_cast<const Matrixf &, const Vectorf &,
                                     const CoordinateStates &>(
                 &L0L2Regressorf::fit),
             pybind11::arg("matData"), pybind11::arg("vectData"),
             pybind11::arg("coordinateStates") = CoordinateStates{});

    using SimpleBranchAndBoundf = l0l2::linearmodel::leastsquares::SimpleBranchAndBound<float>;
    using SimpleBranchAndBoundParamf = SimpleBranchAndBoundf::Param;
    pybind11::class_<SimpleBranchAndBoundParamf>(m,
                                                 "SimpleBranchAndBoundParamf")
        .def(pybind11::init<L0L2RegressorParamf, float, float, unsigned int>(),
             pybind11::arg("regressorParam"),
             pybind11::arg("globalEpsilon") = 1e-8f,
             pybind11::arg("localEpsilon") = 1e-8f,
             pybind11::arg("maximumNumberOfIterations") = 1000000U)
        .def_readonly("regressorParam",
                      &SimpleBranchAndBoundParamf::regressorParam)
        .def_readonly("globalEpsilon",
                      &SimpleBranchAndBoundParamf::globalEpsilon)
        .def_readonly("localEpsilon", &SimpleBranchAndBoundParamf::localEpsilon)
        .def_readonly("maximumNumberOfIterations",
                      &SimpleBranchAndBoundParamf::maximumNumberOfIterations);

    pybind11::class_<SimpleBranchAndBoundf>(m, "SimpleBranchAndBoundf")
        .def(pybind11::init<SimpleBranchAndBoundParamf>(),
             pybind11::arg("param"))
        .def("fit", &SimpleBranchAndBoundf::fit, pybind11::arg("matData"),
             pybind11::arg("vectData"))
        .def("objectiveValue", &SimpleBranchAndBoundf::objectiveValue,
             pybind11::arg("matData"), pybind11::arg("vectData"),
             pybind11::arg("solution"));

    using FullPathSolverd = l0l2::linearmodel::leastsquares::FullPathSolver<double>;
    using FullPathSolverParamd = FullPathSolverd::Param;
    pybind11::class_<FullPathSolverParamd>(m, "FullPathSolverParamd")
        .def(pybind11::init<double, double, Strategy>(),
            pybind11::arg("delta") = 0.0,
            pybind11::arg("beta") = 1.0,
            pybind11::arg("strategy") = Strategy::SequentialFromZeroSolution)
        .def_readonly("delta", &FullPathSolverParamd::delta)
        .def_readonly("beta", &FullPathSolverParamd::beta)
        .def_readonly("strategy", &FullPathSolverParamd::strategy);

    pybind11::class_<FullPathSolverd>(m, "FullPathSolverd")
        .def(pybind11::init<FullPathSolverParamd, bool>(),
             pybind11::arg("param"), pybind11::arg("withintercept") = false)
        .def("fit", &FullPathSolverd::fit, pybind11::arg("matData"),
             pybind11::arg("vectData"))
        .def_static("fitAll", &FullPathSolverd::fitAll,
                    pybind11::arg("matData"), pybind11::arg("vectData"),
                    pybind11::arg("beta"),
                    pybind11::arg("withintercept") = false,
                    pybind11::arg("strategy") = Strategy::SequentialFromZeroSolution);

    using Vectord = l0l2::Vector<double>;
    using Matrixd = l0l2::Matrix<double>;

    using Solutiond = l0l2::linearmodel::leastsquares::Solution<double>;
    pybind11::class_<Solutiond>(m, "Solutiond")
        .def(pybind11::init<Index>())
        .def(pybind11::init<double, Vectord, Vectord>())
        .def_readwrite("delta", &Solutiond::delta)
        .def_readwrite("x", &Solutiond::x)
        .def_readwrite("grad", &Solutiond::grad)
        .def_readwrite("intercept", &Solutiond::intercept)
        .def_readwrite("dualityGap", &Solutiond::dualityGap);

    using L0L2Regressord = l0l2::linearmodel::leastsquares::L0L2Regressor<double>;
    using L0L2RegressorParamd = L0L2Regressord::Param;
    pybind11::class_<L0L2RegressorParamd>(m, "L0L2RegressorParamd")
        .def(pybind11::init<double, double, bool, Strategy, double, unsigned int, double, unsigned int>(),
            pybind11::arg("delta") = 0.0,
            pybind11::arg("beta") = 1.0,
            pybind11::arg("hasIntercept") = false,
            pybind11::arg("strategy") = Strategy::SequentialFromZeroSolution,
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
        .def(pybind11::init<L0L2RegressorParamd>(), pybind11::arg("param"))
        .def("fit",
             pybind11::overload_cast<const Matrixd &, const Vectord &,
                                     const CoordinateStates &>(
                 &L0L2Regressord::fit),
             pybind11::arg("matData"), pybind11::arg("vectData"),
             pybind11::arg("coordinateStates") = CoordinateStates{});

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
        .def("fit", &SimpleBranchAndBoundd::fit, pybind11::arg("matData"),
             pybind11::arg("vectData"))
        .def("objectiveValue", &SimpleBranchAndBoundd::objectiveValue,
             pybind11::arg("matData"), pybind11::arg("vectData"),
             pybind11::arg("solution"));

    mainmodule.attr("__version__") = l0l2::metadata::libVersion;
}

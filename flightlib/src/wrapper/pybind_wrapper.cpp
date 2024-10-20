
// pybind11
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// flightlib
#include "flightlib/envs/env_base.hpp"
#include "flightlib/envs/quadrotor_env.hpp"
#include "flightlib/envs/vec_env.hpp"

namespace py = pybind11;
using namespace flightlib;

// vec_env -> quadrotor_env
PYBIND11_MODULE(flightgym, m) {
	py::class_<VecEnv<QuadrotorEnv>>(m, "QuadrotorEnv_v1")
	    .def(py::init<>())
	    .def(py::init<const std::string&>())
	    .def(py::init<const std::string&, const bool>())
	    // unity
	    .def("close", &VecEnv<QuadrotorEnv>::close)
	    .def("connectUnity", &VecEnv<QuadrotorEnv>::connectUnity)
	    .def("disconnectUnity", &VecEnv<QuadrotorEnv>::disconnectUnity)
	    .def("render", &VecEnv<QuadrotorEnv>::render)
	    .def("spawnTrees", &VecEnv<QuadrotorEnv>::spawnTrees)
	    .def("savePointcloud", &VecEnv<QuadrotorEnv>::savePointcloud)
	    .def("spawnTreesAndSavePointcloud", &VecEnv<QuadrotorEnv>::spawnTreesAndSavePointcloud)
	    // set
	    .def("step", &VecEnv<QuadrotorEnv>::step)
	    .def("reset", &VecEnv<QuadrotorEnv>::reset)
	    .def("setState", &VecEnv<QuadrotorEnv>::setState)
	    .def("setGoal", &VecEnv<QuadrotorEnv>::setGoal)
	    .def("setSeed", &VecEnv<QuadrotorEnv>::setSeed)
	    .def("setMapID", &VecEnv<QuadrotorEnv>::setMapID)
	    // get
	    .def("getNumOfEnvs", &VecEnv<QuadrotorEnv>::getNumOfEnvs)
	    .def("getWorldBox", &VecEnv<QuadrotorEnv>::getWorldBox)
	    .def("getObsDim", &VecEnv<QuadrotorEnv>::getObsDim)
	    .def("getActDim", &VecEnv<QuadrotorEnv>::getActDim)
	    .def("getRewDim", &VecEnv<QuadrotorEnv>::getRewDim)
	    .def("getRGBImage", &VecEnv<QuadrotorEnv>::getRGBImage)
	    .def("getStereoImage", &VecEnv<QuadrotorEnv>::getStereoImage)
	    .def("getDepthImage", &VecEnv<QuadrotorEnv>::getDepthImage)
	    .def("getImgHeight", &VecEnv<QuadrotorEnv>::getImgHeight)
	    .def("getImgWidth", &VecEnv<QuadrotorEnv>::getImgWidth)
	    .def("getRewardNames", &VecEnv<QuadrotorEnv>::getRewardNames)
	    .def("getCostAndGradient", &VecEnv<QuadrotorEnv>::getCostAndGradient)
	    .def("__repr__", [](const VecEnv<QuadrotorEnv>& a) { return "Flightmare Environment"; });
}
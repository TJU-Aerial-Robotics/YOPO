cmake_minimum_required(VERSION 3.0)

project(flightlib VERSION 0.1.0)

message(STATUS "======================  !Flightmare!  ======================")

# ###############################################################################
# Options
# ###############################################################################
option(ENABLE_FAST "Build with optimizations for speed" ON)
option(ENABLE_BLAS "Build using BLAS and LAPACK libraries" OFF)
option(ENABLE_PARALLEL "Build using openmp parallelization" ON)
option(EIGEN_FROM_SYSTTEM "Use the system-provided Eigen" ON)
set(
	EIGEN_ALTERNATIVE "" CACHE STRING
	"Path to alternative Eigen, autodownload if blank"
)

# ###############################################################################
# Finding Dependencies
# ###############################################################################

# Eigen3
message(STATUS "======> Setup Dependencies")

if(EIGEN_FROM_SYSTTEM)
	find_package(Eigen3 3.3.4 QUIET)

	if(EIGEN3_FOUND)
		message(STATUS "Using system provided Eigen.")
		message(${EIGEN3_INCLUDE_DIR})
	else()
		message(STATUS "No sufficient Eigen version (3.3.4) found.")
		message(STATUS "Restoring to download Eigen sources.")
		include(cmake/eigen.cmake)
	endif()
elseif(EIGEN_ALTERNATIVE STREQUAL "")
	include(cmake/eigen.cmake)
else()
	set(EIGEN_INCLUDE_DIR ${EIGEN_ALTERNATIVE})
endif()

message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# PCL (Many VTK errors are reported, but do not affect compilation)
find_package(PCL 1.10 REQUIRED)
message(STATUS "Using system provided PCL.")
message(${PCL_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Including dependencies
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Including dependencies
include(cmake/pybind11.cmake)
include(cmake/yaml.cmake)

# Including OpenMP & CUDA
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)

if(ENABLE_BLAS)
	set(BLA_VENDOR "Generic")
	find_package(BLAS REQUIRED)

	if(BLAS_FOUND)
		message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
	else()
		message(ERROR "Could not enable BLAS because BLAS was not found")
	endif()

	find_package(LAPACK REQUIRED)

	if(LAPACK_FOUND)
		message(STATUS "Found Lapack: ${LAPACK_LIBRARIES}")
	else()
		message(ERROR "Could not enable LAPACK because LAPACK was not found")
	endif()
endif()

# Check for ccache
if(NOT DEFINED CATKIN_DEVEL_PREFIX)
	find_program(CCACHE_PROGRAM ccache)

	if(CCACHE_PROGRAM)
		set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
	else()
		message(INFO "Build time could be improved with ccache!")
		message(INFO "sudo apt install ccache")
	endif()
endif()

# ###############################################################################
# Setup Compilation Flag
# ###############################################################################
message(STATUS "======> Setup Compilation ")

add_definitions(-DEIGEN_STACK_ALLOCATION_LIMIT=1048576)
include_directories(${EIGEN_INCLUDE_DIR} "tests")
include_directories(${OpenCV_INCLUDE_DIRS})

# Set default build type
if(NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
	endif()
endif()

cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS)
message("CUDA Architecture: ${CUDA_ARCH_FLAGS}")

if(${CMAKE_BUILD_TYPE} STREQUAL Release)
	SET(CUDA_NVCC_FLAGS "${CUDA_ARCH_FLAGS};-O3;-use_fast_math" CACHE STRING "nvcc flags" FORCE)
	message("build CUDA with -O3")
else()
	message("build CUDA with Debug")
	SET(CUDA_NVCC_FLAGS "-g ;-G ;${CUDA_ARCH_FLAGS}" CACHE STRING "nvcc flags" FORCE)
	SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
endif()

# Add c++ flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -fPIC -Wall -DNDEBUG  -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -fPIC -Wall -g -fopenmp")
set(CMAKE_CXX_STANDARD 17)

# Architectural flags
if("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "armv7l")
	message(STATUS "Using ARMv7 optimized flags!")
	set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv7-a -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations")
elseif("${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
	message(STATUS "Using ARM aarch64 optimized flags!")
	set(CMAKE_CXX_ARCH_FLAGS " -Wno-psabi -march=armv8-a+crypto -mcpu=cortex-a57+crypto")
else()
	set(CMAKE_CXX_ARCH_FLAGS " -march=native")
endif()

# Optimized flags
if(ENABLE_FAST)
	message(STATUS "Enabling fast optimization flags!")
	set(CMAKE_CXX_FAST_FLAGS " -Ofast")
else()
	set(CMAKE_CXX_FAST_FLAGS " -O0")
endif()

# BLAS Flags
if(BLAS_FOUND AND LAPACK_FOUND)
	message(STATUS "Enabling BLAS and LAPACK")
	set(CMAKE_CXX_BLAS_FLAGS " -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACK -DEIGEN_USE_LAPACKE")
else()
	set(CMAKE_CXX_BLAS_FLAGS "")
endif()

# Summarize Flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FAST_FLAGS} ${CMAKE_CXX_ARCH_FLAGS} ${CMAKE_CXX_PAR_FLAGS}")
string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "The activated CXX RELEASE configuration is:\n ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "The activated CXX DEBUG configuration is:\n ${CMAKE_CXX_FLAGS_DEBUG}")

# ###############################################################################
# Specify Build Resources
# ###############################################################################
message(STATUS "======>  Setup Build  ")

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/arc_utilities)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/sdf_tools)

# Create file lists for flightlib source
file(GLOB_RECURSE FLIGHTLIB_SOURCES
	src/bridges/*.cpp
	src/objects/*.cpp
	src/sensors/*.cpp
	src/sensors/*.cu
	src/envs/*.cpp
	src/common/*.cpp
	src/grad_traj_optimization/*.cpp
)

# Create file lists for flightlib_gym source
file(GLOB_RECURSE FLIGHTLIB_GYM_SOURCES
	src/wrapper/*.cpp
)

# ###############################################################################
# Optional Catkin Build
# ###############################################################################
if(DEFINED CATKIN_DEVEL_PREFIX)
	message(STATUS "======>  Building with -- catkin -- ")
	include(cmake/catkin.cmake)
	return()
endif()

find_package(catkin REQUIRED COMPONENTS
	roscpp
	sdf_tools
	pcl_ros
	pcl_conversions
	sensor_msgs
)

# ###############################################################################
# Setup Build
# ###############################################################################

# Library and Executables
include_directories(include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/sdf_tools/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/arc_utilities/include)

# 1. The Training Lib
if(NOT FLIGHTLIB_SOURCES)
	set(LIBRARY_NAME)
else()
	# flightlib -CUDA
	cuda_add_library(${PROJECT_NAME} ${FLIGHTLIB_SOURCES})
	target_link_libraries(${PROJECT_NAME} # PRIVATE
		${OpenCV_LIBRARIES}
		${PCL_LIBRARIES}
		${CUDA_curand_LIBRARY}
		${CUDA_cublas_LIBRARY}
		${CUDA_LIBRARIES}
		${catkin_LIBRARIES}
		yaml-cpp
		zmq
		zmqpp
		stdc++fs)
	set(LIBRARY_NAME ${PROJECT_NAME})
	set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
endif()

if(FLIGHTLIB_GYM_SOURCES)
	# flightlib_gym (python3 binding with Pybind11)
	pybind11_add_module(flightgym MODULE
		${FLIGHTLIB_GYM_SOURCES})

	if(EIGEN3_FOUND)
		target_include_directories(flightgym PRIVATE

			# ${PROJECT_SOURCE_DIR}/externals/pybind11-src/include
			${PYBIND11_INCLUDE_DIR}
			${PROJECT_SOURCE_DIR}/include
			${EIGEN3_INCLUDE_DIR})
	else()
		target_include_directories(flightgym PRIVATE

			# ${PROJECT_SOURCE_DIR}/externals/pybind11-src/include
			${PYBIND11_INCLUDE_DIR}
			${PROJECT_SOURCE_DIR}/externals/eigen/eigen3 # pybind11 use #include <Eigen/Core>, however, flightmare use #include <eigen3/Eigen/Core>
			${PROJECT_SOURCE_DIR}/include)
	endif()

	target_link_libraries(flightgym PRIVATE ${LIBRARY_NAME})
endif()

if(ENABLE_BLAS AND BLAS_FOUND AND LAPACK_FOUND)
	message(STATUS "Linking standard BLAS ${BLAS_LIBRARIES}")
	target_link_libraries(${LIBRARY_NAME}
		${BLAS_LIBRARIES}
		${LAPACK_LIBRARIES}
		${LAPACKE_LIBRARIES}
	)
endif()

# Build Other ROS PKG
find_package(catkin REQUIRED COMPONENTS
	tf
	nav_msgs
	cv_bridge
)

include_directories(
	${catkin_INCLUDE_DIRS}
	${PROJECT_SOURCE_DIR}/include
)

# 2. ROS Simulation Node
add_executable(flightros_node
	src/ros_nodes/flight_pilot.cpp
	src/ros_nodes/flight_pilot_node.cpp
)

target_link_libraries(flightros_node
	${LIBRARY_NAME}
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	zmqpp
	stdc++fs
)

# 3. The Planner
add_executable(yopo_planner_node
	src/ros_nodes/yopo_planner_node.cpp
)
target_link_libraries(yopo_planner_node
	${LIBRARY_NAME}
	${catkin_LIBRARIES}
)

# 4. Tool: Trajectory Evaluation
add_executable(traj_eval_node
	src/ros_nodes/traj_eval_node.cpp
)
target_link_libraries(traj_eval_node
	${LIBRARY_NAME}
	${catkin_LIBRARIES}
)

# 5. Tool: Map Visualization
add_executable(map_visual_node
	src/ros_nodes/map_visual_node.cpp
)
target_link_libraries(map_visual_node
	${LIBRARY_NAME}
	${catkin_LIBRARIES}
)

message(STATUS "================  !Done. You can build the project now!  ================")

cmake_minimum_required(VERSION 3.0.2)
project(sensor_simulator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        nav_msgs
        std_msgs
        sensor_msgs
        geometry_msgs
        tf2_ros
        cv_bridge
)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
find_package(yaml-cpp REQUIRED)
message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

# CUDA
find_package(CUDA REQUIRED)
# 自动检测本机 GPU 架构, 如果自动检测失败请手动指定
# set(CUDA_ARCH 120)
if(NOT DEFINED CUDA_ARCH)
    execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE GPU_CC ERROR_QUIET)
    string(REGEX MATCH "[0-9]+\\.[0-9]+" GPU_CC "${GPU_CC}")
    string(REPLACE "." "" CUDA_ARCH "${GPU_CC}")
endif()
if(NOT CUDA_ARCH)
    message(FATAL_ERROR "GPU architecture detection failed. "
            "Set it manually in Simulator/src/CMakeLists.txt in line 27, e.g. set(CUDA_ARCH 86)")
endif()
set(ARCH_FLAGS "-gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
message(WARNING "CUDA Architecture: ${ARCH_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${ARCH_FLAGS}")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

## Declare a catkin package
catkin_package(
        INCLUDE_DIRS include
        LIBRARIES sensor_simulator
        CATKIN_DEPENDS roscpp
        DEPENDS OpenCV)

## Specify additional locations of header files
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}  # Eigen库路径
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(sensor_simulator src/test_simulator.cpp src/sensor_simulator.cpp src/maps.cpp src/perlinnoise.cpp)
target_link_libraries(sensor_simulator ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} OpenMP::OpenMP_CXX yaml-cpp)
target_compile_definitions(sensor_simulator PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")


# raycast + 内置 simsense 双目匹配引擎（去 pybind 的原生 C++ 版，见 src/simsense/）
cuda_add_library(raycast_cuda
        src/sensor_simulator.cu
        src/simsense/core.cu
        src/simsense/csct.cu
        src/simsense/cost.cu
        src/simsense/aggr.cu
        src/simsense/wta.cu
        src/simsense/lrcheck.cu
        src/simsense/filter.cu
        src/simsense/camera.cu)
target_link_libraries(raycast_cuda ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(sensor_simulator_cuda src/test_simulator_cuda.cpp src/maps.cpp src/perlinnoise.cpp)
target_link_libraries(sensor_simulator_cuda ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} raycast_cuda yaml-cpp)
target_compile_definitions(sensor_simulator_cuda PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")

add_executable(dataset_generator src/dataset_generator.cpp src/maps.cpp src/perlinnoise.cpp)
target_link_libraries(dataset_generator ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES} raycast_cuda yaml-cpp)
target_compile_definitions(dataset_generator PRIVATE CONFIG_FILE_PATH="${CMAKE_SOURCE_DIR}/config/config.yaml")
# Download and unpack eigen at configure time
message(STATUS "Getting Eigen...")
message(STATUS "** NOTE ** If there's an issue connecting to GitHub, add a proxy to the download link in file flightlib/cmake/eigen_download.cmake.")

configure_file(
  cmake/eigen_download.cmake
  ${PROJECT_SOURCE_DIR}/externals/eigen/CMakeLists.txt)

execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/externals/eigen
  OUTPUT_QUIET)
if(result)
  message(FATAL_ERROR "Download of Eigen failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/externals/eigen
  OUTPUT_QUIET)
if(result)
  message(FATAL_ERROR "Build step for eigen failed: ${result}")
endif()

message(STATUS "Eigen downloaded!")

set(EIGEN_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/externals/eigen)
cmake_minimum_required(VERSION 3.16)
project(nonlinear_fea_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(WOUND_ENABLE_OPENMP "Build with OpenMP-parallel assembly" ON)
option(WOUND_BUILD_TESTS   "Build the CTest suite"               ON)
option(WOUND_USE_MKL       "Use Intel MKL as the Eigen backend"  OFF)

#--------------------------------------------------#
# DEPENDENCIES
#--------------------------------------------------#

# Eigen: prefer an installed config package, fall back to the legacy
# sibling-directory layout used on the Purdue clusters, then to FetchContent.
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Eigen: found installed Eigen3 ${Eigen3_VERSION}")
else()
    find_path(EIGEN3_INCLUDE_DIR
            NAMES signature_of_eigen3_matrix_library
            HINTS ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../eigen
            PATH_SUFFIXES eigen3 include/eigen3)
    if(EIGEN3_INCLUDE_DIR)
        message(STATUS "Eigen: using headers at ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
    else()
        message(STATUS "Eigen: not found locally, fetching 3.4.0")
        include(FetchContent)
        FetchContent_Declare(eigen
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG 3.4.0
                GIT_SHALLOW TRUE)
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(eigen)
    endif()
endif()

# Boost: header-only, only boost/algorithm/string is used by the mesh reader.
# Look for the headers directly so this works with both a system install and the
# sibling-directory layout, without depending on the deprecated FindBoost module.
find_package(Boost QUIET CONFIG)
if(NOT Boost_FOUND)
    find_path(Boost_INCLUDE_DIRS
            NAMES boost/algorithm/string.hpp
            HINTS ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../boost_1_71_0)
endif()
if(NOT Boost_INCLUDE_DIRS)
    message(FATAL_ERROR "Boost headers not found. Install libboost-dev, or place the "
            "Boost tree next to this directory.")
endif()
message(STATUS "Boost: using headers at ${Boost_INCLUDE_DIRS}")

if(WOUND_ENABLE_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_CXX_FOUND)
        message(WARNING "OpenMP not available; building the serial path.")
    endif()
endif()

#--------------------------------------------------#
# CORE LIBRARY
#--------------------------------------------------#

add_library(wound_core
        meshing/myMeshGenerator.cpp
        src/solver.cpp
        src/wound.cpp
        src/local_solver.cpp
        src/element_functions.cpp
        src/file_io.cpp)

target_include_directories(wound_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Boost_INCLUDE_DIRS})

target_link_libraries(wound_core PUBLIC Eigen3::Eigen)

if(WOUND_ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(wound_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(WOUND_USE_MKL)
    set(BLA_VENDOR Intel10_64lp)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(wound_core PUBLIC EIGEN_USE_MKL_ALL MKL_LP64)
    target_link_libraries(wound_core PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()

#--------------------------------------------------#
# DRIVER
#--------------------------------------------------#

add_executable(surgerywoundcpp3D src/results_circle_wound.cpp)
target_link_libraries(surgerywoundcpp3D PRIVATE wound_core)

#--------------------------------------------------#
# TESTS
#--------------------------------------------------#

if(WOUND_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

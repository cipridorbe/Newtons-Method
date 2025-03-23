cmake_minimum_required(VERSION 3.22.1)
project(newton)
set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")
file(GLOB_RECURSE newton_src src/*.cpp)
add_executable(newton main.cpp ${newton_src}) 
include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

if(NOT USE_SYSTEM_EIGEN)
    set(EIGEN_BUILD_CMAKE_PACKAGE TRUE)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        # master on 2025-01-02
        GIT_TAG 7bb23b1e360e22a395eef75061565fc079ae6a85
    )
    fetchcontent_makeavailable(Eigen3)
else()
    find_package(Eigen3 CONFIG REQUIRED)
endif()

target_link_libraries(newton PUBLIC Eigen3::Eigen)

target_include_directories(newton PRIVATE inc/ src/)
target_compile_options(newton PRIVATE -march=native -fconcepts)

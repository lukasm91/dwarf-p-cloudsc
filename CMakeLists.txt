cmake_minimum_required(VERSION 3.24)
project(cloudscpy LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Interpeter Development )

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cloudsc_cuda src/cloudsc_cuda.cu)

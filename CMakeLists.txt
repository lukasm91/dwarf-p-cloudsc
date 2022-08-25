cmake_minimum_required(VERSION 3.24)
project(cloudscpy LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Interpeter Development )
option(HAVE_SINGLE_PRECISION OFF "Use single precision")
if(HAVE_SINGLE_PRECISION)
  set(PRECISION float)
else()
  set(PRECISION double)
endif()

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.10.0
)
FetchContent_MakeAvailable(pybind11)

pybind11_add_module(cloudsc_cuda src/cloudsc_cuda.cu)
target_compile_definitions(cloudsc_cuda PUBLIC PRECISION=${PRECISION})

install(TARGETS cloudsc_cuda DESTINATION lib)
install(FILES run.py DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
install(FILES python_impl.py DESTINATION bin)

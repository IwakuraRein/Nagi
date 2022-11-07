cmake_minimum_required(VERSION 3.9)

project(Nagi VERSION 0.1.1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Enable C++17 for host code
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
find_package(CUDA 10 REQUIRED)

# list(APPEND CUDA_NVCC_FLAGS "-rdc=true")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
endif(WIN32)
########################################

find_package(OpenGL REQUIRED)

include_directories("src/gui")
include_directories("src/core")
include_directories("src/3rdparty/GLFW/include")
include_directories("src/3rdparty/glad/include")
include_directories("src/3rdparty/glm/include")
include_directories("src/3rdparty/imgui/include")
include_directories("src/3rdparty/json")
include_directories("src/3rdparty/oidn/include")
include_directories("src/3rdparty/stb")
include_directories("src/3rdparty/tinyobjloader")

link_directories("src/3rdparty/oidn/lib")
link_directories("src/3rdparty/GLFW/lib")

add_subdirectory(src/gui)

file(GLOB headers src/core/*.hpp src/core/*.h src/core/*.cuh)
file(GLOB sources src/core/*.cpp src/*.c src/core/*.cu)

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

set(libraries
    OpenGL32
    glfw3
    OpenImageDenoise
    gui)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers})
target_link_libraries(${CMAKE_PROJECT_NAME} ${libraries})
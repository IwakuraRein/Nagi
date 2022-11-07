file(GLOB headers *.hpp *.h *.cuh)
file(GLOB sources *.cpp *.c *.cu ${PROJECT_SOURCE_DIR}/src/3rdparty/imgui/*.cpp ${PROJECT_SOURCE_DIR}/src/3rdparty/glad/*.c)

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)

cuda_add_library(gui ${sources} ${headers} OPTIONS)
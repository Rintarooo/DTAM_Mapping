cmake_minimum_required(VERSION 3.0.2)
project(DTAM_project)

## Compile as C++11
# add_compile_options(-std=c++11)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g3 -O0")# -ggdb3
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")# -g3

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")# option, Device side debug: -G
# set(CUDA_NVCC_FLAGS_DEBUG "-G -O0")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")# -G

# find_package(OpenCV REQUIRED)
find_package(OpenCV 3.4.11 REQUIRED)
find_package(PCL 1.11.0 REQUIRED)
set(CMAKE_CUDA_COMPILER  /usr/local/cuda/bin/nvcc)
find_package(CUDA 10.1 REQUIRED)
find_package(Eigen3 REQUIRED)

### directories where header files exists
# cuda_include_directories(
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  FileLoad/
  CostVolume/
  Viewer3D/
  DenoiseDepth/Regulariser/
  include/
)

## cuda_add_library() is same as cuda_add_executable() except that a library is created.
# cuda_add_library(cuda_node src/lib_cuda_code.cu src/libe_none_cuda_code.cpp )

## create executable file named "main"

# add_subdirectory(CostVolume/)# add_subdirectory(src/directory)

# cuda_compile(main
#   main.cpp f_load.cpp
#   -G )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

cuda_add_executable(main
  main.cpp
  FileLoad/f_load.cpp
  CostVolume/CostVolume.cpp
  CostVolume/CostVolume_update.cu
  Viewer3D/pcl3d.cpp
  DenoiseDepth/Regulariser/Regulariser.cpp
  DenoiseDepth/Regulariser/Regulariser_computeG.cu
  DenoiseDepth/Regulariser/Regulariser_update_q_d.cu
  DenoiseDepth/Regulariser/Regulariser_minimizeEaux.cu
)
  # OPTIONS -G
  # OPTIONS -G -g


target_link_libraries(main
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${PCL_LIBRARIES}
)

# for debug
message(WARNING "prefix ${CMAKE_PREFIX_PATH}")    

message("-- CUDA --")
message(WARNING "version: " ${CUDA_VERSION})
message(WARNING "library: " ${CUDA_CUDA_LIBRARY})
message(WARNING "runtime library: " ${CUDA_CUDART_LIBRARY})
message(WARNING "include dirs: " ${CUDA_INCLUDE_DIRS})

message("-- OpenCV --")
message(WARNING "version ${OpenCV_VERSION}")
message(WARNING "install path ${OpenCV_INSTALL_PATH}") 
message(WARNING "config path ${OpenCV_CONFIG_PATH}") # look at the output of this message
message(WARNING "libs ${OpenCV_LIBS}")
message(WARNING "include dirs ${OpenCV_INCLUDE_DIRS}")

message("-- PCL --")
message(WARNING "version ${PCL_VERSION}")
message(WARNING "where libs ${PCL_LIBRARY_DIRS}")
message(WARNING "libs ${PCL_LIBRARIES}")
message(WARNING "components ${PCL_COMPONENTS}")
message(WARNING "include dirs ${PCL_INCLUDE_DIRS}")
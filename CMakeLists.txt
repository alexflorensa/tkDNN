cmake_minimum_required(VERSION 3.15)

project(tkDNN)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wno-deprecated-declarations -Wno-unused-variable ")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/tkDNN)

# project specific flags
if (DEBUG)
    add_definitions(-DDEBUG)
endif ()

if (TKDNN_PATH)
    message("SET TKDNN_PATH:" ${TKDNN_PATH})
    add_definitions(-DTKDNN_PATH="${TKDNN_PATH}")
else ()
    add_definitions(-DTKDNN_PATH="${CMAKE_CURRENT_SOURCE_DIR}")
endif ()


#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
SET(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_30 --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --maxrregcount=32)

find_package(CUDNN REQUIRED)
include_directories(${CUDNN_INCLUDE_DIR})

# compile
file(GLOB tkdnn_CUSRC "src/kernels/*.cu" "src/sorting.cu")
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS})
cuda_add_library(kernels SHARED ${tkdnn_CUSRC})
target_link_libraries(kernels ${CUDA_CUBLAS_LIBRARIES})


#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
message("Eigen DIR: " ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOPENCV")
# if(OpenCV_CUDA_VERSION)
#     add_compile_definitions(OPENCV_CUDACONTRIB)
# endif()

# gives problems in cross-compiling, probably malformed cmake config
find_package(yaml-cpp REQUIRED)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
file(GLOB tkdnn_SRC "src/*.cpp")
set(tkdnn_LIBS kernels ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDNN_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${NVINFER_INCLUDES})
add_library(tkDNN SHARED ${tkdnn_SRC})
target_link_libraries(tkDNN ${tkdnn_LIBS})

# DARKNET
file(GLOB darknet_SRC "tests/darknet/*.cpp")
foreach (test_SRC ${darknet_SRC})
    get_filename_component(test_NAME "${test_SRC}" NAME_WE)
    set(test_NAME test_${test_NAME})
    add_executable(${test_NAME} ${test_SRC})
    target_link_libraries(${test_NAME} tkDNN)
    install(TARGETS ${test_NAME} DESTINATION bin)
endforeach ()


# DEMOS
add_executable(test_rtinference tests/test_rtinference/rtinference.cpp)
target_link_libraries(test_rtinference tkDNN)

add_executable(map_demo demo/demo/map.cpp)
target_link_libraries(map_demo tkDNN)

add_executable(demo demo/demo/demo.cpp)
target_link_libraries(demo tkDNN)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

option(BUILD_SHARED_LIBS "Specifies the type of libraries (SHARED or STATIC) to build" ON)
if (BUILD_SHARED_LIBS)
    add_library(PythonApi SHARED api/PythonApi.cpp)
    target_compile_definitions(PythonApi PRIVATE LIB_EXPORTS=1)
    target_compile_definitions(PythonApi PRIVATE -DDEMO_EXPORTS)
    target_link_libraries(PythonApi tkDNN)
endif ()

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
message("install dir:" ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY include/ DESTINATION include/)
install(TARGETS tkDNN kernels DESTINATION lib)
install(TARGETS test_rtinference demo map_demo DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" # source directory
        DESTINATION "share/tkDNN/cmake/" # target directory
        )
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests/" # source directory
        DESTINATION "share/tkDNN/tests" # target directory
        )

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(ts_ssd)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost 1.65.1 REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Torch REQUIRED)

aux_source_directory(. DIR_SRCS)

add_executable(ts_ssd ${DIR_SRCS})
target_link_libraries(ts_ssd "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" "${Boost_LIBRARIES}")
set_property(TARGET ts_ssd PROPERTY CXX_STANDARD 11)

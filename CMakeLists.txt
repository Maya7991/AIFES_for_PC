# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MainCProject C)

# Set the source directory
set(SRC_DIR src)

# Recursively gather all source files in SRC_DIR
file(GLOB_RECURSE SOURCES ${SRC_DIR}/*.c)

# Include all subdirectories for headers
include_directories(${SRC_DIR})

# Add the executable
add_executable(main ${SOURCES})

target_link_libraries(main m)

# Set compiler flags for debug builds
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
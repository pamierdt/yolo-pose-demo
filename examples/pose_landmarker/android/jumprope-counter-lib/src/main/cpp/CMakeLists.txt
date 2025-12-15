cmake_minimum_required(VERSION 3.18.1)

project(jumprope_counter)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the jump rope counter library
add_library(
    jumprope_counter
    SHARED
    JumpRopeCounter.cpp
    jni_bridge.cpp
)

# Find required Android libraries
find_library(log-lib log)

# Link libraries
target_link_libraries(
    jumprope_counter
    ${log-lib}
)

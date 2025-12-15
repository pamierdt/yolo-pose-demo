cmake_minimum_required(VERSION 3.18.1)

project(pose_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JNI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs")

# Build the pose detector library
add_library(
    pose_detector
    SHARED
    rknn_runner.cpp
)

# Import prebuilt RKNN runtime library
add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION "${JNI_DIR}/${ANDROID_ABI}/librknnrt_dt.so"
)

# Import prebuilt RGA library for hardware acceleration
add_library(rga SHARED IMPORTED)
set_target_properties(rga PROPERTIES
    IMPORTED_LOCATION "${JNI_DIR}/${ANDROID_ABI}/librga_dt.so"
)

# Include directories
target_include_directories(
    pose_detector
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/rga"
)

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

# Link libraries
target_link_libraries(
    pose_detector
    rknnrt
    rga
    ${jnigraphics-lib}
    ${log-lib}
    ${android-lib}
)

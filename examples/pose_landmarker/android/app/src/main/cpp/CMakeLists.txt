cmake_minimum_required(VERSION 3.18.1)

project(rknn_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(JNI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../jni")

add_library(
    rknn_jni
    SHARED
    rknn_jni.cpp
)

add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION "${JNI_DIR}/${ANDROID_ABI}/librknnrt.so"
)

target_include_directories(
    rknn_jni
    PRIVATE
    "${JNI_DIR}/include"
)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(
    rknn_jni
    rknnrt
    jnigraphics
    ${log-lib}
    ${android-lib}
)

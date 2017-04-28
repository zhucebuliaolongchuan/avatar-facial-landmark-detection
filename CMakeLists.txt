cmake_minimum_required(VERSION 2.8)
project(face_landmark_detection)

find_package(OpenCV REQUIRED)
include(../libs/dlib/cmake)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB PROJECT_SOURCES *.cpp)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
target_link_libraries(${PROJECT_NAME} dlib::dlib ${OpenCV_LIBS})

SET(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++")


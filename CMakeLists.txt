cmake_minimum_required(VERSION 3.2)

project(gameai-assignment-vision)

find_package(Boost REQUIRED system thread)
find_package(OpenCV REQUIRED)

include_directories(include)

add_executable(gameai-assignment-vision src/main.cpp)
set_property(TARGET gameai-assignment-vision PROPERTY CXX_STANDARD 11)
set_property(TARGET gameai-assignment-vision PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(gameai-assignment-vision ${Boost_LIBRARIES} ${OpenCV_LIBS})

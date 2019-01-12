cmake_minimum_required(VERSION 2.8.3)
project(svm_train)

add_compile_options(-std=c++11)

set(OpenCV_STATIC OFF)
find_package( OpenCV 3.4 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(svm_train src/main.cpp)


target_link_libraries(
  svm_train
  ${OpenCV_LIBS}
)

message("Open CV version is ${OpenCV_VERSION}")

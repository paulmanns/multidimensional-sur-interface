cmake_minimum_required(VERSION 3.0)

project(multidimensional-sur)

find_package (Eigen3 3.3 REQUIRED)

add_subdirectory(src)
add_subdirectory(python)


cmake_minimum_required(VERSION 3.22)
project(kokkos-himeno)

add_executable(kokkos-himeno main.cpp)
find_package(Kokkos REQUIRED)
target_link_libraries(kokkos-himeno Kokkos::kokkos)
target_compile_options(kokkos-himeno PRIVATE -DELARGE)

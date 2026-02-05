cmake_minimum_required(VERSION 3.10)
project(Mesher)

find_package(CGAL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(ITK REQUIRED)
find_package(TBB REQUIRED)

add_executable(mesher mesher.cpp)

# Link
include(${ITK_USE_FILE})
target_link_libraries(mesher CGAL::CGAL ZLIB::ZLIB TBB::tbb ${ITK_LIBRARIES})

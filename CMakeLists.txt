cmake_minimum_required (VERSION 2.8)
project (sonar_processing)

find_package ( Eigen3 REQUIRED )

file ( GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp )
file ( GLOB HEADERS ${PROJECT_SOURCE_DIR}/src/*.hpp )

include_directories (
    ${CMAKE_INSTALL_PREFIX}/include/
    ${CMAKE_INSTALL_PREFIX}/include/sonar_toolkit
    ${EIGEN3_INCLUDE_DIR}
)

link_directories (
    ${CMAKE_INSTALL_PREFIX}/lib/
    ${CMAKE_INSTALL_PREFIX}/lib/sonar_toolkit
)

add_library (
    sonar_processing STATIC
    ${SOURCES}
)

target_link_libraries (
    sonar_processing
)

install(
    FILES ${HEADERS}
    DESTINATION include/sonar_toolkit/${PROJECT_NAME}
)

install(
    TARGETS sonar_processing
    DESTINATION lib/sonar_toolkit
)

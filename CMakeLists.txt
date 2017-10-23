cmake_minimum_required (VERSION 2.8)
project (sonar_processing)

find_package ( Eigen3 REQUIRED )

file ( GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp )
file ( GLOB HEADERS ${PROJECT_SOURCE_DIR}/src/*.hpp )

include_directories (
    ${CMAKE_INSTALL_PREFIX}/include/
    ${EIGEN3_INCLUDE_DIR}
)

link_directories (
    ${CMAKE_INSTALL_PREFIX}/lib/
)

add_library (
    sonar_processing SHARED
    ${SOURCES}
)

install(
    FILES ${HEADERS}
    DESTINATION include/${PROJECT_NAME}
)

install(
    TARGETS sonar_processing
    DESTINATION lib
)

configure_file(
    src/${PROJECT_NAME}.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
    @ONLY
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc
    DESTINATION lib/pkgconfig
)

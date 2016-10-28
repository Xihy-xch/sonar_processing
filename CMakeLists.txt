cmake_minimum_required (VERSION 2.8)
project (sonar_processing)

set (
    ${PROJECT_NAME}_SOURCE_DIR
    ${PROJECT_SOURCE_DIR}/src
    CACHE PATH "Sonar Target Tracking Source Directory"
    FORCE
)

set (
    ${PROJECT_NAME}_INCLUDE_DIR
    ${PROJECT_SOURCE_DIR}/src
    ${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}
    CACHE PATH "Sonar Target Tracking Include Directory"
    FORCE
)

file ( GLOB ${PROJECT_NAME}_SOURCES PATH ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/*.cpp )

set (
    ${PROJECT_NAME}_SOURCES
    ${${PROJECT_NAME}_SOURCES}
    CACHE STRING "Sonar Target Tracking Source Files"
    FORCE
)

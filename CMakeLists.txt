cmake_minimum_required (VERSION 2.6)
project (sonar_target_tracking)

include ( FindPkgConfig  )

find_package ( OpenCV REQUIRED )

file ( GLOB SRCS RELATIVE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src/*.cpp )

set ( ${PROJECT_NAME}_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include CACHE INTERNAL "Sonar Target Tracking Include Directory" FORCE )

include_directories (
    ${sonar_target_tracking_INCLUDE_DIR}
)

add_library (
    sonar_target_tracking STATIC
    ${SRCS}
)

target_link_libraries (
    sonar_target_tracking
    ${OpenCV_LIBS}
)

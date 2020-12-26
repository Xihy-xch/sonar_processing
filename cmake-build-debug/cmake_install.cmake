# Install script for directory: /home/xch/workspace/vscode-workspaces/C++/sonar_processing

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/sonar_processing" TYPE FILE FILES
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/Denoising.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/HOGDetector.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/HogDescriptorViz.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/ImageFiltering.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/ImageUtil.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/LinearSVM.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/MathUtil.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/Preprocessing.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/QualityMetrics.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/ScanningHolder.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/SonarHolder.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/SonarImagePreprocessing.hpp"
    "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/src/Utils.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsonar_processing.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsonar_processing.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsonar_processing.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/cmake-build-debug/libsonar_processing.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsonar_processing.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsonar_processing.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libsonar_processing.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/cmake-build-debug/sonar_processing.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/xch/workspace/vscode-workspaces/C++/sonar_processing/cmake-build-debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

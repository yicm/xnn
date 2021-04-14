CMAKE_MINIMUM_REQUIRED(VERSION 3.1)

#--------------------------------------------
# Project setting
#--------------------------------------------
INCLUDE(cmake/messagecolor.cmake)
INCLUDE(cmake/utils.cmake)
APP_EXTRACT_VERSION()

PROJECT(HelloApp VERSION ${XNN_APP_VERSION} LANGUAGES CXX)

OPTION(BUILD_TESTS "Build with tests" ON)
OPTION(BUILD_EXAMPLES "Build with examples" ON)
OPTION(BUILD_BENCHMARKS "Build with benchmarks" ON)

MESSAGE(INFO "--------------------------------")
MESSAGE(STATUS "Build XNNApp: ${XNN_APP_VERSION}")
MESSAGE(STATUS "Build with examples: ${BUILD_EXAMPLES}")
MESSAGE(STATUS "Build with tests: ${BUILD_TESTS}")
MESSAGE(STATUS "Build with benchmarks: ${BUILD_BENCHMARKS}")
MESSAGE(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
IF (NOT CMAKE_TOOLCHAIN_FILE)
    MESSAGE(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
ENDIF()

#--------------------------------------------
# Set default build to debug
#--------------------------------------------
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug" FORCE)
ENDIF()

MESSAGE(STATUS "Build type: " ${CMAKE_BUILD_TYPE})
#--------------------------------------------
# Set compilation options
#--------------------------------------------
MESSAGE(INFO "--------------------------------")
INCLUDE(cmake/toolchain_options.cmake)
INCLUDE(cmake/load_3rdparty.cmake)

IF(NOT YOUR_TARGET_OS)
    SET(YOUR_TARGET_OS linux)
ENDIF()
IF(NOT YOUR_TARGET_ARCH)
    SET(YOUR_TARGET_ARCH x86-64)
ENDIF()
MESSAGE(STATUS "Your target os : ${YOUR_TARGET_OS}")
MESSAGE(STATUS "Your target arch : ${YOUR_TARGET_ARCH}")

LOAD_3RDPARTY(YOUR_TARGET_OS YOUR_TARGET_ARCH)

#--------------------------------------------
# Build 3rdparty
#--------------------------------------------
MESSAGE(INFO "--------------------------------")
#--------------------------------------------
ADD_SUBDIRECTORY(3rdparty)

# Build application
#--------------------------------------------
MESSAGE(INFO "--------------------------------")
ADD_SUBDIRECTORY(src)

IF(BUILD_EXAMPLES)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(examples)
ENDIF()

IF(BUILD_TESTS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(tests)
ENDIF()

IF(BUILD_BENCHMARKS)
    MESSAGE(INFO "--------------------------------")
    ADD_SUBDIRECTORY(benchmarks)
ENDIF()



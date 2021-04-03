CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(Tests)

#######################
# find opencv start
#######################
FIND_PACKAGE(OpenCV QUIET COMPONENTS opencv_world)
# for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
IF("${OpenCV_LIBS}" STREQUAL "")
    set(OpenCV_FOUND FALSE)
ENDIF()
IF(NOT OpenCV_FOUND)
    FIND_PACKAGE(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
ENDIF()
IF(NOT OpenCV_FOUND)
    FIND_PACKAGE(OpenCV QUIET COMPONENTS core highgui imgproc)
ENDIF()

IF(NOT OpenCV_FOUND)
    MESSAGE(WARNING "OpenCV not found, examples won't be built")
ELSE()
    MESSAGE(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
    MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
    MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
    MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
ENDIF()
#######################
# find opencv end
#######################

INCLUDE_DIRECTORIES(
    ${SPDLOG_INCLUDE_DIR}
    ${JSONCPP_INCLUDE_DIR}
    ${HEADER_ONLY_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

LINK_DIRECTORIES(
    ${SPDLOG_LIBRARY_DIR}
    ${JSONCPP_LIBRARY_DIR}
    ${MNN_LIBRARY_DIR}
)

FILE(GLOB APP_SOURCES *.cpp)
FOREACH(testsourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" testsourcefilewithoutpath ${testsourcefile})
    STRING(REPLACE ".cpp" "" testname ${testsourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${testsourcefile})
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/tests)
    TARGET_LINK_LIBRARIES(${testname}
        ${OpenCV_LIBS}
        ${SPDLOG_LIBS}
        module_logger
        module_config
        module_mnn
        )

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(testsourcefile ${APP_SOURCES})

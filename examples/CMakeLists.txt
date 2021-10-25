CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
PROJECT(Examples)

INCLUDE_DIRECTORIES(
    ${MNN_INCLUDE_DIR}
    ${NCNN_INCLUDE_DIR}
    ${HEADER_ONLY_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

LINK_DIRECTORIES(
    ${MNN_LIBRARY_DIR}
    ${NCNN_LIBRARY_DIR}
    ${TENGINE_LIBRARY_DIR}
)

FILE(GLOB APP_SOURCES *.cpp)
FOREACH(testsourcefile ${APP_SOURCES})
    STRING(REGEX MATCH "[^/]+$" testsourcefilewithoutpath ${testsourcefile})
    STRING(REPLACE ".cpp" "" testname ${testsourcefilewithoutpath})
    ADD_EXECUTABLE( ${testname} ${testsourcefile})
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin/examples)
    TARGET_LINK_LIBRARIES(${testname}
        module_common
        #module_mnn
        module_ncnn
        module_tengine
        )

    INSTALL(TARGETS ${testname}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
ENDFOREACH(testsourcefile ${APP_SOURCES})

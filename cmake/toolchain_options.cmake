IF(CMAKE_VERSION VERSION_LESS "3.1")
    IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}, -std=gnu++11")
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    ENDIF()
ELSE()
    message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
    SET(CMAKE_CXX_STANDARD 11)
ENDIF()

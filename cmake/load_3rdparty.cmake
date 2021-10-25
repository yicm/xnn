# Once done, this will define
#
#  SPDLOG_INCLUDE_DIR - the SPDLOG include directory
#  SPDLOG_LIBRARY_DIR - the SPDLOG library directory
#  SPDLG_LIBS - link these to use SPDLOG
#
#  ......

MACRO(LOAD_HEADER_ONLY)
    SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/)
    MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")
    FIND_FILE(HEADER_ONLY_INCLUDE_DIR header_only ${3RDPARTY_DIR} NO_DEFAULT_PATH)

    IF(HEADER_ONLY_INCLUDE_DIR)
        MESSAGE(STATUS "HEADER_ONLY_INCLUDE_DIR : ${HEADER_ONLY_INCLUDE_DIR}")
    ELSE()
        MESSAGE(FATAL_ERROR "HEADER_ONLY not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_LIBSPDLOG os arch)
    SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${${os}}_${${arch}})
    MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")
    FIND_FILE(SPDLOG_INCLUDE_DIR include ${3RDPARTY_DIR} NO_DEFAULT_PATH)
    FIND_FILE(SPDLOG_LIBRARY_DIR lib ${3RDPARTY_DIR} NO_DEFAULT_PATH)

    SET(SPDLOG_LIBS
        spdlog
        pthread
        #PARENT_SCOPE no parent
    )
    IF(SPDLOG_INCLUDE_DIR)
        SET(SPDLOG_LIBRARY_DIR "${SPDLOG_LIBRARY_DIR}/spdlog")
        MESSAGE(STATUS "SPDLOG_INCLUDE_DIR : ${SPDLOG_INCLUDE_DIR}")
        MESSAGE(STATUS "SPDLOG_LIBRARY_DIR : ${SPDLOG_LIBRARY_DIR}")
        MESSAGE(STATUS "SPDLOG_LIBS : ${SPDLOG_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "SPDLOG_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_LIBMNN os arch)
    SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${${os}}_${${arch}})
    MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")
    FIND_FILE(MNN_INCLUDE_DIR include ${3RDPARTY_DIR} NO_DEFAULT_PATH)
    FIND_FILE(MNN_LIBRARY_DIR lib ${3RDPARTY_DIR} NO_DEFAULT_PATH)

    SET(MNN_LIBS
        MNN
        pthread
        #PARENT_SCOPE no parent
    )
    IF(MNN_INCLUDE_DIR)
        SET(MNN_LIBRARY_DIR "${MNN_LIBRARY_DIR}/MNN")
        MESSAGE(STATUS "MNN_INCLUDE_DIR : ${MNN_INCLUDE_DIR}")
        MESSAGE(STATUS "MNN_LIBRARY_DIR : ${MNN_LIBRARY_DIR}")
        MESSAGE(STATUS "MNN_LIBS : ${MNN_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "MNN_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_LIBNCNN os arch)
    SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${${os}}_${${arch}})
    MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")
    FIND_FILE(NCNN_INCLUDE_DIR include ${3RDPARTY_DIR} NO_DEFAULT_PATH)
    FIND_FILE(NCNN_LIBRARY_DIR lib ${3RDPARTY_DIR} NO_DEFAULT_PATH)

    SET(NCNN_LIBS
        ncnn
        pthread
        #PARENT_SCOPE no parent
    )
    IF(NCNN_INCLUDE_DIR)
        SET(NCNN_LIBRARY_DIR "${NCNN_LIBRARY_DIR}/ncnn")
        MESSAGE(STATUS "NCNN_INCLUDE_DIR : ${NCNN_INCLUDE_DIR}")
        MESSAGE(STATUS "NCNN_LIBRARY_DIR : ${NCNN_LIBRARY_DIR}")
        MESSAGE(STATUS "NCNN_LIBS : ${NCNN_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "NCNN_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_LIBTENGINE os arch)
    SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${${os}}_${${arch}})
    MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")
    FIND_FILE(TENGINE_INCLUDE_DIR include ${3RDPARTY_DIR} NO_DEFAULT_PATH)
    FIND_FILE(TENGINE_LIBRARY_DIR lib ${3RDPARTY_DIR} NO_DEFAULT_PATH)

    SET(TENGINE_LIBS
        tengine-lite-static
        #PARENT_SCOPE no parent
    )
    IF(TENGINE_INCLUDE_DIR)
        SET(TENGINE_LIBRARY_DIR "${TENGINE_LIBRARY_DIR}/tengine")
        MESSAGE(STATUS "TENGINE_INCLUDE_DIR : ${TENGINE_INCLUDE_DIR}")
        MESSAGE(STATUS "TENGINE_LIBRARY_DIR : ${TENGINE_LIBRARY_DIR}")
        MESSAGE(STATUS "TENGINE_LIBS : ${TENGINE_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "TENGINE_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_JSONCPP os arch)
    SET(3RDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty/target/${${os}}_${${arch}})
    MESSAGE(STATUS "3RDPARTY_DIR: ${3RDPARTY_DIR}")
    FIND_FILE(JSONCPP_INCLUDE_DIR include ${3RDPARTY_DIR} NO_DEFAULT_PATH)
    FIND_FILE(JSONCPP_LIBRARY_DIR lib ${3RDPARTY_DIR} NO_DEFAULT_PATH)

    SET(JSONCPP_LIBS
        jsoncpp
        #PARENT_SCOPE no parent
    )
    IF(JSONCPP_INCLUDE_DIR)
        SET(JSONCPP_LIBRARY_DIR "${JSONCPP_LIBRARY_DIR}/jsoncpp")
        MESSAGE(STATUS "JSONCPP_INCLUDE_DIR : ${JSONCPP_INCLUDE_DIR}")
        MESSAGE(STATUS "JSONCPP_LIBRARY_DIR : ${JSONCPP_LIBRARY_DIR}")
        MESSAGE(STATUS "JSONCPP_LIBS : ${JSONCPP_LIBS}")
    ELSE()
        MESSAGE(FATAL_ERROR "JSONCPP_LIBS not found!")
    ENDIF()
ENDMACRO()

MACRO(LOAD_3RDPARTY os arch)
    LOAD_HEADER_ONLY()
    LOAD_LIBSPDLOG(${os} ${arch})
    LOAD_LIBMNN(${os} ${arch})
    LOAD_LIBNCNN(${os} ${arch})
    LOAD_LIBTENGINE(${os} ${arch})
    LOAD_JSONCPP(${os} ${arch})
ENDMACRO()

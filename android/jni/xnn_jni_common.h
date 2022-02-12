//
// Created by Ethan on 2022/02/08.
//
#ifndef XNN_SDK_JNI_XNN_JNI_COMMON_H
#define XNN_SDK_JNI_XNN_JNI_COMMON_H

#include <jni.h>
#include <string>
#include "common.hpp"

#include <android/log.h>
// #ifdef __cplusplus
// extern "C" {
// #endif

#define XNN_Result_Desc_SIG             "com/biedamingming/xnn/XnnResultDesc"

#define XNN_Handle_SIG                  "com/biedamingming/xnn/XnnHandle"

#define XNN_Image_Format_SIG            "com/biedamingming/xnn/XnnImage$ImageFormat"

#define XNN_Obj_Detect_Result_SIG      "com/biedamingming/xnn/detect/common/XnnObjDetectResultDesc"

#define XNN_Obj_Detect_Position_SIG    "com/biedamingming/xnn/detect/common/XnnObjPosition"

#define XNN_Position_SIG                "com/biedamingming/xnn/detect/common/XnnPosition"

#define LOG_D(message) __android_log_print(ANDROID_LOG_DEBUG, \
            __FUNCTION__, "Line %d: %s\n", __LINE__, message);

#define LOG_E(message) __android_log_print(ANDROID_LOG_ERROR, \
            __FUNCTION__, "Line %d: %s\n", __LINE__, message);

#define DETECTION_PATH "/detection"


jobject jni_success(JNIEnv *env);

jobject jni_error(JNIEnv *env);

jobject getStatus(JNIEnv *env, jint status);

jlong getResultDesc(JNIEnv *env, jobject result_desc);

jlong getHandle(JNIEnv *env, jobject handle);


XNNPixelFormat switchPixFormat(jint value);


// #ifdef __cplusplus
// };
// #endif

#endif //XNN_SDK_JNI_XNN_JNI_COMMON_H

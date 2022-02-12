//
// Created by Ethan on 2022/02/08.
//
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <unistd.h>
#include <sys/stat.h>
#include "xnn_jni_common.h"
#include "common.hpp"
#include <fstream>
#include <mutex>
// #ifdef __cplusplus
// extern "C" {
// #endif

jobject jni_success(JNIEnv *env) {
    return getStatus(env, 0);
}

jobject jni_error(JNIEnv *env) {
    return getStatus(env, -1001);
}

jobject getStatus(JNIEnv *env, jint index) {
    jobject xnn_status;
    jclass clazz = env->FindClass("com/biedamingming/xnn/XnnStatus");
    if (NULL == clazz) {
        LOG_E("Find class failed");
        return NULL;
    }
    jmethodID method_status = env->GetMethodID(clazz, "<init>", "(I)V");
    if (method_status == NULL) {
        LOG_E("Get method failed");
        return NULL;
    }
    xnn_status = env->NewObject(clazz, method_status, index);
    return xnn_status;
}


jlong getHandle(JNIEnv *env, jobject handle) {
    static jclass handle_clazz = NULL;
    if (NULL == handle_clazz) {
        jclass handle_tmp = env->GetObjectClass(handle);
        if (NULL == handle_tmp) {
            LOG_E("Find Class DetectHandle Failed.");
            return (jlong)0L;
        }
        handle_clazz = (jclass)env->NewGlobalRef(handle_tmp);
        env->DeleteLocalRef(handle_tmp);
    }
    jfieldID handle_fieldID = env->GetFieldID(handle_clazz, "p_handle", "J");
    jlong j_handle = env->GetLongField(handle, handle_fieldID);
    return j_handle;
}


XNNPixelFormat switchPixFormat(jint value) {
    switch (value) {
        case 0:
            return XNN_PIX_RGBA;
        case 1:
            return XNN_PIX_RGB;
        case 2:
            return XNN_PIX_BGR;
        case 3:
            return XNN_PIX_GRAY;
        case 4:
            return XNN_PIX_BGRA;
        case 5:
            return XNN_PIX_YUV_NV21;
        case 6:
            return XNN_PIX_YUV_NV12;
        case 7:
            return XNN_PIX_YUV_420P;
        case 8:
            return XNN_PIX_INVALID;
        case 9:
            return XNN_PIX_RGB2GRAY;
        case 10:
            return XNN_PIX_BGR2GRAY;
        case 11:
            return XNN_PIX_BGR2RGB;
        default:
            __android_log_print(ANDROID_LOG_ERROR, __FUNCTION__,
                                "Line[%d] Do not support this pixel format\n", __LINE__);
            return XNN_PIX_INVALID;
    }
}

static void createDir(std::string path) {
    if (access(path.c_str(), 0) != -1) {
        return;
    }
    mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

static void createDirs(std::string dir_path) {
    dir_path += "/files";
    createDir(dir_path);
    dir_path += "/biedamingming";
    createDir(dir_path);
    dir_path += "/models";
    createDir(dir_path);
}

static jstring getPackageName(JNIEnv *env, jobject context) {
    jclass context_clazz = env->GetObjectClass(context);
    jmethodID mId = env->GetMethodID(context_clazz, "getPackageName", "()Ljava/lang/String;");
    jstring packName = static_cast<jstring>(env->CallObjectMethod(context, mId));
    return packName;
}

std::string getPath(JNIEnv *env, jobject context) {
    jstring pkg_name = getPackageName(env, context);
    std::string package_name = env->GetStringUTFChars(pkg_name, 0);
    return "/data/data/" + package_name + "/files/biedamingming/";
}


// #ifdef __cplusplus
// };
// #endif


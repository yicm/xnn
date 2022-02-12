#include "jni_caching.h"

#include <jni.h>

#include "xnn_jni_common.h"
#include <android/log.h>

jboolean FindClass(JNIEnv *env, const char *name, jclass *clazz_out) {
    jclass clazz = env->FindClass(name);
    if (NULL == clazz) {
        LOG_E("can't find class");
        return JNI_FALSE;
    }
    *clazz_out = (jclass) env->NewGlobalRef(clazz);
    return JNI_TRUE;
}

jboolean GetField(JNIEnv *env, jclass *clazz, const char *name, const char *sig, jfieldID *field_out) {
    jfieldID filed = env->GetFieldID(*clazz, name, sig);
    if (filed == nullptr) {
        LOG_E("can not find filed name");
        return JNI_FALSE;
    }
    *field_out = filed;
    return JNI_TRUE;
}


CstructCacheHeader cstruct_cache_header;
static void CachingCstruct(JNIEnv *env) {
    jboolean ret = FindClass(env, "com/biedamingming/xnn/XnnObjDetectResult", &cstruct_cache_header.clz);
    jclass clazz = cstruct_cache_header.clz;
    // Get constructor method
    cstruct_cache_header.constructor = env->GetMethodID(clazz, "<init>", "()V");
    if (!cstruct_cache_header.constructor) {
        LOG_E("Failed to get method id");
    }
    // Get all members field id
    GetField(env, &clazz, "label", "I", &cstruct_cache_header.jid_label);
    GetField(env, &clazz, "score", "F", &cstruct_cache_header.jid_score);

    GetField(env, &clazz, "rect", "Landroid/graphics/Rect;", &cstruct_cache_header.jid_rect);
}

MyRect rect;
static void CachingRect(JNIEnv *env) {
    // Rect
    FindClass(env, "android/graphics/Rect", &rect.clz);
    rect.constructor = env->GetMethodID(rect.clz, "<init>", "()V");
    if (!rect.constructor) {
        LOG_E("Failed to get method id");
    }
    GetField(env, &rect.clz, "left", "I", &rect.left);
    GetField(env, &rect.clz, "top", "I", &rect.top);
    GetField(env, &rect.clz, "right", "I", &rect.right);
    GetField(env, &rect.clz, "bottom", "I", &rect.bottom);
}


void InitCaching(JNIEnv *env) {
    CachingRect(env);
    CachingCstruct(env);
}

void UninitCaching(JNIEnv *env) {
    //env->DeleteGlobalRef(awt_point.clz);
    env->DeleteGlobalRef(cstruct_cache_header.clz);
    env->DeleteGlobalRef(rect.clz);
}
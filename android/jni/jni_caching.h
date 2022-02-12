#ifndef _TEST_JNI_CACHING_H_
#define _TEST_JNI_CACHING_H_

#include <jni.h>

// android.graphics.Rect
typedef struct MyRect_t {
    jclass clz;
    jfieldID left;
    jfieldID top;
    jfieldID right;
    jfieldID bottom;
    jmethodID constructor;
} MyRect;

// com.biedamingming.xnn.XnnObjDetectResult
typedef struct CstructCacheHeader_t {
    jclass clz;
    jfieldID jid_label;
    jfieldID jid_score;
    jfieldID jid_rect;
    jmethodID constructor;
} CstructCacheHeader;

void InitCaching(JNIEnv *env);

void UninitCaching(JNIEnv *env);

#endif
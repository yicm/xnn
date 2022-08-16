//
// Created by Ethan on 2022/02/08.
//
#include "com_biedamingming_xnn_ObjDetect.h"
#include "xnn_jni_common.h"
#include <unistd.h>
#include <string.h>
#include "ncnn_detect.hpp"
#include "jni_caching.h"


#ifdef __cplusplus
extern "C" {
#endif

static jobject combineDetectResult(JNIEnv *env, std::vector<DetectObject> result);

// Is automatically called once the native code is loaded via System.loadLibary(...);
jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOG_E("JNI_OnLoad");
    JNIEnv *env = NULL;
    jint result = JNI_ERR;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        LOG_E("Failed to get env");
        return result;
    } else {
        InitCaching(env);
    }

    return JNI_VERSION_1_6;
}

// Is automatically called once the Classloader is destroyed
void JNI_OnUnload(JavaVM *vm, void *reserved) {
    LOG_E("JNI_OnUnLoad");
    JNIEnv *env = NULL;
    jint result = JNI_ERR;
    if (vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
        LOG_E("Failed to get env");
        return;
    } else {
        UninitCaching(env);
    }

}

/*
 * Class:     com_biedamingming_xnn_ObjDetect
 * Method:    initHandle
 * Signature: (Lcom/biedamingming/xnn/common/XnnObjDetectHandle;Landroid/content/res/AssetManager;Landroid/content/Context;)Lcom/biedamingming/xnn/XnnStatus;
 */
JNIEXPORT jobject JNICALL Java_com_biedamingming_xnn_ObjDetect_initHandle
  (JNIEnv *env, jobject, jobject obj_detect_handle, jobject jmgr, jobject jcontext) {
    LOG_E("Start to new NCNNDetect.");
    int num_class = 1;
    std::string param_path = "output-detect-nobn.param";
    std::string bin_path = "output-detect-nobn.bin";
    int target_size = 256;
    bool is_load_param_bin = false;
    AAssetManager *cmgr = AAssetManager_fromJava(env, jmgr);

    NCNNDetect *ncnn_detect = new NCNNDetect();
    if (!ncnn_detect) {
        LOG_E("Failed to new NCNNDetect.");
        return jni_error(env);
    }
    LOG_E("Success to new NCNNDetect");

    if (!ncnn_detect->init(num_class, param_path, bin_path, target_size, is_load_param_bin, cmgr)) {
        LOG_E("Failed to init.");
        return jni_error(env);
    }
    LOG_E("Success to init NCNNDetect");

    jclass detect_handle_clazz = env->GetObjectClass(obj_detect_handle);
    if (NULL == detect_handle_clazz) {
        LOG_E("Find Class DetectHandle Failed.");
        return jni_error(env);
    }
    LOG_E("Success to find class detecthandle");
    jfieldID fp_work_handle = env->GetFieldID(detect_handle_clazz, "p_handle", "J");
    if (NULL == fp_work_handle) {
        LOG_E("Get FieldID p_work_handle Failed.");
        return jni_error(env);
    }
    env->SetLongField(obj_detect_handle, fp_work_handle, (jlong)ncnn_detect);
    return jni_success(env);
}

/*
 * Class:     com_biedamingming_xnn_ObjDetect
 * Method:    run
 * Signature: (Lcom/biedamingming/xnn/common/XnnObjDetectHandle;Lcom/biedamingming/xnn/XnnImage;Ljava/lang/Boolean;Lcom/biedamingming/xnn/XnnObjDetectResult;)Lcom/biedamingming/xnn/XnnStatus;
 */
JNIEXPORT jobject
JNICALL Java_com_biedamingming_xnn_ObjDetect_run
  (JNIEnv *env, jobject, jobject obj_detect_handle, jobject image, jobject has_obj) {
  // Get Handle
  jlong j_obj_detect_handle = getHandle(env, obj_detect_handle);
  NCNNDetect *p_detect_handle = (NCNNDetect*)j_obj_detect_handle;
  if (!p_detect_handle) {
      LOG_E("Pointer of ObjDetectHandle is Invalid.");
      return jni_error(env);
  }

  // Get Image
  static jclass image_in_clazz = NULL;
  if (NULL == image_in_clazz) {
      jclass image_in_tmp = env->GetObjectClass(image);
      if (NULL == image_in_tmp) {
          LOG_E("Find Class XnnImage failed.");
          return jni_error(env);
      }
      image_in_clazz = (jclass)env->NewGlobalRef(image_in_tmp);
      env->DeleteLocalRef(image_in_tmp);
  }
  jfieldID j_data_fieldID = env->GetFieldID(image_in_clazz, "data", "[B");
  jfieldID j_width_fieldID = env->GetFieldID(image_in_clazz, "width", "I");
  jfieldID j_height_fieldID = env->GetFieldID(image_in_clazz, "height", "I");
  if (NULL == j_data_fieldID || NULL == j_width_fieldID ||
          NULL == j_height_fieldID) {
      LOG_E("Get FieldID failed.");
      return jni_error(env);
  }
  jbyteArray j_data_array = (jbyteArray)env->GetObjectField(image, j_data_fieldID);
  if (NULL == j_data_array) {
      LOG_E("Get byteArray failed.");
      return jni_error(env);
  }
  jbyte *image_data = env->GetByteArrayElements(j_data_array, JNI_FALSE);
  jint width = env->GetIntField(image, j_width_fieldID);
  jint height = env->GetIntField(image, j_height_fieldID);
  if (NULL == image_data) {
      LOG_E("Get image_data Failed.");
      return jni_error(env);
  }
  jmethodID j_get_value_methodID = env->GetMethodID(image_in_clazz, \
          "getType", "()I");
  // get image format
  jint img_format_value = env->CallIntMethod(image, j_get_value_methodID);
  XNNPixelFormat image_format = switchPixFormat(img_format_value);

  XNNImage input_image = {
          .data = (unsigned char*)image_data,
          .src_pixel_format = image_format,
          .width = (unsigned int)width,
          .height = (unsigned int)height,
  };
  // ncnn 默认输入是RGB  todo: bgr 转 rgb, 如果已经是rgb了，直接填写PIXEL_RGB
  input_image.src_pixel_format = XNN_PIX_BGR2RGB;
  // Run
  int TOPK = 1;
  std::vector<DetectObject> result;
  XNNStatus status =  p_detect_handle->run(&input_image, result, TOPK);

  if (XNN_SUCCESS != status) {
      env->ReleaseByteArrayElements(j_data_array, image_data, 0);
      LOG_E("ObjDetectRun Failed.");
      return getStatus(env, status);
  }
  // Set has obj
  static jclass has_obj_clazz = NULL;
  if (NULL == has_obj_clazz) {
      jclass has_obj_tmp = env->GetObjectClass(has_obj);
      if (NULL == has_obj_tmp) {
          env->ReleaseByteArrayElements(j_data_array, image_data, 0);
          LOG_E("GetObjectClass Failed.");
          return jni_error(env);
      }
      has_obj_clazz = (jclass)env->NewGlobalRef(has_obj_tmp);
      env->DeleteLocalRef(has_obj_tmp);
  }
  jfieldID has_obj_value_fieldID = env->GetFieldID(has_obj_clazz, "value", "Z");
  if (NULL == has_obj_value_fieldID) {
      env->ReleaseByteArrayElements(j_data_array, image_data, 0);
      LOG_E("GetFieldID Failed.");
      return jni_error(env);
  }
  jboolean obj = (jboolean) ((result.size() > 0));
  env->SetBooleanField(has_obj, has_obj_value_fieldID, obj);
  // release java image data
  env->ReleaseByteArrayElements(j_data_array, image_data, 0);
  // combine result and return
  return combineDetectResult(env, result);
}

extern MyRect rect;
extern CstructCacheHeader cstruct_cache_header;

static jobject combineDetectResult(JNIEnv *env, std::vector<DetectObject> result) {
    // Create CStruct: If you created an object externally(Java Layer), you don't need to create it here.
    jobject j_struct = env->NewObject(cstruct_cache_header.clz, cstruct_cache_header.constructor);
    LOG_D("Create CStruct Successfully");
    if (result.size() <= 0) {
        return j_struct;
    }
    // Create Rect
    jobject j_rect = env->NewObject(rect.clz, rect.constructor);
    env->SetIntField(j_rect, rect.left, result[0].rect.x);
    env->SetIntField(j_rect, rect.top, result[0].rect.y);
    env->SetIntField(j_rect, rect.right, result[0].rect.x + result[0].rect.width);
    env->SetIntField(j_rect, rect.bottom, result[0].rect.y + result[0].rect.height);
    LOG_E("Create Rect Successfully");
    // set values
    env->SetIntField(j_struct, cstruct_cache_header.jid_label, result[0].label);
    env->SetFloatField(j_struct, cstruct_cache_header.jid_score, result[0].prob);
    env->SetObjectField(j_struct, cstruct_cache_header.jid_rect, j_rect);
    return j_struct;
}

/*
 * Class:     com_biedamingming_xnn_ObjDetect
 * Method:    destroyHandle
 * Signature: (Lcom/biedamingming/xnn/common/XnnObjDetectHandle;)Lcom/biedamingming/xnn/XnnStatus;
 */
JNIEXPORT jobject JNICALL Java_com_biedamingming_xnn_ObjDetect_destroyHandle
  (JNIEnv *env, jobject, jobject handle) {
    jlong j_handle = getHandle(env, handle);
    NCNNDetect *p_handle = (NCNNDetect*)j_handle;
    if (!p_handle) {
        LOG_E("Pointer of ObjDetectHandle is Invalid");
        return jni_error(env);
    }
    p_handle->release();

    delete p_handle;
    return jni_success(env);
}

#ifdef __cplusplus
}
#endif
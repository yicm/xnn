package com.biedamingming.xnn;

import android.content.Context;

import com.biedamingming.xnn.XnnUtils;
import com.biedamingming.xnn.XnnImage;
import com.biedamingming.xnn.XnnStatus;
import com.biedamingming.xnn.common.XnnObjDetectHandle;

import android.content.res.AssetManager;

/**
 * Created by Ethan on 2022-02-08.
 */

public class ObjDetect {
    static {
        XnnUtils.loadLibrary();
    }


    /**
     * Initialize the handle of obj detection module.
     * @param objDetectHandle The handle of obj detection module.
     * @param context The context of android application.
     * @return The status of running.
     * @note * Please call destroyHandle() after calling this function.
     * @note * XnnObjDetectHandle is none-thread-safe. Please don't use the same handle in different threads.
     */
    public native XnnStatus initHandle(XnnObjDetectHandle objDetectHandle, AssetManager mgr, Context context);

    /**
     * Obj detection function which is able to detect one best obj.
     * @param objDetectHandle The handle of obj detection module.
     * @param image The input image.
     * @param trackMode Detection mode: false for static image detection and true for tracking.
     * @param hasObj [out] 0 or 1, 0 if no obj is detected.
     * @param objDetectResultDesc The obj detection result descriptor for the best obj in this image.
     * @return The status of running.
     * @note * Calling initHandle() and initResultDesc() first.
     */
    public native XnnObjDetectResult run(
            XnnObjDetectHandle objDetectHandle,
            XnnImage image,
            Boolean hasObj);


    /**
     * Destroy handle of obj detection module.
     * @param objDetectHandle The handle of obj detection module.
     * @return The status of running.
     * @note * Call this function once after calling initHandle.
     */
    public native XnnStatus destroyHandle(XnnObjDetectHandle objDetectHandle);
}

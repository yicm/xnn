package com.biedamingming.xnn;

/**
 * Created by Ethan on 2022-02-08.
 */

public class XnnUtils {
    static {
        System.out.println("java.library.path: " + System.getProperty("java.library.path"));
        try {
            System.loadLibrary("jni_xnn_sdk");
        } catch (Throwable ex) {
            ex.printStackTrace();
        }
    }

    static public void loadLibrary() {};

    /**
     * Rotation of XnnImage.
     * @param srcImage Source image.
     * @param dstImage Destination image.
     * @param degree The rotation degree. Only support 0, 90, 180, 270 degree.
     * @return The status of running.
     */
    static public native XnnStatus imageRotate(XnnImage srcImage, XnnImage dstImage, int degree);

    /**
     * Get XNN-SDK Version.
     * @return The XNN-SDK version.
     */
    static public native String getXnnVersion();
}

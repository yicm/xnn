package com.biedamingming.xnn;

import java.util.Arrays;

/**
 * Created by Ethan on 2022-02-08.
 */

public class XnnImage {
    public enum  ImageFormat{
        XNN_PIX_FMT_GRAY8(0),
        XNN_PIX_FMT_YUV420P(1),
        XNN_PIX_FMT_NV12(2),
        XNN_PIX_FMT_NV21(3),
        XNN_PIX_FMT_BGRA8888(4),
        XNN_PIX_FMT_BGR888(5);

        private int value;
        ImageFormat(int value) {
            this.value = value;
        }
        public int getValue() {
            return value;
        }
    }

    @Override
    public String toString() {
        return "XnnImage{" +
                "data=" + Arrays.toString(data) +
                ", format=" + format +
                ", width=" + width +
                ", height=" + height +
                '}';
    }

    public XnnImage(byte[] data, ImageFormat format, int width, int height) {
        this.data = data;
        this.format = format;
        this.width = width;
        this.height = height;
    }

    public int getType() {
        return this.format.getValue();
    }

    public byte[] data;
    public ImageFormat format;
    public int width;
    public int height;
}

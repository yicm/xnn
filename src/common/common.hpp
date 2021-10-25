#ifndef _XNN_COMMON_COMMON_HPP_
#define _XNN_COMMON_COMMON_HPP_

#ifdef BUILD_WITH_MNN
#include "MNN/ImageProcess.hpp"
#endif

#include "ncnn/net.h"

// A enum to represent pixel format
typedef enum XNNPixelFormat {
    XNN_PIX_RGBA = 0,
    XNN_PIX_RGB,
    XNN_PIX_BGR,
    XNN_PIX_GRAY,
    XNN_PIX_BGRA,
    XNN_PIX_YUV_NV21,
    XNN_PIX_YUV_NV12,
    XNN_PIX_YUV_420P,
    XNN_PIX_INVALID,
} XNNPixelFormat;

// A structure to represent image
typedef struct XNNImage {
    unsigned char *data;
    XNNPixelFormat src_pixel_format;
    XNNPixelFormat dst_pixel_format;
    unsigned int width;
    unsigned int height;
} XNNImage, *XNNImageDesc;

// A enum to represent running status
typedef enum XNNStatus {
    XNN_SUCCESS  = 0,
    XNN_PARAM_ERROR = -1,
    XNN_INVALID_HANDLE = -2,
    XNN_INVALID_PIXEL_FORMAT = -3,
    XNN_FILE_NOT_FOUND = -4,
    XNN_INVALID_MODEL_FILE_FORMAT = -5,
    XNN_OPENDIR_FAILED = -6,
    XNN_OPEN_FILE_FAILED = -7,
    XNN_INVALID_CONFIG_PARAM = -8,
    XNN_RUN_ERROR = -9
} XNNStatus;

#ifdef BUILD_WITH_MNN
MNN::CV::ImageFormat convertXNNPixFormat2MNN(XNNPixelFormat format);
#endif

ncnn::Mat::PixelType convertXNNPixFormat2NCNN(XNNPixelFormat format);

float* softmax(float* arr, const int len);

#endif


#include "common/common.hpp"

#include <math.h>

#ifdef BUILD_WITH_MNN
MNN::CV::ImageFormat convertXNNPixFormat2MNN(XNNPixelFormat format)
{
    switch (format)
    {
        case XNN_PIX_RGBA:
            return MNN::CV::RGBA;

        case XNN_PIX_RGB:
            return MNN::CV::RGB;

        case XNN_PIX_BGR:
            return MNN::CV::BGR;

        case XNN_PIX_GRAY:
            return MNN::CV::GRAY;

        case XNN_PIX_BGRA:
            return MNN::CV::BGRA;

        case XNN_PIX_YUV_NV21:
            return MNN::CV::YUV_NV21;

        default:
            fprintf(stderr, "Do not support the XNNPixelFormat: %d\n", format);
            return MNN::CV::GRAY;
    }
}
#endif

ncnn::Mat::PixelType convertXNNPixFormat2NCNN(XNNPixelFormat format)
{
 switch (format)
    {
        case XNN_PIX_RGBA:
            return ncnn::Mat::PIXEL_RGBA;

        case XNN_PIX_RGB:
            return ncnn::Mat::PIXEL_RGB;

        case XNN_PIX_BGR:
            return ncnn::Mat::PIXEL_BGR;

        case XNN_PIX_GRAY:
            return ncnn::Mat::PIXEL_GRAY;

        case XNN_PIX_BGRA:
            return ncnn::Mat::PIXEL_BGRA;

        case XNN_PIX_RGB2GRAY:
            return ncnn::Mat::PIXEL_RGB2GRAY;

        case XNN_PIX_BGR2GRAY:
            return ncnn::Mat::PIXEL_BGR2GRAY;

        case XNN_PIX_BGR2RGB:
            return ncnn::Mat::PIXEL_BGR2RGB;

        default:
            fprintf(stderr, "Do not support the XNNPixelFormat: %d\n", format);
            return ncnn::Mat::PIXEL_GRAY;
    }
}

void softmax(float* arr, const int len) {
    float softmax_denominator = 0;

    for (int i = 0; i < len; i++) {
         softmax_denominator += exp(arr[i]);
    }

    for (int i = 0; i < len; i++) {
        float computed_value = exp(arr[i]) / softmax_denominator;
        arr[i] = computed_value;
    }
}

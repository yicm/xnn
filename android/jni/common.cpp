#include "common.hpp"

#include <math.h>



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


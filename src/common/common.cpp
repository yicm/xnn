#include "common/common.hpp"

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
            return MNN::CV::GRAY;
    }
}
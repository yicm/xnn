#ifndef _XNN_COMMON_COMMON_HPP_
#define _XNN_COMMON_COMMON_HPP_

#ifdef BUILD_WITH_MNN
#include "MNN/ImageProcess.hpp"
#endif

#include "ncnn/net.h"

#include <limits.h>

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
    XNN_PIX_RGB2GRAY,
    XNN_PIX_BGR2GRAY,
    XNN_PIX_BGR2RGB,
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

void softmax(float* arr, const int len);


// simple opencv
template<typename _Tp>
static inline _Tp saturate_cast(int v)
{
    return _Tp(v);
}
template<>
inline unsigned char saturate_cast<unsigned char>(int v)
{
    return (unsigned char)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0);
}


template<typename _Tp>
struct Point_
{
    Point_()
        : x(0), y(0)
    {
    }
    Point_(_Tp _x, _Tp _y)
        : x(_x), y(_y)
    {
    }

    template<typename _Tp2>
    operator Point_<_Tp2>() const
    {
        return Point_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y));
    }

    _Tp x;
    _Tp y;
};

typedef Point_<int> Point;
typedef Point_<float> Point2f;

template<typename _Tp>
struct Size_
{
    Size_()
        : width(0), height(0)
    {
    }
    Size_(_Tp _w, _Tp _h)
        : width(_w), height(_h)
    {
    }

    template<typename _Tp2>
    operator Size_<_Tp2>() const
    {
        return Size_<_Tp2>(saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
    }

    _Tp width;
    _Tp height;
};

typedef Size_<int> Size;
typedef Size_<float> Size2f;

template<typename _Tp>
struct Rect_
{
    Rect_()
        : x(0), y(0), width(0), height(0)
    {
    }
    Rect_(_Tp _x, _Tp _y, _Tp _w, _Tp _h)
        : x(_x), y(_y), width(_w), height(_h)
    {
    }
    Rect_(Point_<_Tp> _p, Size_<_Tp> _size)
        : x(_p.x), y(_p.y), width(_size.width), height(_size.height)
    {
    }

    template<typename _Tp2>
    operator Rect_<_Tp2>() const
    {
        return Rect_<_Tp2>(saturate_cast<_Tp2>(x), saturate_cast<_Tp2>(y), saturate_cast<_Tp2>(width), saturate_cast<_Tp2>(height));
    }

    _Tp x;
    _Tp y;
    _Tp width;
    _Tp height;

    // area
    _Tp area() const
    {
        return width * height;
    }
};

template<typename _Tp>
static inline Rect_<_Tp>& operator&=(Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
    a.width = std::min(a.x + a.width, b.x + b.width) - x1;
    a.height = std::min(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    if (a.width <= 0 || a.height <= 0)
        a = Rect_<_Tp>();
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp>& operator|=(Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    _Tp x1 = std::min(a.x, b.x), y1 = std::min(a.y, b.y);
    a.width = std::max(a.x + a.width, b.x + b.width) - x1;
    a.height = std::max(a.y + a.height, b.y + b.height) - y1;
    a.x = x1;
    a.y = y1;
    return a;
}

template<typename _Tp>
static inline Rect_<_Tp> operator&(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c &= b;
}

template<typename _Tp>
static inline Rect_<_Tp> operator|(const Rect_<_Tp>& a, const Rect_<_Tp>& b)
{
    Rect_<_Tp> c = a;
    return c |= b;
}

typedef Rect_<int> Rect;
typedef Rect_<float> Rect2f;

typedef struct DetectObject
{
    Rect_<float> rect;
    int label;
    float prob;
} DetectObject;


#endif


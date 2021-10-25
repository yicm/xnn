#ifndef _XNN_TENGINE_TENGINE_CLAZZ_HPP_
#define _XNN_TENGINE_TENGINE_CLAZZ_HPP_

#include "common/common.hpp"

#include <memory>
#include <vector>

#include "tengine/c_api.h"


namespace xnn
{
    typedef struct Image
    {
        int w;
        int h;
        int c;
        float* data;
    } Image;

    class TengineClazz
    {

    public:
        bool init(int num_class,
                  std::vector<float> &means,
                  std::vector<float> &scales,
                  std::string model_file,
                  int input_size,
                  int tengine_mode = TENGINE_MODE_FP32,
                  bool has_softmax = false);

        XNNStatus run(XNNImage *image, std::vector<std::pair<int, float>>& result, int topk = 5);

        void release();

        Image makeImage(int w, int h, int c);

        Image makeEmptyImage(int w, int h, int c);

        Image loadImage(XNNImage *image);

        void freeImage(Image m);

        void tengineResizeF32(float* data, float* res, int ow, int oh, int c, int h, int w);

        Image imread2caffe(Image resImg, int img_w, int img_h, float* means, float* scale);

        void getInputData(XNNImage *image, float* input_data, std::vector<float> &means, std::vector<float> &scales);

    private:
        graph_t  graph_;
        tensor_t input_tensor_;
        struct options opt;
        int tengine_mode_;

        int num_class_;
        int input_size_;
        std::vector<float> means_;
        std::vector<float> scales_;
        bool has_softmax_;

        float* input_float32_data_;
        int8_t* input_int8_data_;

    };
} // namespace xnn

#endif
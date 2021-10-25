#ifndef _XNN_TENGINE_TENGINE_CLAZZ_HPP_
#define _XNN_TENGINE_TENGINE_CLAZZ_HPP_

#include "common/common.hpp"

#include <memory>
#include <vector>

#include "tengine/c_api.h"


namespace xnn
{

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

    private:
        graph_t  graph_;
        tensor_t input_tensor_;
        struct options opt;

        int num_class_;
        int input_size_;
        std::vector<float> means_;
        std::vector<float> scales_;
        int tengine_mode_;
        bool has_softmax_;

    };
} // namespace xnn

#endif
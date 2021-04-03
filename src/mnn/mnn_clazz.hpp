#ifndef _XNN_MNN_MNN_CLAZZ_HPP_
#define _XNN_MNN_MNN_CLAZZ_HPP_

#include "common/common.hpp"

#include <memory>

#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/expr/Expr.hpp"
#include "MNN/expr/ExprCreator.hpp"

namespace xnn
{

    class MNNClazz
    {

    public:
        bool init(std::string model_path = "");

        XNNStatus run(XNNImage *image, std::vector<std::pair<int, float>>& result, int topk = 5);

        void release();

    private:
        MNN::Interpreter * interpreter_ = nullptr;
        MNN::ScheduleConfig schedule_config_;
        MNN::Session *session_ = nullptr;
        MNN::Tensor *input_tensor_ = nullptr;
        MNN::Tensor *output_tensor_ = nullptr;

        std::vector<int> input_shape_;

        MNN::CV::ImageProcess *pretreat_ = nullptr;

        int output_tensor_size = 0;
        int num_class_;
        
    };
} // namespace xnn

#endif
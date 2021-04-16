#ifndef _XNN_MNN_MNN_CLAZZ_HPP_
#define _XNN_MNN_MNN_CLAZZ_HPP_

#include "common/common.hpp"

#include <memory>
#include <vector>

#include "ncnn/net.h"
#include "ncnn/cpu.h"


namespace xnn
{

    class NCNNClazz
    {

    public:
        bool init(int num_class,
                  std::vector<float> &means,
                  std::vector<float> &normals,
                  std::string param_path,
                  std::string bin_path,
                  bool load_param_bin = true,
                  bool has_softmax = false);

        XNNStatus run(XNNImage *image, std::vector<std::pair<int, float>>& result, int topk = 5);

        void release();
    private:
        void getNetInputName();
        void getNetOutputName();

    private:
        ncnn::Net net_;
        std::vector<ncnn::Blob> blobs_;
        std::vector<ncnn::Layer*> layers_;
        ncnn::Layer *softmax_;
        std::string input_layer_name_;
        std::string output_layer_name_;

        int num_class_;
        std::vector<float> means_;
        std::vector<float> normals_;
        bool has_softmax_;
        bool load_param_bin_;

    };
} // namespace xnn

#endif
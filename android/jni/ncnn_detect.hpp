#ifndef _XNN_NCNN_NCNN_DETECT_HPP_
#define _XNN_NCNN_NCNN_DETECT_HPP_

#include <memory>
#include <vector>

#include "common.hpp"
#include "ncnn/net.h"
#include "ncnn/cpu.h"

#include <android/asset_manager_jni.h>

class NCNNDetect
{
public:
    bool init(int num_class,
                std::string param_path,
                std::string bin_path,
                int input_size,
                bool load_param_bin,
                AAssetManager *mgr);

    XNNStatus run(XNNImage *image, std::vector<DetectObject>& objects, int topk = 5);

    void release();

private:
    ncnn::Net net_;
    std::vector<ncnn::Blob> blobs_;
    std::vector<ncnn::Layer*> layers_;
    std::string input_layer_name_;

    int num_class_;
    int input_size_;
    std::vector<float> means_;
    std::vector<float> normals_;
    bool load_param_bin_;

    float prob_threshold_ = 0.25f;
    float nms_threshold_ = 0.45f;

};

#endif
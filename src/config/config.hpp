#ifndef _XNN_CONFIG_CONFIG_HPP_
#define _XNN_CONFIG_CONFIG_HPP_

#include "jsoncpp/json.h"
#include "common/common.hpp"

#include <vector>

class XNNConfig
{
public:
    static XNNConfig *GetInstance();
    bool hasParsed();
    bool parseConfig(std::string filename = "./config.json");

    // getters
    std::string getModel();
    int getNumClass();
    XNNPixelFormat getSrcFormat();
    XNNPixelFormat getDstFormat();
    std::vector<float> getMeans();
    std::vector<float> getNormal();
    bool hasSoftmax();

private:
    std::string model_;
    int num_class_;
    XNNPixelFormat src_format_;
    XNNPixelFormat dst_format_;
    std::vector<float> means_;
    std::vector<float> normal_;
    bool has_softmax_ = false;

private:
    Json::Value app_root_;
    volatile bool has_parsed_ = false;
};

#endif
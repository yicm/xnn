#include "config/config.hpp"

#include <iostream>
#include <fstream>

XNNConfig *XNNConfig::GetInstance()
{
    static XNNConfig instance;
    return &instance;
}

bool XNNConfig::hasParsed()
{
    return has_parsed_;
}

bool XNNConfig::parseConfig(std::string filename)
{
    std::ifstream ifs(filename, std::ios::in);
    Json::CharReaderBuilder builder;
    // support json comment
    builder["collectComments"] = true;

    JSONCPP_STRING errs;
    if (!parseFromStream(builder, ifs, &app_root_, &errs))
    {
        fprintf(stderr, "Failed to parse from json stream.\n");
        ifs.close();
        return false;
    }
    ifs.close();
    // num_class
    num_class_ = app_root_["num_class"].asInt();
    // src_format
    std::string src_format = app_root_["src_format"].asString();
    if (src_format == "RGB")
    {
        src_format_ = XNN_PIX_RGB;
    }
    else if (src_format == "GRAY")
    {
        src_format_ = XNN_PIX_GRAY;
    }
    else
    {
        fprintf(stderr, "Do not support the source format: %s\n", src_format.c_str());
    }
    // dst_format
    std::string dst_format = app_root_["dst_format"].asString();
    if (dst_format == "RGB")
    {
        dst_format_ = XNN_PIX_RGB;
    }
    else if (dst_format == "GRAY")
    {
        dst_format_ = XNN_PIX_GRAY;
    }
    else
    {
        fprintf(stderr, "Do not support the destination format: %s\n", src_format.c_str());
    }
    // mean
    for (int i = 0; i < app_root_["mean"].size(); i++)
    {
        means_.push_back(app_root_["mean"][i].asFloat());
    }
    // normal
    for (int i = 0; i < app_root_["normal"].size(); i++)
    {
        normal_.push_back(app_root_["normal"][i].asFloat());
    }
    // has_softmax
    has_softmax_ = app_root_["has_softmax"].asBool();

    has_parsed_ = true;
}

int XNNConfig::getNumClass()
{
    return num_class_;
}

XNNPixelFormat XNNConfig::getSrcFormat()
{
    return src_format_;
}

XNNPixelFormat XNNConfig::getDstFormat()
{
    return dst_format_;
}

std::vector<float> XNNConfig::getMeans()
{
    return means_;
}

std::vector<float> XNNConfig::getNormal()
{
    return normal_;
}

bool XNNConfig::hasSoftmax()
{
    return has_softmax_;
}
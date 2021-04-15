#include "ncnn/ncnn_clazz.hpp"

#include <iostream>
#include <memory>

namespace xnn
{
    bool NCNNClazz::init(int num_class,
                         std::vector<float> &means,
                         std::vector<float> &normals,
                         std::string param_path,
                         std::string bin_path,
                         bool load_param_bin,
                         bool has_softmax)
    {
        if (num_class <= 0)
        {
            fprintf(stderr, "Parameter error: invalid number(%d) of class\n", num_class);
            return false;
        }
        if (means.size() <= 0 || normals.size() <= 0)
        {
            fprintf(stderr, "Parameter error: the size of MEANs or NORMALs is 0\n");
            return false;
        }
        // get class number
        num_class_ = num_class;
        // get means and normals
        means_ = means;
        normals_ = normals;
        // setting has_softmax
        has_softmax_ = has_softmax;
        // create interpreter from mnn file
        if (param_path.size() != 0 && bin_path.size() != 0)
        {
            net_.opt.use_vulkan_compute = false;
            if (load_param_bin) {
                net_.load_param_bin(param_path.c_str());
            } else {
                net_.load_param(param_path.c_str());
            }
            net_.load_model(bin_path.c_str());
        }
        else
        {
            fprintf(stderr, "Parameter error: param or bin file can not be empty.\n");
            return false;
        }
        // create softmax layer
        if (!has_softmax_)
        {
            softmax_ = ncnn::create_layer("Softmax");
        }

        return true;
    }

    XNNStatus NCNNClazz::run(XNNImage *image, std::vector<std::pair<int, float>> &result, int topk)
    {
        if (!image || !image->data)
        {
            fprintf(stderr, "Input parameter error\n");
            return XNN_PARAM_ERROR;
        }
        // clear the old result
        result.clear();

        ncnn::Mat in;
        in = ncnn::Mat::from_pixels_resize(image->data, convertXNNPixFormat2NCNN(image->src_pixel_format), image->width, image->height, 64, 64);

        if (means_.size() == 1)
        {
            const float mean_vals[1] = {means_[0]};
            const float norm_vals[1] = {normals_[0]};
            in.substract_mean_normalize(mean_vals, norm_vals);
        }
        else if (means_.size() == 3)
        {
            const float mean_vals[3] = {means_[0], means_[1], means_[2]};
            const float norm_vals[3] = {normals_[0], normals_[1], normals_[2]};
            in.substract_mean_normalize(mean_vals, norm_vals);
        }
        else {
            fprintf(stderr, "Means error\n");
            return XNN_PARAM_ERROR;
        }
        ncnn::Extractor extractor = net_.create_extractor();
        extractor.input(0, in);
        ncnn::Mat out;
        extractor.extract(160, out);
        // manually call softmax on the fc output
        if (!has_softmax_)
        {
            ncnn::ParamDict pd;
            softmax_->load_param(pd);
            softmax_->forward_inplace(out, net_.opt);
            out = out.reshape(out.w * out.h * out.c);
        }
        std::vector<std::pair<int, float>> sorted_result(out.w);
        for (int i = 0; i < out.w; i++)
        {
            sorted_result[i] = std::make_pair(i, out[i]);
        }

        std::sort(sorted_result.begin(), sorted_result.end(),
                  [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });
        for (int i = 0; i < topk; ++i)
        {
            // the first is class index, the seconde is score
            result.push_back(std::make_pair(sorted_result[i].first, sorted_result[i].second));
        }
        return XNN_SUCCESS;
    }

    void NCNNClazz::release()
    {
        delete softmax_;
        softmax_ = nullptr;
    }

} // namespace xnn

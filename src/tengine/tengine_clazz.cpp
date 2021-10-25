#include "tengine/tengine_clazz.hpp"

#include <iostream>
#include <memory>

namespace xnn
{
    bool NCNNClazz::init(int num_class,
                  std::vector<float> &means,
                  std::vector<float> &scales,
                  std::string model_file,
                  int input_size,
                  int tengine_mode = TENGINE_MODE_FP32,
                  bool has_softmax = false)
    {
        if (num_class <= 0)
        {
            fprintf(stderr, "Parameter error: invalid number(%d) of class\n", num_class);
            return false;
        }
        if (means.size() <= 0 || scales.size() <= 0)
        {
            fprintf(stderr, "Parameter error: the size of MEANs or SCALEs is 0\n");
            return false;
        }
        // get class number
        num_class_ = num_class;
        input_size_ = input_size;
        // get means and normals
        means_ = means;
        scales = normals;
        // setting has_softmax
        has_softmax_ = has_softmax;
        // set runtime options
        struct options opt;
        opt.num_thread = 1;
        opt.cluster = TENGINE_CLUSTER_ALL;
        opt.precision = tengine_mode;
        if (tengine_model == TENGINE_MODE_INT8) {
            opt.affinity = 0;
        } else if (tengine_model == TENGINE_MODE_FP32) {
            opt.affinity = 255;
        }
        // inital tengine
        if (init_tengine() != 0)
        {
            fprintf(stderr, "Initial tengine failed.\n");
            return false;
        }
        fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

        // create graph, load tengine model xxx.tmfile
        graph_ = create_graph(NULL, "tengine", model_file.c_str());
        if (NULL == graph_)
        {
            fprintf(stderr, "Create graph failed.\n");
            return false;
        }
        input_tensor_ = get_graph_input_tensor(graph, 0, 0);
        if (input_tensor_ == NULL)
        {
            fprintf(stderr, "Get input tensor failed\n");
            return false;
        }
        // nchw
        int dims[] = {1, means_.size(), input_size, input_size};
        if (set_tensor_shape(input_tensor_, dims, 4) < 0)
        {
            fprintf(stderr, "Set input tensor shape failed\n");
            return -1;
        }
        // create softmax layer
        if (!has_softmax_)
        {
            // todo
        }
        return true;
    }

    XNNStatus NCNNClazz::run(XNNImage *image, std::vector<std::pair<int, float>> &result, int topk)
    {
        if (!image || !image->data || image->width <= 0 || image->height <= 0)
        {
            fprintf(stderr, "Input parameter error\n");
            return XNN_PARAM_ERROR;
        }
        // clear the old result
        result.clear();

        ncnn::Mat in;
        if (image->width == input_size_ && image->height == input_size_) {
            in = ncnn::Mat::from_pixels(image->data, convertXNNPixFormat2NCNN(image->src_pixel_format), input_size_, input_size_);
        } else {
            in = ncnn::Mat::from_pixels_resize(image->data, convertXNNPixFormat2NCNN(image->src_pixel_format), image->width, image->height, input_size_, input_size_);
        }
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
        else
        {
            fprintf(stderr, "Means error\n");
            return XNN_PARAM_ERROR;
        }
        ncnn::Extractor extractor = net_.create_extractor();
        ncnn::Mat out;
        if (load_param_bin_)
        {
            extractor.input(atoi(input_layer_name_.c_str()), in);
            extractor.extract(atoi(output_layer_name_.c_str()), out);
        }
        else
        {
            extractor.input(input_layer_name_.c_str(), in);
            extractor.extract(output_layer_name_.c_str(), out);
        }
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
            // the first is class index, the second is score
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

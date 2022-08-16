#include "ncnn/ncnn_clazz.hpp"

#include <iostream>
#include <memory>

namespace xnn
{
    void NCNNClazz::getNetInputName()
    {
        if (load_param_bin_)
        {
            input_layer_name_ = std::to_string(0);
            return;
        }
        for (size_t i = 0; i < layers_.size(); i++)
        {
            const ncnn::Layer *layer = layers_[i];
            if (layer->type == "Input" && !load_param_bin_)
            {
                for (size_t j = 0; j < layer->tops.size(); j++)
                {
                    int blob_index = layer->tops[j];
                    std::string name = blobs_[blob_index].name.c_str();
                    input_layer_name_ = name;
                }
            }
        }
    }

    void NCNNClazz::getNetOutputName()
    {
        if (load_param_bin_)
        {
            output_layer_name_ = std::to_string(blobs_.size() - 1);
            return;
        }
        for (size_t i = 0; i < layers_.size(); i++)
        {
            const ncnn::Layer *layer = layers_[i];
            for (size_t j = 0; j < layer->bottoms.size(); j++)
            {
                int blob_index = layer->bottoms[j];
                std::string name = blobs_[blob_index + 1].name;
                if (!load_param_bin_ && layer->type == "InnerProduct")
                {
                    output_layer_name_ = name;
                }
            }
        }
    }

    bool NCNNClazz::init(int num_class,
                         std::vector<float> &means,
                         std::vector<float> &normals,
                         std::string param_path,
                         std::string bin_path,
                         int input_size,
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
        input_size_ = input_size;
        // get means and normals
        means_ = means;
        normals_ = normals;
        // setting has_softmax
        has_softmax_ = has_softmax;
        // is param file binary file?
        load_param_bin_ = load_param_bin;
        // create interpreter from param/bin file
        if (param_path.size() != 0 && bin_path.size() != 0)
        {
            net_.opt.use_vulkan_compute = false;
            if (load_param_bin)
            {
                // 加密方式
                net_.load_param_bin(param_path.c_str());
            }
            else
            {
                // 非加密方式
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
        // get input name and output name of net
        blobs_ = net_.mutable_blobs();
        layers_ = net_.layers();


        if(load_param_bin){
            input  	= 0;
            output 	= blobs_.size() - 1;
        }
        else{
            for(size_t i = 0; i < layers_.size(); i++){		// get input & output name
                const ncnn::Layer * layer = layers_[i];

                // get input name
                if(layer->type == "Input"){
                    for (size_t j = 0; j < layer->tops.size(); j++){
                        int blob_index = layer->tops[j];
                        strcpy(input_str, blobs_[blob_index].name.c_str());
                         //blobs_[blob_index].name.copy(input_str, blobs_[blob_index].name.length(), 0);
                         //input_str[blobs_[blob_index].name.length()] = '\0';
                    }
                }

                // get output name

                for (size_t j = 0; j < layer->bottoms.size(); j++){
                    int blob_index = layer->bottoms[j];
                    if(layer->type == "InnerProduct"){
                        //blobs_[blob_index + 1].name.copy(output_str, blobs_[blob_index + 1].name.length(), 0);
                        strcpy(output_str, blobs_[blob_index + 1].name.c_str());
                        //output_str[blobs_[blob_index + 1].name.length()] = '\0';
                    }

                }
            }

        }


        getNetInputName();
        getNetOutputName();
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
        std::cout << "input_layer_name:===" << input_str << "====" << std::endl;
        std::cout << "output_layer_name:===" <<  output_str << "===="<< std::endl;
        std::cout << "input_layer_name:===" << input_layer_name_.c_str() << "====" << std::endl;
        std::cout << "output_layer_name:===" <<  output_layer_name_.c_str() << "===="<< std::endl;
        if (load_param_bin_)
        {
            // 加密方式
            // extractor.input(atoi(input_layer_name_.c_str()), in);
            // extractor.extract(atoi(output_layer_name_.c_str()), out);
            extractor.input(input, in);
            extractor.extract(output, out);
        }
        else
        {
            // 非加密方式
            // extractor.input(input_layer_name_.c_str(), in);
            // extractor.extract(output_layer_name_.c_str(), out);
            extractor.input(input_str, in);
            extractor.extract(output_str, out);
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

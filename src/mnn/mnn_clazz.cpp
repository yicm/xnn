#include "mnn/mnn_clazz.hpp"

#include "config/config.hpp"

#include <iostream>
#include <memory>

namespace xnn
{
    void MNNClazz::init(const std::string model_path, const int width, const int height)
    {
        if (!XNNConfig::GetInstance()->hasParsed()) {
            XNNConfig::GetInstance()->parseConfig();
        }
        num_class_ = XNNConfig::GetInstance()->getNumClass();

        interpreter_ = MNN::Interpreter::createFromFile(model_path.c_str());

        schedule_config_.type = MNN_FORWARD_CPU;
        schedule_config_.numThread = 4;
        session_ = interpreter_->createSession(schedule_config_);
        input_tensor_ = interpreter_->getSessionInput(session_, nullptr);

        auto shape = input_tensor_->shape();
        // the model has not input dimension
        fprintf(stdout, "shape.size = %lu\n", shape.size());
        if (shape.size() == 0)
        {
            shape.resize(4);
        }
        else
        {
            fprintf(stdout, "shape = [%d, %d, %d, %d]\n", shape[0], shape[1], shape[2], shape[3]);
        }
        // set batch to be 1
        shape[0] = 1;

        interpreter_->resizeSession(session_);

        output_tensor_ = interpreter_->getSessionOutput(session_, nullptr);

        output_tensor_size = output_tensor_->elementSize();
    }

    XNNStatus MNNClazz::run(XNNImage *image, std::vector<std::pair<int, float>> &result, int topk)
    {
        if (!image || !image->data)
        {
            fprintf(stderr, "Input parameter error\n");
            return XNN_PARAM_ERROR;
        }
        // clear the old result
        result.clear();
        // set input shape
        auto shape = input_tensor_->shape();
        if (image->pixel_format == XNN_PIX_BGR || image->pixel_format == XNN_PIX_RGB)
        {
            shape[1] = 3;
        }
        if (image->pixel_format == XNN_PIX_GRAY)
        {
            shape[1] = 1;
        }
        shape[2] = image->width;
        shape[3] = image->height;
        // resize input tensor shape
        interpreter_->resizeTensor(input_tensor_, shape);
        // create image pretreat
        if (!pretreat_)
        {
            MNN::CV::ImageProcess::Config config;
            config.sourceFormat = convertXNNPixFormat2MNN(XNNConfig::GetInstance()->getSrcFormat());;
            config.destFormat = convertXNNPixFormat2MNN(image->pixel_format);
            std::vector<float> means = XNNConfig::GetInstance()->getMeans();
            for (int i = 0; i < means.size(); i++) {
                config.mean[i] = means[i];
            }
            std::vector<float> normal = XNNConfig::GetInstance()->getNormal();
            for (int i = 0; i < normal.size(); i++) {
                config.normal[i] = normal[i];
            }

            pretreat_ = MNN::CV::ImageProcess::create(config);
        }
        pretreat_->convert(image->data, image->height, image->width, 0, input_tensor_);
        interpreter_->runSession(session_);

        std::vector<std::pair<int, float>> sorted_result(output_tensor_size);
        {
            // default float value
            auto output_data_ptr = output_tensor_->host<float>();
            // softmax
            if (XNNConfig::GetInstance()->hasSoftmax())
            {
                auto input = MNN::Express::_Input({1, num_class_}, MNN::Express::NCHW);
                auto input_ptr = input->writeMap<float>();
                memcpy(input_ptr, output_data_ptr, num_class_ * sizeof(float));
                auto output_softmax = MNN::Express::_Softmax(input);
                auto got_output = output_softmax->readMap<float>();
                for (int i = 0; i < output_tensor_size; ++i)
                {
                    sorted_result[i] = std::make_pair(i, got_output[i]);
                }
            }
            // non softmax
            {
                for (int i = 0; i < output_tensor_size; ++i)
                {
                    sorted_result[i] = std::make_pair(i, output_data_ptr[i]);
                }
            }
            std::sort(sorted_result.begin(), sorted_result.end(),
                      [](std::pair<int, float> a, std::pair<int, float> b) { return a.second > b.second; });
            for (int i = 0; i < topk; ++i)
            {
                // the first is class index, the seconde is score
                result.push_back(std::make_pair(sorted_result[i].first, sorted_result[i].second));
            }
        }
        return XNN_SUCCESS;
    }

    void MNNClazz::release()
    {
        bool release_status = interpreter_->releaseSession(session_);
        if (!release_status)
        {
            // TODO: log
            fprintf(stderr, "Failed to release session\n");
        }
        MNN::Interpreter *interpreter_ = nullptr;
        MNN::ScheduleConfig schedule_config_;
        delete interpreter_;
        interpreter_ = nullptr;
        session_ = nullptr;
    }

} // namespace xnn
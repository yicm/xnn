#include "mnn/mnn_clazz.hpp"

#include "config/config.hpp"

#include <iostream>
#include <memory>

namespace xnn
{
    bool MNNClazz::init(std::string model_path)
    {
        if (!XNNConfig::GetInstance()->hasParsed()) {
            if (!XNNConfig::GetInstance()->parseConfig()) {
                return false;
            }
        }
        // get class number
        num_class_ = XNNConfig::GetInstance()->getNumClass();
        // create interpreter from mnn file
        if (model_path.size() != 0) {
            interpreter_ = MNN::Interpreter::createFromFile(model_path.c_str());
        } else {
            interpreter_ = MNN::Interpreter::createFromFile(XNNConfig::GetInstance()->getModel().c_str());
        }
        if (interpreter_ == nullptr) {
            fprintf(stderr, "Failed to create interpreter from file: %s, maybe this file doesn't exist.\n", XNNConfig::GetInstance()->getModel().c_str());
            return false;
        }

        // schedule config setting
        schedule_config_.type = MNN_FORWARD_CPU;
        schedule_config_.numThread = 4;
        MNN::BackendConfig backend_config;
        backend_config.precision = MNN::BackendConfig::Precision_Normal;
        schedule_config_.backendConfig = &backend_config;

        // create session
        session_ = interpreter_->createSession(schedule_config_);
        // get input tensor
        input_tensor_ = interpreter_->getSessionInput(session_, nullptr);
        // input shape setting
        input_shape_ = input_tensor_->shape();
        // the model has not input dimension
        if (input_shape_.size() == 0)
        {
            input_shape_.resize(4);
            // todo: set shape to input size
        }
        else
        {
            fprintf(stdout, "shape = [%d, %d, %d, %d]\n", input_shape_[0], input_shape_[1], input_shape_[2], input_shape_[3]);
        }
        // set batch to be 1
        input_shape_[0] = 1;

        interpreter_->resizeTensor(input_tensor_, input_shape_);
        interpreter_->resizeSession(session_);

        output_tensor_ = interpreter_->getSessionOutput(session_, nullptr);
        output_tensor_size = output_tensor_->elementSize();

        return true;
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
        // create image pretreat
        if (!pretreat_)
        {
            MNN::CV::ImageProcess::Config config;
            config.filterType = MNN::CV::BILINEAR;
            config.sourceFormat = convertXNNPixFormat2MNN(XNNConfig::GetInstance()->getSrcFormat());;
            config.destFormat = convertXNNPixFormat2MNN(XNNConfig::GetInstance()->getDstFormat());
            std::vector<float> means = XNNConfig::GetInstance()->getMeans();
            for (int i = 0; i < means.size(); i++) {
                config.mean[i] = means[i];
            }
            std::vector<float> normal = XNNConfig::GetInstance()->getNormal();
            for (int i = 0; i < normal.size(); i++) {
                config.normal[i] = normal[i];
            }
            fprintf(stdout, "src_format=%d, dst_format=%d\n", config.sourceFormat, config.destFormat);
            pretreat_ = MNN::CV::ImageProcess::create(config);
        }
        // resize image
        MNN::CV::Matrix trans;
        trans.setScale((float)(image->width)/(input_shape_[3]), (float)(image->height)/(input_shape_[2]));
        pretreat_->setMatrix(trans);
        // convert image
        pretreat_->convert(image->data, image->width, image->height, 0, input_tensor_);
        interpreter_->runSession(session_);

        std::vector<std::pair<int, float>> sorted_result(output_tensor_size);
        {
            // default float value
            auto output_data_ptr = output_tensor_->host<float>();
            // softmax
            if (!XNNConfig::GetInstance()->hasSoftmax())
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
            else {
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
        interpreter_->releaseModel();
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

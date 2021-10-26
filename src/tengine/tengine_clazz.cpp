#include "tengine/tengine_clazz.hpp"

#include <iostream>
#include <memory>

namespace xnn
{
    bool TengineClazz::init(int num_class,
                  std::vector<float> &means,
                  std::vector<float> &scales,
                  std::string model_file,
                  int input_size,
                  int tengine_mode,
                  bool has_softmax)
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
        scales_ = scales;
        // input channel
        int input_channel = means_.size();
        // setting has_softmax
        has_softmax_ = has_softmax;
        // set runtime options
        struct options opt;
        opt.num_thread = 1;
        opt.cluster = TENGINE_CLUSTER_ALL;
        opt.precision = tengine_mode;
        int img_size = input_size * input_size * input_channel;
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
        input_tensor_ = get_graph_input_tensor(graph_, 0, 0);
        if (input_tensor_ == NULL)
        {
            fprintf(stderr, "Get input tensor failed\n");
            return false;
        }

        // nchw
        int dims[] = {1, input_channel, input_size, input_size};
        if (set_tensor_shape(input_tensor_, dims, 4) < 0)
        {
            fprintf(stderr, "Set input tensor shape failed\n");
            return -1;
        }
        // input setting
        tengine_mode_ = tengine_mode;
        if (tengine_mode == TENGINE_MODE_FP32) {
            input_float32_data_ = (float*)malloc(img_size * sizeof(float));
            if (input_float32_data_ == NULL) {
                return -1;
            }
            if (set_tensor_buffer(input_tensor_, input_float32_data_, img_size * sizeof(float)) < 0) {
                fprintf(stderr, "Set input tensor buffer failed, image size = %d\n", img_size);
                return -1;
            }
            opt.affinity = 0;
        } else if (tengine_mode == TENGINE_MODE_INT8) {
            input_int8_data_ = (int8_t*)malloc(img_size);
            if (input_int8_data_ == NULL) {
                return -1;
            }
            if (set_tensor_buffer(input_tensor_, input_int8_data_, img_size) < 0)
            {
                fprintf(stderr, "Set input tensor buffer failed\n");
                return -1;
            }
            opt.affinity = 255;
        }
        // create softmax layer
        if (!has_softmax_)
        {
            // todo
        }

        // prerun graph, set work options(num_thread, cluster, precision)
        if (prerun_graph_multithread(graph_, opt) < 0)
        {
            fprintf(stderr, "Prerun multithread graph failed.\n");
            return -1;
        }

        return true;
    }

    XNNStatus TengineClazz::run(XNNImage *image, std::vector<std::pair<int, float>> &result, int topk)
    {
        if (!image || !image->data || image->width <= 0 || image->height <= 0)
        {
            fprintf(stderr, "Input parameter error\n");
            return XNN_PARAM_ERROR;
        }
        // clear the old result
        result.clear();
        // prepare process input data, set the data mem to input tensor
        getInputData(image, input_float32_data_, means_, scales_);
        // run graph
        if (run_graph(graph_, 1) < 0) {
            fprintf(stderr, "Run graph failed\n");
            return XNN_RUN_ERROR;
        }
        /* get the result of classification */
        tensor_t output_tensor = get_graph_output_tensor(graph_, 0, 0);
        float* output_data = (float*)get_tensor_buffer(output_tensor);
        int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);

        // manually call softmax on the fc output
        if (!has_softmax_) {
            // out
        }
        // get sorted result
        std::vector<std::pair<int, float>> sorted_result(output_size);
        for (int i = 0; i < output_size; i++)
        {
            sorted_result[i] = std::make_pair(i, output_data[i]);
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

    void TengineClazz::release()
    {
        // release tengine
        free(input_float32_data_);
        postrun_graph(graph_);
        destroy_graph(graph_);
        release_tengine();
    }

    // --------------------------------------------
    Image TengineClazz::makeImage(int w, int h, int c) {
        Image out = makeEmptyImage(w, h, c);
        out.data = (float*)calloc((size_t)h * w * c, sizeof(float));
        return out;
    }

    Image TengineClazz::makeEmptyImage(int w, int h, int c) {
        Image out;
        out.data = 0;
        out.h = h;
        out.w = w;
        out.c = c;
        return out;
    }

    Image TengineClazz::loadImage(XNNImage *image) {
        // default image channel is 1
        int c = 1;
        int src_c = c;
        int w = image->width;
        int h = image->height;
        unsigned char* data = image->data;
        Image im = makeImage(w, h, c);
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    int dst_index = i + w * j + w * h * k;
                    int src_index = k + src_c * i + src_c * w * j;
                    im.data[dst_index] = (float)data[src_index];
                }
            }
        }

        return im;
    }

    void TengineClazz::tengineResizeF32(float* data, float* res, int ow, int oh, int c, int h, int w) {
        float _scale_x = (float)(w) / (float)(ow);
        float _scale_y = (float)(h) / (float)(oh);
        float offset = 0.5f;

        int16_t* buf = (int16_t*)malloc((ow + ow + ow + oh + oh + oh) * sizeof(int16_t));
        int16_t* xCoef = (int16_t*)(buf);
        int16_t* xPos = (int16_t*)(buf + ow + ow);
        int16_t* yCoef = (int16_t*)(buf + ow + ow + ow);
        int16_t* yPos = (int16_t*)(buf + ow + ow + ow + oh + oh);

        for (int i = 0; i < ow; i++)
        {
            float fx = (float)(((float)i + offset) * _scale_x - offset);
            int sx = (int)fx;
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0.f;
            }
            if (sx >= w - 1)
            {
                sx = w - 2;
                fx = 0.f;
            }
            xCoef[i] = fx * 2048;
            xCoef[i + ow] = (1.f - fx) * 2048;
            xPos[i] = sx;
        }

        for (int j = 0; j < oh; j++)
        {
            float fy = (float)(((float)j + offset) * _scale_y - offset);
            int sy = (int)fy;
            fy -= sy;
            if (sy < 0)
            {
                sy = 0;
                fy = 0.f;
            }
            if (sy >= h - 1)
            {
                sy = h - 2;
                fy = 0.f;
            }
            yCoef[j] = fy * 2048;
            yCoef[j + oh] = (1.f - fy) * 2048;
            yPos[j] = sy;
        }

        //    int32_t* row = new int32_t[ow + ow];
        int32_t* row = (int32_t*)malloc((ow + ow) * sizeof(int32_t));

        for (int k = 0; k < c; k++)
        {
            int32_t channel = k * w * h;
            for (int j = 0; j < oh; j++)
            {
    #ifdef __ARM_NEON
                int32x4_t fy_0 = vdupq_n_s32(yCoef[j + oh]);
                int32x4_t _fy = vdupq_n_s32(yCoef[j]);
    #endif
                int32_t* p0_u = row;
                int32_t* p0_d = row + ow;
                int32_t yPosValue = yPos[j] * w + channel;
                for (int i = 0; i < ow; i++)
                {
                    int32_t data0 = (int32_t) * (data + yPosValue + xPos[i]) * xCoef[i + ow] >> 11;
                    int32_t data1 = (int32_t) * (data + yPosValue + xPos[i] + 1) * xCoef[i] >> 11;
                    int32_t data2 = (int32_t) * (data + yPosValue + w + xPos[i]) * xCoef[i + ow] >> 11;
                    int32_t data3 = (int32_t) * (data + yPosValue + w + xPos[i] + 1) * xCoef[i] >> 11;
                    p0_u[i] = ((data0) + (data1));
                    p0_d[i] = ((data2) + (data3));
                }
    #ifdef __ARM_NEON
                for (int i = 0; i < (ow & -4); i += 4)
                {
                    int32x4_t c1DataR = vmulq_s32(vld1q_s32(p0_u + i), fy_0);
                    int32x4_t c1DataL = vmulq_s32(vld1q_s32(p0_d + i), _fy);
                    int32x4_t c1Data_int = vshrq_n_s32(vaddq_s32(c1DataR, c1DataL), 11);
                    float32x4_t c1Data_float = vcvtq_f32_s32(c1Data_int);
                    vst1q_f32(res, c1Data_float);

                    res += 4;
                }

                for (int i = ow & ~3; i < ow; i++)
                {
                    int32_t data0 = *(p0_u + i) * yCoef[j + oh];
                    int32_t data1 = *(p0_d + i) * yCoef[j];
                    *res = (data0 + data1) >> 11;
                    res++;
                }
    #else
                for (int i = 0; i < ow; i++)
                {
                    int32_t data0 = *(p0_u + i) * yCoef[j + oh];
                    int32_t data1 = *(p0_d + i) * yCoef[j];
                    *res = (data0 + data1) >> 11;
                    res++;
                }
    #endif
            }
        }

        free(row);
        free(buf);
    }


    Image TengineClazz::imread2caffe(Image resImg, int img_w, int img_h, float* means, float* scale) {
        for (int c = 0; c < resImg.c; c++) {
            for (int i = 0; i < resImg.h; i++) {
                for (int j = 0; j < resImg.w; j++) {
                    int index = c * resImg.h * resImg.w + i * resImg.w + j;
                    resImg.data[index] = (resImg.data[index] - means[c]) * scale[c];
                }
            }
        }
        return resImg;
    }

    void TengineClazz::freeImage(Image m) {
        if (m.data) {
            free(m.data);
        }
    }

    void TengineClazz::getInputData(XNNImage *image, float* input_data, std::vector<float> &means,
                  std::vector<float> &scales) {
        // The default input channel of the model is 1 !!!
        float mean[1] = {means[0]};
        float scale[1] = {scales[0]};

        Image out = loadImage(image);
        //image res_img = makeImage(out.w, out.h, out.c);
        //tengineResizeF32(out.data, res_img.data, out.w, out., out.c, out.h, out.w);
        out = imread2caffe(out, out.w, out.h, mean, scale);

        memcpy(input_data, out.data, sizeof(float) * out.c * out.w * out.h);
        freeImage(out);
    }

} // namespace xnn

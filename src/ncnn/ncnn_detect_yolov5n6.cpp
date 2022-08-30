#include "ncnn/ncnn_detect.hpp"

#include <float.h>

#include <iostream>
#include <memory>

#define MAX_STRIDE 64

namespace xnn
{

    static inline float intersection_area(const DetectObject& a, const DetectObject& b)
    {
        Rect_<float> inter = a.rect & b.rect;
        return inter.area();
    }

    static void qsort_descent_inplace(std::vector<DetectObject>& faceobjects, int left, int right)
    {
        int i = left;
        int j = right;
        float p = faceobjects[(left + right) / 2].prob;

        while (i <= j)
        {
            while (faceobjects[i].prob > p)
                i++;

            while (faceobjects[j].prob < p)
                j--;

            if (i <= j)
            {
                // swap
                std::swap(faceobjects[i], faceobjects[j]);

                i++;
                j--;
            }
        }

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                if (left < j) qsort_descent_inplace(faceobjects, left, j);
            }
            #pragma omp section
            {
                if (i < right) qsort_descent_inplace(faceobjects, i, right);
            }
        }
    }

    static void qsort_descent_inplace(std::vector<DetectObject>& faceobjects)
    {
        if (faceobjects.empty())
            return;

        qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
    }

    static void nms_sorted_bboxes(const std::vector<DetectObject>& faceobjects, std::vector<int>& picked, float nms_threshold)
    {
        picked.clear();

        const int n = faceobjects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++)
        {
            areas[i] = faceobjects[i].rect.area();
        }

        for (int i = 0; i < n; i++)
        {
            const DetectObject& a = faceobjects[i];

            int keep = 1;
            for (int j = 0; j < (int)picked.size(); j++)
            {
                const DetectObject& b = faceobjects[picked[j]];

                // intersection over union
                float inter_area = intersection_area(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold)
                    keep = 0;
            }

            if (keep)
                picked.push_back(i);
        }
    }

    static inline float sigmoid(float x)
    {
        return static_cast<float>(1.f / (1.f + exp(-x)));
    }

    static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<DetectObject>& objects)
    {
        const int num_grid = feat_blob.h;

        int num_grid_x;
        int num_grid_y;
        if (in_pad.w > in_pad.h)
        {
            num_grid_x = in_pad.w / stride;
            num_grid_y = num_grid / num_grid_x;
        }
        else
        {
            num_grid_y = in_pad.h / stride;
            num_grid_x = num_grid / num_grid_y;
        }

        const int num_class = feat_blob.w - 5;

        const int num_anchors = anchors.w / 2;

        for (int q = 0; q < num_anchors; q++)
        {
            const float anchor_w = anchors[q * 2];
            const float anchor_h = anchors[q * 2 + 1];

            const ncnn::Mat feat = feat_blob.channel(q);

            for (int i = 0; i < num_grid_y; i++)
            {
                for (int j = 0; j < num_grid_x; j++)
                {
                    const float* featptr = feat.row(i * num_grid_x + j);

                    // find class index with max class score
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k];
                        if (score > class_score)
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }

                    float box_score = featptr[4];

                    float confidence = sigmoid(box_score) * sigmoid(class_score);

                    if (confidence >= prob_threshold)
                    {
                        float dx = sigmoid(featptr[0]);
                        float dy = sigmoid(featptr[1]);
                        float dw = sigmoid(featptr[2]);
                        float dh = sigmoid(featptr[3]);

                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f;
                        float y0 = pb_cy - pb_h * 0.5f;
                        float x1 = pb_cx + pb_w * 0.5f;
                        float y1 = pb_cy + pb_h * 0.5f;

                        DetectObject obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0;
                        obj.rect.height = y1 - y0;
                        obj.label = class_index;
                        obj.prob = confidence;

                        objects.push_back(obj);
                    }
                }
            }
        }
    }


    bool NCNNDetect::init(int num_class,
                         std::string param_path,
                         std::string bin_path,
                         int input_size,
                         bool load_param_bin)
    {
        if (num_class <= 0)
        {
            fprintf(stderr, "Parameter error: invalid number(%d) of class\n", num_class);
            return false;
        }
        // get class number
        num_class_ = num_class;
        input_size_ = input_size;
        // is param file binary file?
        load_param_bin_ = load_param_bin;
        // create interpreter from param/bin file
        if (param_path.size() != 0 && bin_path.size() != 0)
        {
            net_.opt.use_vulkan_compute = false;
            if (load_param_bin)
            {
                net_.load_param_bin(param_path.c_str());
            }
            else
            {
                net_.load_param(param_path.c_str());
            }
            net_.load_model(bin_path.c_str());
        }
        else
        {
            fprintf(stderr, "Parameter error: param or bin file can not be empty.\n");
            return false;
        }

        // get input name and output name of net
        input_layer_name_ = "images";

        return true;
    }

void saveImgRawData(const char *filename, unsigned char *raw, int w, int h, int c) {
    char data[128] = "";
    sprintf(data, "%s.raw_%d_%d_%d", filename, w, h, c);
    FILE *fp = fopen(data, "wb");
    fwrite(raw, 1, w * h * c, fp);
    fclose(fp);
}

    XNNStatus NCNNDetect::run(XNNImage *image, std::vector<DetectObject>& objects, int topk)
    {
        if (!image || !image->data || image->width <= 0 || image->height <= 0)
        {
            fprintf(stderr, "Input parameter error\n");
            return XNN_PARAM_ERROR;
        }
        // clear the old result
        objects.clear();

        // letterbox pad to multiple of MAX_STRIDE
        int w = image->width;
        int h = image->height;
        fprintf(stdout, "imgw=%d, imgh=%d, input_size=%d\n", w, h, input_size_);
        float scale = 1.f;
        if (w > h)
        {
            scale = (float)input_size_ / w;
            w = input_size_;
            h = h * scale;
        }
        else
        {
            scale = (float)input_size_ / h;
            h = input_size_;
            w = w * scale;
        }
        // w = input_size_;
        // h = input_size_;

        ncnn::Mat in;
        if (image->width == input_size_ && image->height == input_size_) {
            in = ncnn::Mat::from_pixels(image->data, convertXNNPixFormat2NCNN(image->src_pixel_format), input_size_, input_size_);
        } else {
            fprintf(stdout, "src pix format=%d\n", image->src_pixel_format);
            fprintf(stdout, "w=%d, h=%d, input_size=%d, imgw = %d, imgh=%d\n", w, h, input_size_, image->width, image->height);
            //saveImgRawData("detect_input.raw", image->data, image->width, image->height, 3);
            in = ncnn::Mat::from_pixels_resize(image->data, convertXNNPixFormat2NCNN(image->src_pixel_format), image->width, image->height, w, h);
        }
        fprintf(stdout, "xxxxx\n");

        // pad to target_size rectangle
        int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
        int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor extractor = net_.create_extractor();
        // in extract
        if (load_param_bin_)
        {
            extractor.input(atoi(input_layer_name_.c_str()), in_pad);
        }
        else
        {
            extractor.input(input_layer_name_.c_str(), in_pad);
        }
        // out extract
        std::vector<DetectObject> proposals;
        // stride 8
        {
            ncnn::Mat out;
            extractor.extract("output", out);
            ncnn::Mat anchors(6);
            anchors[0] = 19.f;
            anchors[1] = 27.f;
            anchors[2] = 44.f;
            anchors[3] = 40.f;
            anchors[4] = 38.f;
            anchors[5] = 94.f;

            std::vector<DetectObject> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold_, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;
            extractor.extract("468", out);
            ncnn::Mat anchors(6);
            anchors[0] = 96.f;
            anchors[1] = 68.f;
            anchors[2] = 86.f;
            anchors[3] = 152.f;
            anchors[4] = 180.f;
            anchors[5] = 137.f;

            std::vector<DetectObject> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold_, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            extractor.extract("508", out);
            ncnn::Mat anchors(6);
            anchors[0] = 140.f;
            anchors[1] = 301.f;
            anchors[2] = 303.f;
            anchors[3] = 264.f;
            anchors[4] = 238.f;
            anchors[5] = 542.f;

            std::vector<DetectObject> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold_, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

         // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold_);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)\

        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
            float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(image->width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(image->height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(image->width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(image->height - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }

        return XNN_SUCCESS;
    }

    void NCNNDetect::release()
    {

    }

} // namespace xnn

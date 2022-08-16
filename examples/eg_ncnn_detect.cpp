#include "common/common.hpp"
#include "ncnn/ncnn_detect.hpp"

#include <iostream>
#include <chrono>

#define TOPK 5
#define LOOP 5


void readRawData(const char *filename, unsigned char *data) {
    FILE *fp = NULL;
    // open file
    fp = fopen(filename,"rb");
    if(fp == NULL)
    {
        printf("failed to read file\n");
        exit(-1);
    }
    // get file size
    fseek (fp , 0 , SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    // read data
    fread(data, sizeof(char), file_size, fp);
    fclose(fp);
    fp = NULL;
}


int main(int argc, char *argv[])
{
    if (argc < 5) {
        fprintf(stderr, "Usage: %s [image raw data file] [image_width] [image_height] [target_size]\n", argv[0]);
        return -1;
    }

    // get parameters from config.json
    int num_class = 1;
    std::string bin_path = "output-detect-nobn.bin";
    std::string param_path = "output-detect-nobn.param";
    bool is_load_param_bin = false;
    int target_size = atoi(argv[4]);
    // init
    xnn::NCNNDetect ncnn_detect;
    if (!ncnn_detect.init(num_class, param_path, bin_path, target_size, is_load_param_bin)) {
        fprintf(stderr, "Failed to init.\n");
        return -1;
    }

    // run
    int img_channel = 3; // XNN_PIX_GRAY;
    XNNImage image;
    image.width = atoi(argv[2]);
    image.height = atoi(argv[3]);
    image.data = new unsigned char[image.width * image.height * img_channel];
    image.src_pixel_format = XNN_PIX_BGR2RGB; // XNN_PIX_RGB;
    image.dst_pixel_format = XNN_PIX_BGR2RGB;

    readRawData(argv[1], image.data);

    long long average_time = 0;
    std::vector<DetectObject> result;
    for (int i = 0; i < LOOP; i++) {
        // start timing
        auto start = std::chrono::system_clock::now();
        // run classfication
        ncnn_detect.run(&image, result, TOPK);

        // print result
        for (int i = 0; i < result.size(); ++i) {
            fprintf(stdout, "%d = %.5f at %.2f %.2f (w=%.2f x h=%.2f)\n",
                result[i].label, result[i].prob, result[i].rect.x, result[i].rect.y,
                result[i].rect.width, result[i].rect.height);
            //fprintf(stdout, "result[%d] =  [label=%d, socre=%f]\n", i, result[i].label, result[i].prob);
        }
        fprintf(stdout, "-------------result size=%lu-------------\n", result.size());
        // abort timer
        auto end = std::chrono::system_clock::now();
        auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        fprintf(stdout, "%ld ms at a time.\n", int_ms.count());
        average_time += int_ms.count();
    }

    fprintf(stdout, "The average time of %d times is %lld ms\n", LOOP, average_time / LOOP);

    // release
    delete []image.data;
    image.data = nullptr;
    ncnn_detect.release();

    return 0;
}

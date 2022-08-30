#include "common/common.hpp"
#include "ncnn/yolo_fastv2.hpp"

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

    bool is_load_param_bin = false;
    int target_size = atoi(argv[4]);
    // init
    xnn::yoloFastestv2 api;

    api.loadModel("./output-detect-fast-opt.param", "./output-detect-fast-opt.bin", target_size);

    std::vector<xnn::TargetBox> boxes;

    // run
    int img_channel = 3; // XNN_PIX_GRAY;
    XNNImage image;
    image.width = atoi(argv[2]);
    image.height = atoi(argv[3]);
    image.data = new unsigned char[image.width * image.height * img_channel];
    image.src_pixel_format = XNN_PIX_BGR2GRAY; // XNN_PIX_RGB;
    image.dst_pixel_format = XNN_PIX_BGR2GRAY;

    readRawData(argv[1], image.data);

    long long average_time = 0;

    for (int i = 0; i < LOOP; i++) {
        // start timing
        auto start = std::chrono::system_clock::now();
        // run classfication
        api.detection(&image, boxes);

        // print result
        for (int i = 0; i < boxes.size(); ++i) {
            fprintf(stdout, "(x1=%d, y1=%d)- (x2=%d, y2=%d), score=%.2f, category=%d\n",
                boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2,
                boxes[i].score * 100, boxes[i].cate);
        }
        fprintf(stdout, "-------------result size=%lu-------------\n", boxes.size());
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

    return 0;
}

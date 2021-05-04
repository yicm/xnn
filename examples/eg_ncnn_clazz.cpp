#include "common/common.hpp"
#include "ncnn/ncnn_clazz.hpp"

#include <chrono>

#define TOPK 5
#define LOOP 10

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
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [image raw data file]\n", argv[0]);
        return -1;
    }
    // get parameters from config.json
    int num_class = 45;
    std::vector<float> means ={116.28f};
    std::vector<float> normals = {0.017507f};
    std::string bin_path = "model_199-int8.bin";
    std::string param_path = "model_199-int8.param.bin";
    bool has_softmax = false;
    bool is_load_param_bin = true;
    int input_size = 64;
     // init
    xnn::NCNNClazz ncnn_clazz;
    if (!ncnn_clazz.init(num_class, means, normals, param_path, bin_path, is_load_param_bin, has_softmax)) {
        fprintf(stderr, "Failed to init.\n");
        return -1;
    }

    // run, 1 is gray image
    int img_channel = 1;
    XNNImage image;
    image.data = new unsigned char[64 * 64 * img_channel];
    image.width = 64;
    image.height = 64;
    image.src_pixel_format = XNN_PIX_GRAY;
    image.dst_pixel_format = XNN_PIX_GRAY;
    readRawData(argv[1], image.data);

    long long average_time = 0;
    std::vector<std::pair<int, float>> result;
    for (int i = 0; i < LOOP; i++) {
        // start timing
        auto start = std::chrono::system_clock::now();
        // run classfication
        ncnn_clazz.run(&image, result, TOPK);

        // print result
        for (int i = 0; i < TOPK; ++i) {
            fprintf(stdout, "result[%d] =  [class=%d, socre=%f]\n", i, result[i].first, result[i].second);
        }
        fprintf(stdout, "--------------------------\n");
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
    ncnn_clazz.release();

    return 0;
}

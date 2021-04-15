#include "config/config.hpp"
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
    // init config
    XNNConfig::GetInstance()->parseConfig();
    // get parameters from config.json
    int num_class = XNNConfig::GetInstance()->getNumClass();
    std::vector<float> means = XNNConfig::GetInstance()->getMeans();
    std::vector<float> normals = XNNConfig::GetInstance()->getNormal();
    std::string bin_path = XNNConfig::GetInstance()->getBin();
    std::string param_path = XNNConfig::GetInstance()->getParam();
    bool has_softmax = XNNConfig::GetInstance()->hasSoftmax();
    bool is_load_param_bin = XNNConfig::GetInstance()->isLoadParamBin();
    int input_size = XNNConfig::GetInstance()->getInputSize();
    // init
    xnn::NCNNClazz ncnn_clazz;
    if (!ncnn_clazz.init(num_class, means, normals, param_path, bin_path, is_load_param_bin, has_softmax)) {
        fprintf(stderr, "Failed to init.\n");
        return -1;
    }
   
    // run
    int img_channel = XNNConfig::GetInstance()->getSrcFormat() == XNN_PIX_GRAY ? 1 : 3;
    XNNImage image;
    image.data = new unsigned char[input_size * input_size * img_channel];
    image.width = input_size;
    image.height = input_size;
    image.src_pixel_format = XNNConfig::GetInstance()->getSrcFormat();
    image.dst_pixel_format = XNNConfig::GetInstance()->getDstFormat();
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

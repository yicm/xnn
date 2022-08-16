#include <iostream>
#include <chrono>
#include <vector>

#include <dirent.h>

#include "config/config.hpp"
#include "ncnn/ncnn_clazz.hpp"

//#define USE_STB
#ifdef USE_STB
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#define TOPK 5

void readRawData(const char *filename, unsigned char *data)
{
    FILE *fp = NULL;
    // open file
    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        printf("failed to read file\n");
        exit(-1);
    }
    // get file size
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    rewind(fp);
    // read data
    size_t size = fread(data, sizeof(char), file_size, fp);
    fclose(fp);
    fp = NULL;
}

static int getFiles(std::string path, std::vector<std::string> &filenames)
{
    DIR *pDir;
    struct dirent *ptr;
    if (!(pDir = opendir(path.c_str())))
    {
        return -1;
    }
    while ((ptr = readdir(pDir)) != 0)
    {
        // filter out the '.' and the '..' directory
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0)
        {
            filenames.push_back(path + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    return 0;
}

static std::vector<std::string> split(const std::string &str, const std::string &delim)
{
    std::vector<std::string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == std::string::npos)
            pos = str.length();
        std::string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(token);
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}

void saveImgRawData(int clazz, unsigned char *raw, int w, int h, int c) {
    char data[128] = "";
    sprintf(data, "%d.raw_%d_%d_%d", clazz, w, h, c);
    FILE *fp = fopen(data, "wb");
    fwrite(raw, 1, w * h * c, fp);
    fclose(fp);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s [Test dataset root directory]\n", argv[0]);
        return -1;
    }
    // test data set root directory
    const char *image_path = argv[1];
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
    if (!ncnn_clazz.init(num_class, means, normals, param_path, bin_path, input_size, is_load_param_bin, has_softmax)) {
        fprintf(stderr, "Failed to init.\n");
        return -1;
    }
    XNNImage image;
    image.src_pixel_format = XNNConfig::GetInstance()->getSrcFormat();
    image.dst_pixel_format = XNNConfig::GetInstance()->getDstFormat();
    std::vector<std::pair<int, float>> result;
    // statistics
    long total_num = 0;
    long correct_num = 0;
    long error_num = 0;
    long long average_time = 0;
    // get val dataset
    std::vector<std::string> dataset_clazz_dirs;
    getFiles(image_path, dataset_clazz_dirs);
    for (int i = 0; i < dataset_clazz_dirs.size(); i++)
    {
        // get object name of ground truth
        std::cout << dataset_clazz_dirs[i] << std::endl;
        std::vector<std::string> dataset_path_split = split(dataset_clazz_dirs[i], "/");
        int clazz = atoi(dataset_path_split[dataset_path_split.size() - 1].c_str());
        // get all file in a sub directory
        std::vector<std::string> clazz_all_files;
        getFiles(dataset_clazz_dirs[i], clazz_all_files);
        std::cout << "all file num " << clazz_all_files.size() << std::endl;
        for (int j = 0; j < clazz_all_files.size(); j++)
        {
            // start timing
            auto start = std::chrono::system_clock::now();
            // read image
            int desired_channels = 1;
            int h, w, channel;
            #ifdef USE_STB
            auto input_image = stbi_load(clazz_all_files[j].c_str(), &w, &h, &channel, desired_channels);
            if (!input_image)
            {
                fprintf(stderr, "failed to read image: %s\n", clazz_all_files[j].c_str());
                break;
            }
            #else
            cv::Mat input_image = cv::imread(clazz_all_files[j], image.src_pixel_format == XNN_PIX_GRAY ? 0 : 1);
            if (input_image.empty()) {
                fprintf(stderr, "failed to read image: %s\n", clazz_all_files[j].c_str());
                break;
            }
            h = input_image.rows;
            w = input_image.cols;
            channel = input_image.channels();
            #endif
            std::cout << "===" << clazz_all_files[j] << ", w=" << w << ", h=" << h << ", c=" << channel << std::endl;
            #ifndef USE_STB
            image.data = (uint8_t *)input_image.data;
            #else
            image.data = (uint8_t *)input_image;
            #endif
            image.width = w;
            image.height = h;
            // run classfication
            //saveImgRawData(clazz, image.data, w, h ,channel);
            ncnn_clazz.run(&image, result, TOPK);
            #ifdef USE_STB
            stbi_image_free(input_image);
            #endif

            // print and count result
            const int TOP1_INDEX = 0;
            fprintf(stdout, "result[%d] =  [class=%d, socre=%f]\n", TOP1_INDEX, result[TOP1_INDEX].first, result[TOP1_INDEX].second);
            if (result[TOP1_INDEX].first == clazz)
            {
                correct_num++;
            }
            else
            {
                error_num++;
            }
            total_num++;

            // abort timer
            auto end = std::chrono::system_clock::now();
            auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            fprintf(stdout, "%ld ms at a time.\n", int_ms.count());
            average_time += int_ms.count();
        }
    }

    fprintf(stdout, "The total number is %ld\n", total_num);
    fprintf(stdout, "The average time is %lld ms\n", average_time / total_num);
    fprintf(stdout, "The Accuracy is %f, error rate is %f\n",
            (float)correct_num / (float)total_num, (float)error_num / (float)total_num);

    // release
    ncnn_clazz.release();

    return 0;
}

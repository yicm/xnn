#include "config/config.hpp"

int main()
{
    XNNConfig::GetInstance()->parseConfig();

    fprintf(stdout, "src format: %d\n", XNNConfig::GetInstance()->getSrcFormat());
    fprintf(stdout, "dst format: %d\n", XNNConfig::GetInstance()->getDstFormat());

    std::vector<float> means = XNNConfig::GetInstance()->getMeans();
    fprintf(stdout, "means[");
    for (int i = 0; i < means.size(); i++) {
        fprintf(stdout, "%f, ", means[i]);
    }
    fprintf(stdout, "]\n");

    std::vector<float> normal = XNNConfig::GetInstance()->getNormal();
    fprintf(stdout, "normal[");
    for (int i = 0; i < normal.size(); i++) {
        fprintf(stdout, "%f, ", normal[i]);
    }
    fprintf(stdout, "]\n");

    fprintf(stdout, "has softmax: %d\n", XNNConfig::GetInstance()->hasSoftmax() ? 1 : 0);

    return 0;
}

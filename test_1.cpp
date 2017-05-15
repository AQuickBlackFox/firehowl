#include"include/firehowl/firehowl.h"

using namespace firehowl;

#define W1_HEIGHT 16
#define W1_WIDTH 256
#define X1_HEIGHT 256
#define X1_WIDTH 256
#define Y1_HEIGHT W1_HEIGHT
#define Y1_WIDTH X1_WIDTH

int main(){
    Tensor<float, W1_HEIGHT, W1_WIDTH> W1;
    Tensor<float, X1_HEIGHT, X1_WIDTH> X1;
    Tensor<float, Y1_HEIGHT, Y1_WIDTH> Y1;

    W1[0] = 1.0f;
    X1[0] = 1.0f;
    Y1[0] = 1.0f;

    std::cout<<W1[0]<<" "<<X1[0]<<" "<<Y1[0]<<std::endl;
    hipStream_t stream;
    hipStreamCreate(&stream);
    W1.SyncDevice(stream);
    hipDeviceSynchronize();
    W1[0] = 2.0f;
    W1.SyncHost(stream);
    hipDeviceSynchronize();
    std::cout<<W1[0]<<std::endl;
    MatMul<float, W1_HEIGHT, W1_WIDTH, X1_HEIGHT, X1_WIDTH>(stream, Y1, W1, X1);
    hipDeviceSynchronize();
}

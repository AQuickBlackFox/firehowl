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

    for(int j=0;j<W1_HEIGHT;j++) {
        for(int i=0;i<W1_WIDTH;i++) {
            W1(j, i) = 1.0f;
        }
    }

    for(int j=0;j<X1_HEIGHT;j++) {
        for(int i=0;i<X1_WIDTH;i++) {
            X1(j, i) = 1.0f;
        }
    }

    for(int j=0;j<Y1_HEIGHT;j++) {
        for(int i=0;i<Y1_WIDTH;i++) {
            Y1(j, i) = 0.0f;
        }
    }



    std::cout<<W1[0]<<" "<<X1[0]<<" "<<Y1[0]<<std::endl;
/*
    hipStream_t stream;
    hipStreamCreate(&stream);
    W1.SyncDevice(stream);
    X1.SyncDevice(stream);
    Y1.SyncDevice(stream);
*/
    MatMul<float, W1_HEIGHT, W1_WIDTH, X1_HEIGHT, X1_WIDTH>(Y1, W1, X1);
    hipDeviceSynchronize();
    std::cout<<Y1[0]<<std::endl;
}

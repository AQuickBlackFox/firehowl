#include"include/firehowl/firehowl.h"

using namespace firehowl;

#define W1_HEIGHT 16
#define W1_WIDTH 256
#define X1_HEIGHT 256
#define X1_WIDTH 256
#define Y1_HEIGHT W1_HEIGHT
#define Y1_WIDTH X1_WIDTH
#define X2_HEIGHT Y1_HEIGHT
#define X2_WIDTH  Y1_WIDTH
#define W2_HEIGHT 256
#define W2_WIDTH  X2_HEIGHT
#define Y2_HEIGHT W2_HEIGHT
#define Y2_WIDTH  X2_WIDTH

int main(){
    Tensor<float, W1_HEIGHT, W1_WIDTH> W1;
    Tensor<float, X1_HEIGHT, X1_WIDTH> X1;
    Tensor<float, Y1_HEIGHT, Y1_WIDTH> Y1;
    Tensor<float, X2_HEIGHT, X2_WIDTH> X2;
    Tensor<float, W2_HEIGHT, W2_WIDTH> W2;
    Tensor<float, Y2_HEIGHT, Y2_WIDTH> Y2;

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

    for(int j=0;j<W2_HEIGHT;j++) {
        for(int i=0;i<W2_WIDTH;i++) {
            W2(j, i) = 1.0f;
        }
    }

    for(int j=0;j<X2_HEIGHT;j++) {
        for(int i=0;i<X2_WIDTH;i++) {
            X2(j, i) = 1.0f;
        }
    }

    for(int j=0;j<Y2_HEIGHT;j++) {
        for(int i=0;i<Y2_WIDTH;i++) {
            Y2(j, i) = 0.0f;
        }
    }

    std::cout<<W1[0]<<" "<<X1[0]<<" "<<Y1[0]<<std::endl;
    MatMul<float, W1_HEIGHT, W1_WIDTH, X1_HEIGHT, X1_WIDTH>(Y1, W1, X1);
    Tanh<float, Y1_HEIGHT, Y1_WIDTH>(X2, Y1);
    MatMul<float, W2_HEIGHT, W2_WIDTH, X2_HEIGHT, X2_WIDTH>(Y2, W2, X2);
    hipDeviceSynchronize();
    std::cout<<X2[0]<<std::endl;
}

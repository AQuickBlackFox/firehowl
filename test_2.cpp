#include "include/firehowl/firehowl.h"

#define X1_WIDTH 256
#define X1_HEIGHT 256
#define W1_HEIGHT 16
#define W1_WIDTH X1_HEIGHT
#define Y1_HEIGHT W1_HEIGHT
#define Y1_WIDTH X1_WIDTH

template<typename T, int height, int width>
void PrintMatrix(Matrix<T, height, width> M) {
    std::cout << "Height x Width: " << height << " x " << width << std::endl;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            std::cout << M.ptr[i + j*width] << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    float *x1, *w1, *b1, *y1;
    x1 = new float[X1_HEIGHT*X1_WIDTH];
    w1 = new float[W1_HEIGHT*W1_WIDTH];
    y1 = new float[Y1_HEIGHT*Y1_WIDTH];
    b1 = new float[Y1_HEIGHT*Y1_WIDTH];

    for (int j = 0; j < X1_HEIGHT; j++) {
        for (int i = 0; i < X1_WIDTH; i++) {
            x1[i + j*X1_WIDTH] = 1.0f;
        }
    }
    for (int j = 0; j < W1_HEIGHT; j++) {
        for (int i = 0; i < W1_WIDTH; i++) {
            w1[i + j*W1_WIDTH] = 0.5f;
        }
    }
    for (int j = 0; j < Y1_HEIGHT; j++) {
        for (int i = 0; i < Y1_WIDTH; i++) {
            y1[i + j*Y1_WIDTH] = 0.0f;
        }
    }
    for (int j = 0; j < Y1_HEIGHT; j++) {
        for (int i = 0; i < Y1_WIDTH; i++) {
            y1[i + j*Y1_WIDTH] = 0.0f;
        }
    }

    for (int j = 0; j < Y1_HEIGHT; j++) {
        for (int i = 0; i < Y1_WIDTH; i++) {
            b1[i + j*Y1_WIDTH] = 1.0f;
        }
    }

    hipStream_t stream;
    hipStreamCreate(&stream);

    float *x1d, *w1d, *b1d, *y1d;
	hipMalloc(&x1d, sizeof(float)*X1_WIDTH*X1_HEIGHT);
    hipMalloc(&w1d, sizeof(float)*W1_WIDTH*W1_HEIGHT);
    hipMalloc(&y1d, sizeof(float)*Y1_WIDTH*Y1_HEIGHT);
    hipMalloc(&b1d, sizeof(float)*Y1_WIDTH*Y1_HEIGHT);

    hipMemcpy(x1d, x1, sizeof(float)*X1_WIDTH*X1_HEIGHT, hipMemcpyHostToDevice);
    hipMemcpy(w1d, w1, sizeof(float)*W1_WIDTH*W1_HEIGHT, hipMemcpyHostToDevice);
    hipMemcpy(y1d, y1, sizeof(float)*Y1_WIDTH*Y1_HEIGHT, hipMemcpyHostToDevice);
    hipMemcpy(b1d, b1, sizeof(float)*Y1_WIDTH*Y1_HEIGHT, hipMemcpyHostToDevice);

    Matrix<float, X1_HEIGHT, X1_WIDTH> X1(x1d);
    Matrix<float, W1_HEIGHT, W1_WIDTH> W1(w1d);
    Matrix<float, Y1_HEIGHT, Y1_WIDTH> Y1(y1d);
    Matrix<float, Y1_HEIGHT, Y1_WIDTH> B1(b1d);

    firehowl::MatMul<float, W1_HEIGHT, W1_WIDTH, X1_HEIGHT, X1_WIDTH>(stream, Y1, W1, X1, B1);
    firehowl::Tanh<float, W1_HEIGHT, X1_WIDTH>(stream, Y1, Y1);
    hipDeviceSynchronize();

    hipMemcpy(y1, y1d, Y1_HEIGHT*Y1_WIDTH * sizeof(float), hipMemcpyDeviceToHost);
    Matrix<float, Y1_HEIGHT, Y1_WIDTH> Y(y1);
    PrintMatrix(Y);
    std::cout << std::endl;
}

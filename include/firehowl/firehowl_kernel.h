#pragma once

#include "firehowl_rt.h"
#include "firehowl_math.h"

#undef hipLaunchKernel 
#define hipLaunchKernel hipLaunchKernelGGL

#define FH_TILE_X_16 16
#define FH_TILE_Y_16 16

#define FH_TILE_X_32 32
#define FH_TILE_Y_32 32

template<typename T, int height, int width>
struct Matrix {
    int h, w;
    T *ptr;
    __device__ __host__ Matrix(T *ptr) : ptr(ptr), h(height), w(width) {}
    __device__ __host__ Matrix() : ptr(nullptr), h(height), w(width) {}
    __device__ __host__ inline T& operator[](int idx) {
        return ptr[idx];
    }
    __device__ __host__ inline const T& operator[](int idx) const {
        return ptr[idx];
    }
    __device__ __host__ inline T& operator()(int y, int x) {
        return ptr[x + y * width];
    }
    __device__ __host__ inline const T& operator()(int y, int x) const {
        return ptr[x + y * width];
    }
    __device__ __host__ ~Matrix() {}
};

template<typename T, int height, int width>
struct Tensor {
private:
    Matrix<T, height, width> hMat;
    Matrix<T, height, width> dMat;
    int h, w;
    int devSync, hostSync;
public:
    Tensor() : h(height), w(width), devSync(0), hostSync(1) { 
        FH_HIP_CHECK(hipHostMalloc(&hMat.ptr, height*width*sizeof(T), 0));
        FH_HIP_CHECK(hipMalloc(&dMat.ptr, height*width*sizeof(T)));
    }
    void SyncHost(hipStream_t stream) {
        FH_HIP_CHECK(hipMemcpyAsync(hMat.ptr, dMat.ptr, height*width*sizeof(T), hipMemcpyDeviceToHost, stream));
    }
    void SyncDevice(hipStream_t stream) {
        FH_HIP_CHECK(hipMemcpyAsync(dMat.ptr, hMat.ptr, height*width*sizeof(T), hipMemcpyHostToDevice, stream));
    }
    void CheckSyncHost(hipStream_t stream) {
        CHECK_HOST(stream);
    }
    void CheckSyncDevice(hipStream_t stream) {
        CHECK_DEVICE(stream);
    }
    inline T& operator[](int idx) {
        CHECK_HSYNC(init.getStream());
        return hMat.ptr[idx];
    }
    inline const T& operator[](int idx) const {
        CHECK_HSYNC(init.getStream());
        return hMat.ptr[idx];
    }
    inline T& operator()(int y, int x) {
        CHECK_HSYNC(init.getStream());
        return hMat.ptr[x + y * width];
    }
    inline const T& operator()(int y, int x) const {
        CHECK_HSYNC(init.getStream());
        return hMat.ptr[x + y * width];
    }
    inline T* getDPtr() { return dMat.ptr; }
    inline T* getHPtr() { return hMat.ptr; }
    ~Tensor() {
        FH_HIP_CHECK(hipHostFree(hMat.ptr));
        FH_HIP_CHECK(hipFree(dMat.ptr));
    }
};

template<typename T, int w_height, int w_width, int x_height, int x_width>
__global__ void FireHowlDot(
    Matrix<T, w_height, x_width> Y,
    Matrix<T, w_height, w_width> W,
    Matrix<T, x_height, x_width> X)
{
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;

    int bx = hipBlockIdx_x;
    int by = hipBlockIdx_y;

    int row = ty + by * FH_TILE_X_16;
    int col = tx + bx * FH_TILE_X_16;

    __shared__ T sW[FH_TILE_Y_16][FH_TILE_X_16];
    __shared__ T sX[FH_TILE_Y_16][FH_TILE_X_16];
    T C = 0;

    for (int j = 0; j < w_width / FH_TILE_Y_16; j++) {
        sW[ty][tx] = W[row*w_width + (j*FH_TILE_X_16 + tx)];
        sX[ty][tx] = X[col + (j*FH_TILE_X_16+ty)*x_width];
        __syncthreads();
        for (int i = 0; i < FH_TILE_Y_16; i++) {
            C = C + sW[ty][i] * sX[i][tx];
        }
        __syncthreads();
        Y[row*x_width+col] = C;
    }
}

template<typename T, int w_height, int w_width, int x_height, int x_width>
__global__ void FireHowlDotBias(
    Matrix<T, w_height, x_width> Y,
    Matrix<T, w_height, w_width> W,
    Matrix<T, x_height, x_width> X,
    Matrix<T, w_height, x_width> B)
{
    int tx = hipThreadIdx_x;
    int ty = hipThreadIdx_y;

    int bx = hipBlockIdx_x;
    int by = hipBlockIdx_y;

    int row = ty + by * FH_TILE_X_16;
    int col = tx + bx * FH_TILE_X_16;

    __shared__ T sW[FH_TILE_Y_16][FH_TILE_X_16];
    __shared__ T sX[FH_TILE_Y_16][FH_TILE_X_16];
    T c = 0;

    for (int j = 0; j < w_width / FH_TILE_Y_16; j++) {
        sW[ty][tx] = W[row*w_width + (j*FH_TILE_X_16 + tx)];
        sX[ty][tx] = X[col + (j*FH_TILE_X_16+ty)*x_width];
        __syncthreads();
        for (int i = 0; i < FH_TILE_Y_16; i++) {
            c = c + sW[ty][i] * sX[i][tx];
        }
        T b = B[row*x_width+col];
        __syncthreads();
        Y[row*x_width+col] = c + b;
    }
}

template<typename T, int height, int width>
__global__ void FireHowlTanh(
    Matrix<T, height, width> Y,
    Matrix<T, height, width> X)
{
    int tx = hipThreadIdx_x;
    int bx = hipBlockIdx_x;
    int id = tx + bx * FH_TILE_X_16 * FH_TILE_Y_16;
    T x = __fh_exp(2 * X[id]);
    x = (x - 1)/(x + 1);
    Y[id] = x;
}

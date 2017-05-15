#ifndef FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_H
#define FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_H

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "firehowl_kernel.h"

namespace firehowl {

template<typename T, int w_height, int w_width, int x_height, int x_width>
void MatMul(hipStream_t stream,
                Tensor<T, w_height, x_width> &Y, 
                Tensor<T, w_height, w_width> &W,
                Tensor<T, x_height, x_width> &X)
{
    dim3 dimGrid(x_width/FH_TILE_X, w_height/FH_TILE_X);
    dim3 dimBlock(FH_TILE_X, FH_TILE_Y);
    hipLaunchKernelGGL((FireHowlDot<T, w_height, w_width, x_height, x_width>), dimGrid, dimBlock, 0, stream, Y.getDPtr(), W.getDPtr(), X.getDPtr());
}

template<typename T, int w_height, int w_width, int x_height, int x_width>
void MatMul(hipStream_t stream,
                Tensor<T, w_height, x_width> &Y, 
                Tensor<T, w_height, w_width> &W,
                Tensor<T, x_height, x_width> &X,
                Tensor<T, w_height, x_width> &B)
{
    dim3 dimGrid(x_width/FH_TILE_X, w_height/FH_TILE_X);
    dim3 dimBlock(FH_TILE_X, FH_TILE_Y);
    hipLaunchKernelGGL((FireHowlDotBias<T, w_height, w_width, x_height, x_width>), dimGrid, dimBlock, 0, stream, Y.getDPtr(), W.getDPtr(), X.getDPtr(), B.dMat);
}

template<typename T, int height, int width>
void Tanh(hipStream_t stream,
            Tensor<T, height, width> &Y,
            Tensor<T, height, width> &X)
{
    dim3 dimGrid((width * height)/(FH_TILE_X * FH_TILE_X));
    dim3 dimBlock(FH_TILE_X * FH_TILE_Y);
    hipLaunchKernelGGL((FireHowlTanh<T, height, width>), dimGrid, dimBlock, 0, stream, Y.getDPtr(), X.getDPtr());

}

} // end namespace firehowl

#endif

#ifndef FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_H
#define FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_H

#include "firehowl_kernel.h"

namespace firehowl {

template<typename T, int w_height, int w_width, int x_height, int x_width>
void MatMul(Tensor<T, w_height, x_width> &Y, 
                Tensor<T, w_height, w_width> &W,
                Tensor<T, x_height, x_width> &X)
{
    hipStream_t stream = init.getStream();
    Y.CheckSyncDevice(stream);
    W.CheckSyncDevice(stream);
    X.CheckSyncDevice(stream);
    dim3 dimGrid(x_width/FH_TILE_X_16, w_height/FH_TILE_Y_16);
    dim3 dimBlock(FH_TILE_X_16, FH_TILE_Y_16);
    hipLaunchKernelGGL((FireHowlDot<T, w_height, w_width, x_height, x_width>), dimGrid, dimBlock, 0, stream, Y.getDPtr(), W.getDPtr(), X.getDPtr());
}

template<typename T, int w_height, int w_width, int x_height, int x_width>
void MatMul(Tensor<T, w_height, x_width> &Y, 
                Tensor<T, w_height, w_width> &W,
                Tensor<T, x_height, x_width> &X,
                Tensor<T, w_height, x_width> &B)
{
    hipStream_t stream = init.getStream();
    Y.CheckSyncDevice(stream);
    W.CheckSyncDevice(stream);
    X.CheckSyncDevice(stream);
    B.CheckSyncDevice(stream);
    dim3 dimGrid(x_width/FH_TILE_X_16, w_height/FH_TILE_Y_16);
    dim3 dimBlock(FH_TILE_X_16, FH_TILE_Y_16);
    hipLaunchKernelGGL((FireHowlDotBias<T, w_height, w_width, x_height, x_width>), dimGrid, dimBlock, 0, stream, Y.getDPtr(), W.getDPtr(), X.getDPtr(), B.dMat);
}

template<typename T, int height, int width>
void Tanh(Tensor<T, height, width> &Y,
            Tensor<T, height, width> &X)
{
    hipStream_t stream = init.getStream();
    Y.CheckSyncDevice(stream);
    X.CheckSyncDevice(stream);
    dim3 dimGrid((width * height)/(FH_TILE_X_16 * FH_TILE_X_16));
    dim3 dimBlock(FH_TILE_X_16 * FH_TILE_Y_16);
    hipLaunchKernelGGL((FireHowlTanh<T, height, width>), dimGrid, dimBlock, 0, stream, Y.getDPtr(), X.getDPtr());

}

} // end namespace firehowl

#endif

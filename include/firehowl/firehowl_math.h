#pragma once

#include"firehowl_defines.h"

template<typename T>
__device__ T __fh_exp(T input);

__device__ float __fh_exp(float input) {
    return __expf(input);
}

__device__ half __fh_exp(half input) {
    return __expf(float(input));
}

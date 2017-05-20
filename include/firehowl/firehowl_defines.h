#ifndef FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_DEFINES_H
#define FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_DEFINES_H

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#define FH_HIP_CHECK(status) \
    if(status != hipSuccess) { std::cout<<"Got: "<<hipGetErrorString(status)<<" at line: "<<__LINE__<<" in file: "<<__FILE__<<std::endl; }

#define FH_FUNC_NAME() std::cout<<"Func Name: "<<__func__<<std::endl;

#endif

#ifndef FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_DEFINES_H
#define FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_DEFINES_H

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>
#include<hip/hip_fp16.h>

#define CHECK_HOST(stream) if(devSync == 1) { SyncHost(stream); devSync = 0; hostSync = 1; }
#define CHECK_DEVICE(stream) if(hostSync == 1) { SyncDevice(stream); devSync = 1; hostSync = 0; }

#define CHECK_HSYNC(stream) if(devSync == 1) { SyncHost(stream); hipDeviceSynchronize(); devSync = 0; hostSync = 1; }
#define CHECK_DSYNC(stream) if(hostSync == 1) { SyncDevice(stream); hipDeviceSynchronize(); devSync = 1; hostSync = 0; }


#define FH_HIP_CHECK(status) \
    if(status != hipSuccess) { std::cout<<"Got: "<<hipGetErrorString(status)<<" at line: "<<__LINE__<<" in file: "<<__FILE__<<std::endl; }

#define FH_FUNC_NAME() std::cout<<"Func Name: "<<__func__<<std::endl;

#endif

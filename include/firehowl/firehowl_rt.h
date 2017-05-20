#ifndef FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_RT_H
#define FIREHOWL_INCLUDE_FIREHOWL_FIREHOWL_RT_H

#include "firehowl_defines.h"

#define FH_NUM_STREAMS 4

class fhInit{
private:
    std::vector<hipStream_t> vStreams;
    int id;
public:
    fhInit(){
        id = 0;
        vStreams.resize(FH_NUM_STREAMS);
        for(int i=0;i<FH_NUM_STREAMS;i++) {
            FH_HIP_CHECK(hipStreamCreate(&vStreams[i]));
        }
    }
    ~fhInit(){
        for(int i=0;i<FH_NUM_STREAMS;i++) {
            FH_HIP_CHECK(hipStreamDestroy(vStreams[i]));
        }
    }
    hipStream_t getStream(){ id++; if(id == 1024) { id=0; } return vStreams[id%FH_NUM_STREAMS];  }
};

static class fhInit init;

#endif

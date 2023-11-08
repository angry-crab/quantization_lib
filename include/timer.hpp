#ifndef TIMER_HPP
#define TIMER_HPP

#include "cuda_utils.hpp"

class EventTimer{
public:
    EventTimer(){
        CHECK_CUDA_ERROR(cudaEventCreate(&begin_));
        CHECK_CUDA_ERROR(cudaEventCreate(&end_));
    }

    virtual ~EventTimer(){
        CHECK_CUDA_ERROR(cudaEventDestroy(begin_));
        CHECK_CUDA_ERROR(cudaEventDestroy(end_));
    }

    void start(cudaStream_t stream){
        CHECK_CUDA_ERROR(cudaEventRecord(begin_, stream));
    }

    float stop(const char* prefix = "timer", bool print = true){
        float times = 0;
        CHECK_CUDA_ERROR(cudaEventRecord(end_, stream_));
        CHECK_CUDA_ERROR(cudaEventSynchronize(end_));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&times, begin_, end_));
        if(print) printf("[TIME] %s:\t\t%.5f ms\n", prefix, times);
        return times;
    }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t begin_ = nullptr, end_ = nullptr;
};




#endif // TIMER_HPP
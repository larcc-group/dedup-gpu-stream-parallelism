#pragma once
#include <iostream>
#include <vector>
#ifndef __CL_ENABLE_EXCEPTIONS
#define __CL_ENABLE_EXCEPTIONS
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif
#include "gpu_util.h"
#include <cuda_runtime.h>
class MatcherBase
{
  public:
    float timeSpentOnMemoryHostToDevice;
    float timeSpentOnMemoryDeviceToHost;
    float timeSpentOnKernel;

    virtual int Init()
    {
        timeSpentOnMemoryHostToDevice = 0;
        timeSpentOnMemoryDeviceToHost = 0;
        timeSpentOnKernel = 0;

        return 0;
    };
    virtual int FindMatchBatchUsingDeviceInput(unsigned char *input, int inputSize, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex)
    {
        printf("Operation not supported");
        exit(ENOTSUP);
    };

    virtual int FindMatchBatchUsingDeviceInput(cl::Buffer input, int inputSize, int offset, int *matches_length, int *matches_offset, int *matchSize)
    {
        printf("Operation not supported cl");
        exit(ENOTSUP);
    };
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch)
    {
        return -1;
    };
};

class MatcherSequential : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
    virtual int FindMatchBatchInMemory(unsigned char *input, int sizeInput, int * breakPositions, int breakSize,int *matches_length, int *matches_offset);
   
};
class MatcherCuda : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
    //Find all at once
    virtual int FindMatchBatchUsingDeviceInput(unsigned char *input, int inputSize, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex);
    virtual int FindMatchBatchInMemory(unsigned char *d_input, int sizeInput, int * d_breakPositions, int breakSize,int *matches_length, int *matches_offset, int device);
    virtual int FindMatchBatchInMemory(unsigned char *d_input, int sizeInput, int * d_breakPositions, int breakSize,int *matches_length, int *matches_offset, int device,cudaStream_t cuda_stream);

};
class MatcherOpenAcc : public MatcherBase
{
  public:
    virtual int Init();
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
};
class MatcherOpenCl : public MatcherBase
{
  public:
    virtual int Init();
    virtual int Init(OCLDeviceObjects* device);
    virtual int FindMatchBatch(char *buffer, int bufferSize, int *matches_length, int *matches_offset, int *matchSize, bool isLast, int currentMatchCount, int currentBatch);
    virtual int FindMatchBatchUsingDeviceInput(cl::Buffer input, int inputSize, int offsetLength, int *matches_length, int *matches_offset, int *matchSize,int deviceIndex);
#ifndef OPENCL_C
    virtual int FindMatchBatchInMemory(cl::Buffer d_input, int sizeInput, cl::Buffer d_breakPositions, int breakSize,int *matches_length, int *matches_offset, int device);
#else
    virtual int FindMatchBatchInMemory(cl_mem d_input, size_t sizeInput, cl_mem d_breakPositions, size_t breakSize,int *matches_length, int *matches_offset,  OCLDeviceObjects*  device,cl_command_queue queue);
#endif
  private:
    cl::Kernel FindMatchKernel;
    cl::Kernel FindMatchWithoutBufferKernel;
    cl::Kernel FindMatchBatchKernel;
    cl::Context context;

    #ifdef OPENCL_C
    cl_context contextC;
    cl_program programC;
    #endif
};

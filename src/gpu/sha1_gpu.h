#pragma once

#include "gpu_util.h"
#include <cuda_runtime.h>
struct Sha1OCLDeviceObjects {
    OCLDeviceObjects* device;
    cl_program sha1program;
    // cl_kernel kernel;

    cl_command_queue queue;
};
typedef struct Sha1OCLDeviceObjects Sha1OCLDeviceObjects;


int sha1DigestBatchWrapper(unsigned char * d_input, int sizeInput, int * d_beginOffset, int * length, unsigned char * sha1Output);
int sha1DigestBatchWrapperWithStream(unsigned char * d_input, int sizeInput, int * d_beginOffset, int * length, unsigned char * sha1Output,cudaStream_t cuda_stream);

// int sha1DigestBatchWrapperCl(cl_mem d_input,int sizeInput, cl_mem d_beginOffset, int * length, unsigned char * sha1Output);
int sha1DigestBatchWrapperClWithDevice(cl_mem d_input, int sizeInput, cl_mem  d_beginOffsets, int *lengths, unsigned char *sha1Output, Sha1OCLDeviceObjects* device, cl_command_queue queue);
Sha1OCLDeviceObjects* createSha1OCLDeviceObjects(OCLDeviceObjects* device);





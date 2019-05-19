
// #include "cudautils.h"
#include <iostream>
#include <fstream>
#include "gpu_util.h"
#include "sha1_gpu.h"

#define SHA1_LEN 20
#ifndef GPU_BLOCK_SIZE
#define GPU_BLOCK_SIZE 256
#endif




int sha1DigestBatchWrapperClWithDevice(cl_mem d_input, int sizeInput, cl_mem  d_beginOffsets, int *lengths, unsigned char *sha1Output, Sha1OCLDeviceObjects * device, cl_command_queue queue)
{

    
    auto context = device->device->context;
    // cl::Buffer d_beginOffsets(gpuContext, CL_MEM_READ_ONLY, sizeof(int) * sizeInput);
    cl_int err;
    cl_mem d_lengths = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * sizeInput, NULL, &err);
    OpenCLCheckError(err);

    cl_mem d_sha1Output = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeInput * sizeof(unsigned char) * SHA1_LEN, NULL, &err);
    OpenCLCheckError(err);

    
    int sizeToLaunch = sizeInput;

    int size = sizeToLaunch;
    if (sizeToLaunch % GPU_BLOCK_SIZE != 0)
    {
        size = sizeToLaunch + GPU_BLOCK_SIZE - (sizeToLaunch % (GPU_BLOCK_SIZE));
    }
    int i_parm = 0;     
    
    cl_kernel kernel = clCreateKernel(device->sha1program, "sha1BatchKernel", &err);
    OpenCLCheckError(err);
    OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &d_input) );
    OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &sizeInput) );
    OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &d_beginOffsets) );
    OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &d_lengths) );
    OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &d_sha1Output) );
    

    cl_event evt_write1;
    OpenCLCheckError( clEnqueueWriteBuffer(queue, d_lengths, CL_FALSE, 0, sizeof(int) * sizeInput, lengths, 0, NULL, &evt_write1) );


    size_t localSize[1] = { GPU_BLOCK_SIZE};
    // Total number of work-items
    size_t globalSize[1] = { size};    
    int dimensions = 1;
    // cl_event evt;
    OpenCLCheckError( clEnqueueNDRangeKernel(queue, kernel, dimensions, NULL, globalSize, NULL, 0, NULL, NULL) );
    
    OpenCLCheckError( clEnqueueReadBuffer(queue, d_sha1Output, CL_FALSE, 0, sizeInput * sizeof(unsigned char) * SHA1_LEN, sha1Output,0, NULL, NULL) );

    // OpenCLCheckError( clWaitForEvents(1, &evt_read1) );
    err = clFinish(queue);
    OpenCLCheckError(err);
    clReleaseMemObject(d_lengths);
    clReleaseMemObject(d_sha1Output);
    clReleaseKernel(kernel);
   
    return 0;
}



Sha1OCLDeviceObjects* createSha1OCLDeviceObjects(OCLDeviceObjects* device){
    Sha1OCLDeviceObjects* sha1Ocl = new Sha1OCLDeviceObjects();
    sha1Ocl->device = device;

    std::ifstream sourceFile("gpu/sha1_gpu.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    
    cl_int status;
    cl_context context = device->context;
    const char *kernelSource = sourceCode.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &status);
    status = clBuildProgram(program, 1, &device->device, NULL, NULL, NULL);
    OpenCLCheckBuildError(status, program, &device->device);
    sha1Ocl->sha1program = program;
    return sha1Ocl;


}
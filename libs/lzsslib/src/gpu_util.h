

#pragma once 

#include <CL/cl.hpp>
#include <vector>
#include <CL/opencl.h>
#include "oclbase.h"
#define OCL_ERROR_CHECK

void setDeviceIds(std::vector<int> deviceIds);
std::vector<int> getDeviceIds();
std::vector<cl::Device> getOpenClDevices();
cl::Context getOpenClDefaultContext();
cl::CommandQueue getOpenClDefaultCommandQueue(int index);


cl_context getOpenClDefaultContextC();
cl_command_queue getOpenClDefaultCommandQueueC(int index);
std::vector<cl_device_id> getOpenClDevicesC();


int logOpenClBuildError(cl::Program program, cl::Error err);

struct OCLDeviceObjects {
    cl_device_id device;
    cl_context context;
    // cl_command_queue queue;
    cl_program program;
    // cl_kernel kernel;
};

OCLDeviceObjects* createOCLDevice(int deviceIndex);
void releaseOCLDevice(OCLDeviceObjects* oclDevice);
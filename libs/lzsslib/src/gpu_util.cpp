#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include "gpu_util.h"
#include "oclbase.h"
#include <fstream>

std::vector<int> deviceIds = {0};
std::vector<cl::Device> devices;
cl::Context context;
std::vector<cl::CommandQueue> queue;

bool initialized = false;

bool initializedC = false;
std::vector<cl_command_queue> queueC;
std::vector<cl_device_id> devicesC;
cl_context contextC;


void setDeviceIds(std::vector<int> item)
{
    deviceIds = item;
}
std::vector<int> getDeviceIds()
{
    return deviceIds;
}
void initOpenCl()
{
    printf("initOpenCl\n");
    try
    {
        // printf("initOpenCL\n");

        unsigned int platform_id = 0;

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        platforms[platform_id].getDevices(CL_DEVICE_TYPE_GPU, &devices); // Select the platform.
        std::vector<cl::Device> devicesUsed = {};
        
        for (size_t i = 0; i < deviceIds.size(); i++)
        {
            if (devices.size() - 1 < deviceIds[i])
            {
                printf("Device %i not found\n",deviceIds[i]);
                exit(-1);
            }
            devicesUsed.push_back(devices[deviceIds[i]]); // Select the device.
        }
        
        // context = cl::Context(devicesUsed);
        context = cl::Context(devices);
        
        for (size_t i = 0; i < deviceIds.size(); i++)
        {
            printf("Using device %i \n",deviceIds[i]);
            if (devices.size() - 1 < deviceIds[i])
            {
                printf("Device %i not found\n",deviceIds[i]);
                exit(-1);
            }
            queue.push_back(cl::CommandQueue(context, devices[deviceIds[i]])); // Select the device.
        }
    }
    catch (cl::Error err)
    {
        printf( "Error: %i(%s) ", err.what(), err.err());
        exit(EXIT_FAILURE);
    }
    initialized = true;
}


cl::Context getOpenClDefaultContext()
{
    if (!initialized)
    {
        initOpenCl();
    }
    return context;
}
std::vector<cl::Device> getOpenClDevices()
{
    if (!initialized)
    {
        initOpenCl();
    }
    return devices;
}
cl::CommandQueue getOpenClDefaultCommandQueue(int index)
{
    if (!initialized)
    {
        initOpenCl();
    }
    return queue[index % queue.size()];
}

int logOpenClBuildError(cl::Program program, cl::Error err)
{
    if (err.err() == CL_BUILD_PROGRAM_FAILURE)
    {
        for (cl::Device dev : devices)
        {
            // Check the build status
            cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
            if (status != CL_BUILD_ERROR)
                continue;

            // Get the build log
            std::string name = dev.getInfo<CL_DEVICE_NAME>();
            std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            // std::cerr << "Build log for " << name << ":" << std::endl
            //             << buildlog << std::endl;

            printf("Error in build for %s: %s\n", name.c_str(), buildlog.c_str());
        }
        exit(EXIT_FAILURE);
    }
    return 0;
}



















void getGPUs(cl_device_id** devices, int* total_devices) {

    int total = 0;
    
    cl_uint platformCount;
    OpenCLCheckError( clGetPlatformIDs(0, NULL, &platformCount) );

    cl_platform_id* platforms = new cl_platform_id[platformCount];
    OpenCLCheckError( clGetPlatformIDs(platformCount, platforms, NULL) );

    for (int i = 0; i < platformCount; ++i) {
        cl_uint deviceCount;
        OpenCLCheckError( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount) );
        
        cl_device_id* dev_aux = new cl_device_id[total + deviceCount];
        if (total > 0) {
            memcpy(dev_aux, **devices, total);
        }
        if (*devices) delete *devices;
        *devices = dev_aux;

        OpenCLCheckError( clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, deviceCount, &((*devices)[total]), NULL) );
        total += deviceCount;
    }
    
    *total_devices = total;
}
void initOpenClC()
{
    cl_int status;
    cl_device_id* devices = NULL;
    int total_devices = 0;
    getGPUs(&devices, &total_devices);
    std::vector<cl_device_id> devicesUsed = {};
        
    for (size_t i = 0; i < total_devices; i++)
    {
        if (total_devices - 1 < deviceIds[i])
        {
            printf("Device %i not found\n",deviceIds[i]);
            exit(-1);
        }
        devicesUsed.push_back(devices[deviceIds[i]]); // Select the device.
    }
    devicesC = devicesUsed;
    cl_context context = clCreateContext(NULL, total_devices, devices, NULL, NULL, &status);
    OpenCLCheckError(status);

    contextC = context;


    for (size_t i = 0; i <total_devices; i++)
    {
        printf("Using device %i \n",deviceIds[i]);
        cl_command_queue queue = clCreateCommandQueue(context, devices[i], 0, &status);
        queueC.push_back( clCreateCommandQueue(context, devices[i], 0, &status));
        OpenCLCheckError(status);
       
    }
    initializedC = true;
}


cl_context getOpenClDefaultContextC()
{
    if(!initializedC){
        initOpenClC();
    }
    return contextC;
}
cl_command_queue getOpenClDefaultCommandQueueC(int index)
{
    if(!initializedC){
        initOpenClC();
    }
    return queueC[index];
}

std::vector<cl_device_id> getOpenClDevicesC(){
    if(!initializedC){
        initOpenClC();
    }
    return devicesC;
}




OCLDeviceObjects* createOCLDevice(int deviceIndex){
    cl_int status;

    
    cl_device_id* devices = NULL;
    int total_devices = 0;
    getGPUs(&devices, &total_devices);

    auto device = devices[deviceIndex];


    auto oclDevice = new OCLDeviceObjects();
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    OpenCLCheckError(status);
    oclDevice->context = context;
    oclDevice->device = device;
    // cl_command_queue queue = clCreateCommandQueue(context, device, 0, &status);
    // OpenCLCheckError(status);

    
    // oclDevice->queue = queue;

    std::ifstream sourceFile("matcher_kernel_opencl.cl");
    std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

    const char *kernelSource = sourceCode.c_str();
    oclDevice->program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &status);
    OpenCLCheckError(status);

    status = clBuildProgram(oclDevice->program , 1, &device, NULL, NULL, NULL);
    OpenCLCheckBuildError(status, programC, &device);

    return oclDevice;
}

void releaseOCLDevice(OCLDeviceObjects* oclDevice){
    // clReleaseCommandQueue(oclDevice->queue);
    clReleaseContext(oclDevice->context);
    clReleaseProgram(oclDevice->program);
    delete oclDevice;
}

void compileLzss(cl_program* program, cl_device_id* device){
}
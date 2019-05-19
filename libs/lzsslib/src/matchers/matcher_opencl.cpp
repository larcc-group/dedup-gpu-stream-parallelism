#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <chrono>
#include <fstream>
#include "lzss.h"

#include "lzlocal.h"
#include "matcher_base.h"
#include "gpu_util.h"

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>


cl::Kernel FindMatchKernelCache;
cl::Kernel FindMatchWithoutBufferKernelCache;
cl::Kernel FindMatchBatchKernelCache;

std::string sourceCodeCache;
#ifdef OPENCL_C
cl_program programCCache;
#endif
bool openClKernelInitialized = false;
int MatcherOpenCl::Init(){
    MatcherOpenCl::Init(NULL);
}
int MatcherOpenCl::Init(OCLDeviceObjects* device){
    MatcherBase::Init();
    #ifdef OPENCL_C

    if(device != NULL){
        return 0 ;
    }
    #endif
    if(openClKernelInitialized){

        context = getOpenClDefaultContext();    

        FindMatchKernel = FindMatchKernelCache;
        FindMatchWithoutBufferKernel = FindMatchWithoutBufferKernelCache;
        FindMatchBatchKernel = FindMatchBatchKernelCache;
        

        #ifdef OPENCL_C
        programC = programCCache;
        contextC =  device->context;
        #endif
        return 0;
    }
    // Read the program source
    std::ifstream sourceFile("matcher_kernel_opencl.cl");
    std::string sourceCode( std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    sourceCodeCache = sourceCode;
    #ifdef OPENCL_C

    printf("compiling...\n");
    cl_int status;
    cl_context contextC = getOpenClDefaultContextC();
    const char *kernelSource = sourceCode.c_str();
    programC = clCreateProgramWithSource(contextC, 1, (const char**)&kernelSource, NULL, &status);
    OpenCLCheckError(status);
    
    programCCache = programC;
    status = clBuildProgram(programC, getOpenClDevicesC().size(), getOpenClDevicesC().data(), NULL, NULL, NULL);
    OpenCLCheckBuildError(status, programC, getOpenClDevicesC()[0]);
    // FindMatchBatchKernelC = clCreateKernel(programC, "FindMatchBatchInMemoryKernel", &status);
    #endif
    cl::Program program;
    try{

        context = getOpenClDefaultContext();    

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
        // Make program from the source code
        program = cl::Program(context, source);

        // Build the program for the devices
        program.build(getOpenClDevices());


        // Make kernel
        FindMatchKernel = cl::Kernel (program, "FindMatchBatchKernel");
        FindMatchWithoutBufferKernel = cl::Kernel (program, "FindMatchBatchKernelWithoutBuffer");
        FindMatchBatchKernel = cl::Kernel (program, "FindMatchBatchInMemoryKernel");
        openClKernelInitialized  = true;
        FindMatchKernelCache = FindMatchKernel;
        FindMatchWithoutBufferKernelCache = FindMatchWithoutBufferKernel;
        FindMatchBatchKernelCache = FindMatchBatchKernel;
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        logOpenClBuildError(program, err); 
    }

}

int MatcherOpenCl::FindMatchBatch(char* buffer, int bufferSize, int* matches_length, int* matches_offset, int* matchSize, bool isLast,int currentMatchCount , int currentBatch) {
	
    int bufferSizeAdjusted = bufferSize - MAX_CODED;
    auto queue = getOpenClDefaultCommandQueue(currentBatch);
	if (isLast) {
		bufferSizeAdjusted += MAX_CODED;
    }
    int matchCount = bufferSizeAdjusted - WINDOW_SIZE;
	*matchSize = matchCount;
    
    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + ( sizeToLaunch % BLOCK_SIZE > 0?1 : 0);
    int threads = BLOCK_SIZE; 


     try{

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        cl::Buffer buffer_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * bufferSize);
        cl::Buffer buffer_matches_length = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);
        cl::Buffer buffer_matches_offset = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);

        queue.enqueueWriteBuffer( buffer_buffer, CL_FALSE, 0, sizeof(char) * bufferSize, buffer );
        queue.finish();
        
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        timeSpentOnMemoryHostToDevice += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        // Set the kernel arguments
        FindMatchKernel.setArg( 0, buffer_buffer );
        FindMatchKernel.setArg( 1, bufferSize );
        FindMatchKernel.setArg( 2, buffer_matches_length);
        FindMatchKernel.setArg( 3, buffer_matches_offset);
        FindMatchKernel.setArg( 4, bufferSizeAdjusted );
        FindMatchKernel.setArg( 5, currentMatchCount );
        FindMatchKernel.setArg( 6, isLast?1:0 );


        int size = sizeToLaunch;
        if(sizeToLaunch % BLOCK_SIZE != 0){
            size = sizeToLaunch + BLOCK_SIZE - (sizeToLaunch % (BLOCK_SIZE) );
        }
        // Execute the kernel
        cl::NDRange global( size  );
        cl::NDRange local( BLOCK_SIZE  );
        
        #ifdef DEBUG
            std::cout 
                << "Size launch "
                << global[0]  << "."
                << local[0] << "." 
                << BLOCK_SIZE << "." 
                << sizeToLaunch << "." 
                << std::endl;
        #endif
        queue.enqueueNDRangeKernel( FindMatchKernel, cl::NullRange, global, local );
        queue.finish();

        end= std::chrono::steady_clock::now();
        timeSpentOnKernel += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        queue.enqueueReadBuffer( buffer_matches_length, CL_TRUE, 0, sizeof(int) * matchCount, matches_length );
        queue.enqueueReadBuffer( buffer_matches_offset, CL_TRUE, 0, sizeof(int) * matchCount, matches_offset );
        queue.finish();
        end= std::chrono::steady_clock::now();
        timeSpentOnMemoryDeviceToHost += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}



int MatcherOpenCl::FindMatchBatchUsingDeviceInput(cl::Buffer input, int inputSize, int offset, int *matches_length, int *matches_offset, int *matchSize, int deviceIndex){
    cl_int err = 0;
    auto queue = getOpenClDefaultCommandQueue(deviceIndex);
   
	
    int matchCount = inputSize;
    *matchSize = inputSize;
    
    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + ( sizeToLaunch % BLOCK_SIZE > 0?1 : 0);
    int threads = BLOCK_SIZE; 


     try{
        if(offset > 0){
            cl_buffer_region region = {offset,inputSize};
            // cl::BufferRegion region = ;
            input = input.createSubBuffer(CL_MEM_READ_ONLY,CL_BUFFER_CREATE_TYPE_REGION,  &region, &err);
        }
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        // cl::Buffer buffer_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * bufferSize);
        cl::Buffer buffer_matches_length = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);
        cl::Buffer buffer_matches_offset = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);

        // queue.enqueueWriteBuffer( buffer_buffer, CL_FALSE, 0, sizeof(char) * bufferSize, buffer );
        // queue.finish();
        
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        timeSpentOnMemoryHostToDevice += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        // Set the kernel arguments
        FindMatchWithoutBufferKernel.setArg( 0, input );
        FindMatchWithoutBufferKernel.setArg( 1, inputSize );
        FindMatchWithoutBufferKernel.setArg( 2, buffer_matches_length);
        FindMatchWithoutBufferKernel.setArg( 3, buffer_matches_offset);


        int size = sizeToLaunch;
        if(sizeToLaunch % BLOCK_SIZE != 0){
            size = sizeToLaunch + BLOCK_SIZE - (sizeToLaunch % (BLOCK_SIZE) );
        }
        // Execute the kernel
        cl::NDRange global( size  );
        cl::NDRange local( BLOCK_SIZE  );
        
        #ifdef DEBUG
            std::cout 
                << "Size launch "
                << global[0]  << "."
                << local[0] << "." 
                << BLOCK_SIZE << "." 
                << sizeToLaunch << "." 
                << std::endl;
        #endif
        queue.enqueueNDRangeKernel( FindMatchWithoutBufferKernel, cl::NullRange, global, local );
        queue.finish();

        end= std::chrono::steady_clock::now();
        timeSpentOnKernel += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        begin = std::chrono::steady_clock::now();
        queue.enqueueReadBuffer( buffer_matches_length, CL_TRUE, 0, sizeof(int) * matchCount, matches_length );
        queue.enqueueReadBuffer( buffer_matches_offset, CL_TRUE, 0, sizeof(int) * matchCount, matches_offset );
        queue.finish();
        end= std::chrono::steady_clock::now();
        timeSpentOnMemoryDeviceToHost += std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    }
    catch(cl::Error err) {
        std::cout << "Error: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}

#ifndef OPENCL_C
int MatcherOpenCl::FindMatchBatchInMemory(cl::Buffer d_input, int sizeInput, cl::Buffer d_breakPositions, int breakSize,int *matches_length, int *matches_offset, int device){
    cl_int err = 0;
    auto queue = getOpenClDefaultCommandQueue(device);
    auto context = getOpenClDefaultContext();
	
    int matchCount = sizeInput;    
    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + ( sizeToLaunch % BLOCK_SIZE > 0?1 : 0);
    int threads = BLOCK_SIZE; 

     try{
        // cl::Buffer buffer_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * bufferSize);
        cl::Buffer buffer_matches_length = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);
        cl::Buffer buffer_matches_offset = cl::Buffer(context,  CL_MEM_READ_ONLY, sizeof(int) * matchCount);


        // Set the kernel arguments
        FindMatchBatchKernelCache.setArg( 0, d_input );
        FindMatchBatchKernelCache.setArg( 1, sizeInput );
        FindMatchBatchKernelCache.setArg( 2, d_breakPositions);
        FindMatchBatchKernelCache.setArg( 3, breakSize);
        FindMatchBatchKernelCache.setArg( 4, buffer_matches_length);
        FindMatchBatchKernelCache.setArg( 5, buffer_matches_offset);


        int size = sizeToLaunch;
        if(sizeToLaunch % BLOCK_SIZE != 0){
            size = sizeToLaunch + BLOCK_SIZE - (sizeToLaunch % (BLOCK_SIZE) );
        }
        // Execute the kernel
        cl::NDRange global( size  );
        cl::NDRange local( BLOCK_SIZE  );
        
        queue.enqueueNDRangeKernel( FindMatchBatchKernelCache, cl::NullRange, global, local );
        queue.finish();


        queue.enqueueReadBuffer( buffer_matches_length, CL_FALSE, 0, sizeof(int) * matchCount, matches_length );
        queue.enqueueReadBuffer( buffer_matches_offset, CL_FALSE, 0, sizeof(int) * matchCount, matches_offset );
        queue.finish();
    }
    catch(cl::Error err) {
        std::cout << "Error FindMatchBatchInMemory: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}

#else
int MatcherOpenCl::FindMatchBatchInMemory(cl_mem d_input, size_t sizeInput, cl_mem d_breakPositions, size_t breakSize,int *matches_length, int *matches_offset, OCLDeviceObjects* device,cl_command_queue queue){
    cl_int err = 0;
    auto context = device->context;
	
    int matchCount = sizeInput;    
    int sizeToLaunch = matchCount;
    int blocks = sizeToLaunch / BLOCK_SIZE + ( sizeToLaunch % BLOCK_SIZE > 0?1 : 0);
    int threads = BLOCK_SIZE; 

     try{
        // cl::Buffer buffer_buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * bufferSize);
        cl_mem buffer_matches_length = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * matchCount, NULL, &err);
        OpenCLCheckError(err);
        cl_mem buffer_matches_offset = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * matchCount, NULL, &err);
        OpenCLCheckError(err);
        int i_parm = 0;


        cl_kernel kernel = clCreateKernel(device->program, "FindMatchBatchInMemoryKernel", &err);
        OpenCLCheckError(err);

        OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &d_input) );
        OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &sizeInput) );
        OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &d_breakPositions) );
        OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(int), &breakSize) );
        OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &buffer_matches_length) );
        OpenCLCheckError( clSetKernelArg(kernel, i_parm++, sizeof(cl_mem), &buffer_matches_offset) );




        int size = sizeToLaunch;
        if(sizeToLaunch % BLOCK_SIZE != 0){
            size = sizeToLaunch + BLOCK_SIZE - (sizeToLaunch % (BLOCK_SIZE) );
        }
        
         size_t localSize[1] = { BLOCK_SIZE};
        // Total number of work-items
        size_t globalSize[1] = { size};
        int dimensions = 1;
        err =  clEnqueueNDRangeKernel(queue, kernel, dimensions, NULL, globalSize, localSize, 0, NULL,NULL) ;
        OpenCLCheckError(err);

            err  = (clFinish(queue));
        OpenCLCheckError(err);
        err = ( clEnqueueReadBuffer(queue, buffer_matches_length, CL_FALSE, 0, sizeof(int) * matchCount, matches_length, 0, NULL, NULL) );
        OpenCLCheckError(err);
            err  = (clFinish(queue));
        OpenCLCheckError(err);
        err = ( clEnqueueReadBuffer(queue, buffer_matches_offset, CL_FALSE, 0, sizeof(int) * matchCount, matches_offset, 0, NULL, NULL) );
        OpenCLCheckError(err);
        // OpenCLCheckError( clWaitForEvents(1, &evt_read2) );
        err  = (clFinish(queue));
        OpenCLCheckError(err);

        clReleaseKernel(kernel);
        clReleaseMemObject(buffer_matches_length);
        clReleaseMemObject(buffer_matches_offset);
    }
    catch(cl::Error err) {
        std::cout << "Error FindMatchBatchInMemory: " << err.what() << "(" << err.err() << ")" << std::endl;
        exit( EXIT_FAILURE );
    }
	return 0;
}

#endif
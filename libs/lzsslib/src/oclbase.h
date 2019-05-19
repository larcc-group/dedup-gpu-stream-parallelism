#ifndef __OCLBASE_INCLUDED__
#define __OCLBASE_INCLUDED__

#include <CL/opencl.h>

#define OCL_ERROR_CHECK

const char *__openCLGetErrorString(cl_int error);
// #define OpenCLCheckError(status)    __openCLCheckError( status, __FILE__, __LINE__ )
#define OpenCLCheckError(status)      if(status != 0 ) {printf("ERROR IN OPERATION ON %s : %i %i\n",__FILE__,__LINE__,status);exit(-1);}
// #define OpenCLCheckBuildError(status, program, device)    __openCLPrintBuildError( status, program, device, __FILE__, __LINE__ )
#define OpenCLCheckBuildError(status, program, device)    if(status != 0 ) printf("ERROR IN BUILD");

// void __openCLCheckError( cl_int status, const char *file, const int line );

// void __openCLPrintBuildError(cl_int status, cl_program program, cl_device_id device, const char *file, const int line);

#endif
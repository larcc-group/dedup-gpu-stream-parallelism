/***************************************************************************
*          Lempel, Ziv, Storer, and Szymanski Encoding and Decoding
*
*   File    : lzss.h
*   Purpose : Header for LZSS encode and decode routines.  Contains the
*             prototypes to be used by programs linking to the LZSS
*             library.
*   Author  : Michael Dipperstein
*   Date    : February 21, 2004
*
****************************************************************************
*
* LZSS: An ANSI C LZSS Encoding/Decoding Routine
* Copyright (C) 2004, 2006, 2007, 2014 by
* Michael Dipperstein (mdipper@alumni.engr.ucsb.edu)
*
* This file is part of the lzss library.
*
* The lzss library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 3 of the
* License, or (at your option) any later version.
*
* The lzss library is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser
* General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
***************************************************************************/
#ifndef _LZSS_H
#define _LZSS_H

/***************************************************************************
*                               PROTOTYPES
***************************************************************************/

/***************************************************************************
* LZSS encoding and decoding prototypes for functions with file pointer
* parameters.  Provide these functions with a pointer to the open binary
* file to be encoded/decoded (fpIn) and pointer to the open binary target
* file (fpOut).  It is the job of the function caller to open the files
* prior to callings these functions and to close the file after these
* functions have been called.
*
* These functions return 0 for success and -1 for failure.  errno will be
* set in the event of a failure. 
***************************************************************************/
#include "file_stream.h"
#include "statistics.h"
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif
#include <CL/opencl.h>
#include "gpu_util.h"
#include <cuda_runtime.h>
#include <vector>
typedef enum{
    CPU_ORIGINAL,
    CPU_SEQUENTIAL,
    GPU_CUDA,
    GPU_OPENACC,
    GPU_OPENCL,
} ExecutionPlan;

#define SMALL_BUFFER_FAILURE 101

extern int LzssBatchSize;

int EncodeLZSSGpu(FileStream* fpIn, FileStream *fpOut, ExecutionPlan plan, AppStatistics* metrics);
//int EncodeLZSSGpu(FileStream* fpIn, FileStream *fpOut, ExecutionPlan plan, AppStatistics* metrics);
int EncodeLZSSCpuGpu(FileStream* fpIn, FileStream *fpOut, ExecutionPlan plan, AppStatistics* metrics, int workers);
int EncodeLZSS(FileStream* fpIn, FileStream *fpOut,AppStatistics* metrics);
int DecodeLZSS(FileStream *fpIn, FILE *fpOut);

int LzssEncodeMemory(unsigned char * input, int sizeInput, unsigned char * output, int sizeOutput, int * outCompressedSize);

int LzssEncodeMemoryGpu(unsigned char * input,  int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize);

int LzssEncodeMemoryGpu(unsigned char *input, unsigned char *d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize);

int LzssEncodeMemoryGpu(unsigned char *input, unsigned char *d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize, int device);

int LzssDecodeMemory(unsigned char * input, int sizeInput, unsigned char * output, int sizeOutput, int * outDecompressedSize);

int LzssEncodeMemoryGpu(unsigned char *input, cl::Buffer d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize);

int LzssEncodeMemoryGpu(unsigned char *input, cl::Buffer d_input, int sizeInput, unsigned char *output, int sizeOutput, int *outCompressedSize, int deviceId);

// int LzssEncodeMemoryGpuBatch(unsigned char *input, unsigned char *d_input, int sizeInput, int * breakPositions, vector<vector<char>> *output, int device);
using namespace std;
int LzssEncodeMemoryCpuBatch(unsigned char *input, int sizeInput, int * breakPositions, int breakSize, vector<vector<unsigned char>> *output);
int LzssEncodeMemoryGpuBatch(unsigned char *input,unsigned char *d_input, int sizeInput, int * breakPositions, int * d_breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output, int device);
int LzssEncodeMemoryGpuBatch(unsigned char *input,unsigned char *d_input, int sizeInput, int * breakPositions, int * d_breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output, int device, cudaStream_t cuda_stream);
// #ifndef OPENCL_C
int LzssEncodeMemoryGpuBatch(unsigned char *input, cl::Buffer d_input, int sizeInput, int * breakPositions,  cl::Buffer d_breakPositions, int breakSize, std::vector<std::vector<unsigned char>> *output, int device);
// #else
int LzssEncodeMemoryGpuBatch(unsigned char *input, cl_mem d_input, size_t sizeInput, int * breakPositions,  cl_mem d_breakPositions, size_t breakSize, std::vector<std::vector<unsigned char>> *output, OCLDeviceObjects* device, cl_command_queue queue);
// #endif
#endif      /* ndef _LZSS_H */

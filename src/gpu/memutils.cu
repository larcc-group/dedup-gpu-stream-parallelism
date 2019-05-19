#include "memutils.h"
#include "cudautils.h"
void allocateGpu(void ** pointer, size_t size){
    CUDA_SAFE_CALL(cudaMalloc(pointer, size));
}
void setDevice(int index){
    cudaSetDevice(index);
}
void freeGpu(void * pointer){
    cudaFree(pointer);
}
void moveToGpu(void * dst, void * src, size_t size){
    CUDA_SAFE_CALL(cudaMemcpyAsync(dst,src,size, cudaMemcpyHostToDevice));
}
void startGpu(){
    cudaSetDevice(0);
}
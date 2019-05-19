#pragma once
void allocateGpu(void ** pointer, size_t size);
void setDevice(int index);
void moveToGpu(void * dst, void * src, size_t size);
void freeGpu(void * pointer);
void startGpu();
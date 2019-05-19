# Makefile for dedup kernel

CUDAPATH = /usr/local/cuda/lib64 
CUDAINCLUDE = /usr/local/cuda/include
CC=/usr/bin/gcc
CXX = g++
SPAR_PATH = $(HOME)/spar/
SPAR = $(SPAR_PATH)bin/spar
SPAR_LIB = $(SPAR_PATH)libraries
NVCC=nvcc
NVCCFLAGS = --std=c++11  -D__CL_ENABLE_EXCEPTIONS   -DOPENCL_C
CXXFLAGS = --std=c++11 -fpermissive
CFLAGS +=  -Wall -fno-strict-aliasing -D_XOPEN_SOURCE=600 
CF=-O3 -g -DPARSEC_VERSION=3.0-beta-20150206
# CF=-O3 -g -funroll-loops -static-libgcc -Wl,--hash-style=both,--as-needed -DPARSEC_VERSION=3.0-beta-20150206

SSL=../libs/ssl/include
LSSL=../libs/ssl/lib

ZLIB=../libs/zlib/include
LZLIB=..../libs/zlib/lib

BZLIB=../libs/bzip2
LBZLIB=../libs/bzip2



LZSSLIB=../libs/lzsslib/src
LZSSLIB=../libs/lzsslib/src

LIBS += -lm


# Uncomment the following to enable lzss compression
CFLAGS += -DENABLE_LZSS_COMPRESSION
LIBS += -llzss

# Uncomment the following to enable gzip compression
CFLAGS += -DENABLE_GZIP_COMPRESSION 
LIBS += -lz

# Uncomment the following to enable bzip2 compression
CFLAGS += -DENABLE_BZIP2_COMPRESSION
LIBS += -lbz2

CFLAGS += -w


CFLAGS += -DENABLE_GZIP_COMPRESSION 
CFLAGS += -D__CL_ENABLE_EXCEPTIONS 

# CFLAGS += -DUSE_OPENCL
# CFLAGS += -DPRINT_HASHES
# CFLAGS += -DUSE_VECTOR
#CFLAGS += -DENABLE_STATISTICS -pthread

LIBINCLUDES =  -I $(SSL) -I $(ZLIB) -I $(BZLIB) -I $(LZSSLIB) -I $(CUDAINCLUDE)
all: sequential cuda opencl cuda_spar opencl_spar spar cuda_spar_2xmemory opencl_spar_2xmemory 
sequential: lzsslib
	$(NVCC) -g -c $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	$(NVCC) -g -c $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar.cpp -o encoder_spar.o
	# $(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar.cpp -o encoder_spar.o -spar_ordered
	# $(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.cpp -o encoder_spar_cl.o -spar_ordered
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_sequential encoder_spar.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o

spar: 
	$(NVCC) -g -c -DENABLE_SPAR $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	$(NVCC) -g -c -DENABLE_SPAR $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DENABLE_SPAR $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	# $(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar_new.cpp -o encoder_spar_new.o
	# $(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar.cpp -o encoder_spar.o
	# $(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar.cpp -o encoder_spar.o -spar_ordered
	$(SPAR) -c -DENABLE_SPAR  $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_new.cpp -o encoder_spar_new.o -spar_ordered
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c -DENABLE_SPAR $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) -DENABLE_SPAR $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_spar encoder_spar_new.o hashtable.o util.o dedup.o rabin.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o

opencl:
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar_cl.alternative.cpp -o encoder_spar_cl.o
	# $(SPAR) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.cpp -o encoder_spar_cl.o -spar_ordered
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_opencl encoder_spar_cl.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o


opencl_spar:
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  $(LZSSLIB)/gpu_util.cpp -o gpu_util.o -lOpenCL
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar_cl.cpp -o encoder_spar_cl.o
	$(SPAR) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.alternative.cpp -o encoder_spar_cl.o -spar_ordered -spar_ondemand
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_opencl_spar encoder_spar_cl.o $(LZSSLIB)/gpu_util.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o
opencl_spar_2xmemory:
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  $(LZSSLIB)/gpu_util.cpp -o gpu_util.o -lOpenCL
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar_cl.cpp -o encoder_spar_cl.o
	$(SPAR) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.alternative.2xmemory.cpp -o encoder_spar_cl.o -spar_ordered -spar_ondemand
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_opencl_spar_2xmemory encoder_spar_cl.o $(LZSSLIB)/gpu_util.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o

# opencl_spar:
# 	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
# 	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
# 	$(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
# 	# $(NVCC) -g -c -DUSE_OPENCL -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  $(LZSSLIB)/gpu_util.cpp -o gpu_util.o -lOpenCL
# 	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
# 	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar_cl.cpp -o encoder_spar_cl.o
# 	$(SPAR) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.cpp -o encoder_spar_cl.o -spar_ordered
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
# 	$(CXX) -c $(CF) -DUSE_OPENCL -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
# 	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_opencl_spar encoder_spar_cl.o $(LZSSLIB)/gpu_util.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

# 	rm -f *.o


cuda:
	$(NVCC) -g -c -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	$(NVCC) -g -c -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	$(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar_cuda.alternative.cpp -o encoder_spar_cuda.o
	# $(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar.cpp -o encoder_spar.o -spar_ordered
	# # $(SPAR) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.cpp -o encoder_spar_cl.o -spar_ordered
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_cuda encoder_spar_cuda.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o


cuda_spar:
	$(NVCC) -g -c -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	# $(NVCC) -g -c -DENABLE_SERIAL_GPU  $(NVCCFLAGS) -I $(LZSSLIB) gpu_util.cpp -o gpu_util.o
	$(NVCC) -g -c -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DENABLE_SERIAL_GPU $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar.cpp -o encoder_spar.o
	# $(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -I ./ -spar_file encoder_spar_cuda.cpp -o encoder_spar_cuda.o -spar_ordered
	$(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -I ./ -spar_file encoder_spar_cuda.alternative.cpp -o encoder_spar_cuda.o -spar_ordered
	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -I ./ encoder_spar_cuda.alternative.cpp -o encoder_spar_cuda.o
	# # $(SPAR) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.cpp -o encoder_spar_cl.o -spar_ordered
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_cuda_spar encoder_spar_cuda.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o $(LZSSLIB)/gpu_util.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o



cuda_spar_2xmemory:
	$(NVCC) -g -c -DENABLE_SERIAL_GPU  -DCUDA_MALLOC  $(NVCCFLAGS) -I $(LZSSLIB) gpu/memutils.cu -o gpu/memutils.o
	# $(NVCC) -g -c -DENABLE_SERIAL_GPU  -DCUDA_MALLOC  $(NVCCFLAGS) -I $(LZSSLIB) gpu_util.cpp -o gpu_util.o
	$(NVCC) -g -c -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(NVCCFLAGS) -I $(LZSSLIB) gpu/sha1_gpu.cu -o gpu/sha1_gpu.o
	$(NVCC) -g -c -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(NVCCFLAGS) -I $(LZSSLIB)  gpu/sha1_gpu_cl.cpp -o gpu/sha1_gpu_cl.o
	# $(NVCC) -g -c $(NVCCFLAGS) gpu/sha1.cu -o gpu/sha1.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC  $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) hashtable.cpp -o hashtable.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) util.cpp -o util.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) dedup.cpp -o dedup.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) rabin.cpp -o rabin.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder.cpp -o encoder.o
	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) encoder_spar.cpp -o encoder_spar.o
	# $(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -I ./ -spar_file encoder_spar_cuda.cpp -o encoder_spar_cuda.o -spar_ordered
	$(SPAR) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -I ./ -I./gpu -I$(CUDAINCLUDE) -spar_file encoder_spar_cuda.alternative.2xmemory.cpp -o encoder_spar_cuda.o -spar_ordered
	# $(CXX) -c $(CF) $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -I ./ encoder_spar_cuda.alternative.cpp -o encoder_spar_cuda.o
	# # $(SPAR) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) -spar_file encoder_spar_cl.cpp -o encoder_spar_cl.o -spar_ordered
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) decoder.cpp -o decoder.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) mbuffer.cpp -o mbuffer.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) sha.cpp -o sha.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) queue.cpp -o queue.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES) binheap.cpp -o binheap.o
	$(CXX) -c $(CF) -DENABLE_SERIAL_GPU  -DCUDA_MALLOC $(CXXFLAGS) $(CFLAGS) $(LIBINCLUDES)  tree.cpp -o tree.o
	$(CXX) $(CF) $(CXXFLAGS) $(CFLAGS) -L$(CUDAPATH) -I$(SSL)  -I$(ZLIB) -L$(LZLIB) -L$(LSSL) -L$(LBZLIB)  -L$(LZSSLIB) -o dedup_cuda_spar_2xmemory encoder_spar_cuda.o gpu/sha1_gpu.o  gpu/sha1_gpu_cl.o gpu/memutils.o hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o queue.o binheap.o tree.o $(LZSSLIB)/gpu_util.o  $(LIBS) -lcudart -lOpenCL -lpthread

	rm -f *.o

lzsslib:
	make -C $(LZSSLIB)
	cp $(LZSSLIB)/*.cl ./
clean:
	rm -f *.o

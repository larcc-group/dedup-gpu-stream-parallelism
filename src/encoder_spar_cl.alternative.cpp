

#include <assert.h>
#include <strings.h>
#include <math.h>
#include <limits.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/stat.h>
#include <chrono>

#include "util.h"
#include "dedupdef.h"
#include "encoder.h"
#include "debug.h"
#include "hashtable.h"
#include "config.h"
#include "rabin.h"
#include "mbuffer.h"
#include "gpu_util.h"

// #ifdef ENABLE_GZIP_COMPRESSION
#include <zlib.h>
// #endif //ENABLE_GZIP_COMPRESSION

// #ifdef ENABLE_BZIP2_COMPRESSION
#include <bzlib.h>
// #endif //ENABLE_BZIP2_COMPRESSION

// #ifdef ENABLE_LZSS_COMPRESSION
#include <lzss.h>
// #endif //ENABLE_LZSS_COMPRESSION

// #ifdef ENABLE_PARSEC_HOOKS
// #include <hooks.h>
// #endif //ENABLE_PARSEC_HOOKS

// #ifdef ENABLE_SERIAL_GPU
#include "gpu/memutils.h"
#include "gpu/sha1_gpu.h"
// #endif

#define INITIAL_SEARCH_TREE_SIZE 4096

// #ifdef ENABLE_SERIAL_GPU
//The configuration block defined in main
extern config_t *conf;

//Hash table data structure & utility functions
extern struct hashtable *cache;

extern int rf_win;
extern int rf_win_dataprocess;
int deviceSize;
chunk_t *lastChunk = NULL;

void *SerialIntegratedPipelineGpuOpenCl(struct thread_args targs, int deviceSize);
void *SerialIntegratedPipelineGpuCl(struct thread_args targs)
{

  SerialIntegratedPipelineGpuOpenCl(targs, getDeviceIds().size());
}
















void *SerialIntegratedPipelineGpuOpenCl(struct thread_args targs, int deviceSize)
{
  // printf("Using %i devices\n",deviceSize);
  //   struct thread_args *args = (struct thread_args *)targs;

  Sha1OCLDeviceObjects** devices = new Sha1OCLDeviceObjects*[deviceSize];
  for(int i = 0; i < deviceSize; i++){
    OCLDeviceObjects* lzssDevice = createOCLDevice(getDeviceIds()[i]);
    Sha1OCLDeviceObjects* sha1Device = createSha1OCLDeviceObjects(lzssDevice);
    cl_int status;
    cl_command_queue queue = clCreateCommandQueue(lzssDevice->context, lzssDevice->device, 0, &status);
    OpenCLCheckError(status);
    sha1Device->queue = queue;
    devices[i] = sha1Device; 
  }
  struct thread_args *args = &targs;
  size_t preloading_buffer_seek = 0;
  int fd_out;
  int fd;
  fd = args->fd;
  fd_out = create_output_file(conf->outfile);
  int r;

  chunk_t *temp = NULL;
  chunk_t *chunk = NULL;
  lastChunk = NULL;
  u32int *rabintab = malloc(256 * sizeof rabintab[0]);
  u32int *rabinwintab = malloc(256 * sizeof rabintab[0]);
  if (rabintab == NULL || rabinwintab == NULL)
  {
    EXIT_TRACE("Memory allocation failed.\n");
  }

  rf_win_dataprocess = 0;
  rabininit(rf_win_dataprocess, rabintab, rabinwintab);

  //Sanity check
  if (MAXBUF < 8 * ANCHOR_JUMP)
  {
    printf("WARNING: I/O buffer size is very small. Performance degraded.\n");
    fflush(NULL);
  }

  int currentBatch = 0;
  std::vector<int> deviceIds = getDeviceIds();

  //read from input file / buffer
  [[ spar::ToStream(), spar::Input(temp, chunk, rabintab, rabinwintab, args, preloading_buffer_seek, r, fd, fd_out, currentBatch, deviceIds, deviceSize,devices) ]] 
  while (1)
  {
        size_t bytes_left; //amount of data left over in last_mbuffer from previous iteration

        //Check how much data left over from previous iteration resp. create an initial chunk
        if (temp != NULL)
        {
          bytes_left = temp->uncompressed_data.n;
        }
        else
        {
          bytes_left = 0;
        }

        //Make sure that system supports new buffer size
        if (MAXBUF + bytes_left > SSIZE_MAX)
        {
          EXIT_TRACE("Input buffer size exceeds system maximum.\n");
        }
        //Allocate a new chunk and create a new memory buffer
        chunk = (chunk_t *)malloc(sizeof(chunk_t));
        if (chunk == NULL)
          EXIT_TRACE("Memory allocation failed.\n");
        r = mbuffer_create(&chunk->uncompressed_data, MAXBUF + bytes_left);
        if (r != 0)
        {
          EXIT_TRACE("Unable to initialize memory buffer.\n");
        }
        chunk->header.state = CHUNK_STATE_UNCOMPRESSED;
        if (bytes_left > 0)
        {
          //FIXME: Short-circuit this if no more data available

          //"Extension" of existing buffer, copy sequence number and left over data to beginning of new buffer
          //NOTE: We cannot safely extend the current memory region because it has already been given to another thread
          memcpy(chunk->uncompressed_data.ptr, temp->uncompressed_data.ptr, temp->uncompressed_data.n);
          mbuffer_free(&temp->uncompressed_data);
          free(temp);
          temp = NULL;
        }
        //Read data until buffer full
        size_t bytes_read = 0;
        if (conf->preloading)
        {
          size_t max_read = MIN(MAXBUF, args->input_file.size - preloading_buffer_seek);
          memcpy(chunk->uncompressed_data.ptr + bytes_left, args->input_file.buffer + preloading_buffer_seek, max_read);
          bytes_read = max_read;
          preloading_buffer_seek += max_read;
        }
        else
        {
          while (bytes_read < MAXBUF)
          {
            r = read(fd, chunk->uncompressed_data.ptr + bytes_left + bytes_read, MAXBUF - bytes_read);
            if (r < 0)
              switch (errno)
              {
              case EAGAIN:
                EXIT_TRACE("I/O error: No data available\n");
                break;
              case EBADF:
                EXIT_TRACE("I/O error: Invalid file descriptor\n");
                break;
              case EFAULT:
                EXIT_TRACE("I/O error: Buffer out of range\n");
                break;
              case EINTR:
                EXIT_TRACE("I/O error: Interruption\n");
                break;
              case EINVAL:
                EXIT_TRACE("I/O error: Unable to read from file descriptor\n");
                break;
              case EIO:
                EXIT_TRACE("I/O error: Generic I/O error\n");
                break;
              case EISDIR:
                EXIT_TRACE("I/O error: Cannot read from a directory\n");
                break;
              default:
                EXIT_TRACE("I/O error: Unrecognized error\n");
                break;
              }
            if (r == 0)
              break;
            bytes_read += r;
          }
        }
        //No data left over from last iteration and also nothing new read in, simply clean up and quit
        if (bytes_left + bytes_read == 0)
        {
          mbuffer_free(&chunk->uncompressed_data);
          free(chunk);
          chunk = NULL;
          break;
        }
        //Shrink buffer to actual size
        if (bytes_left + bytes_read < chunk->uncompressed_data.n)
        {
          r = mbuffer_realloc(&chunk->uncompressed_data, bytes_left + bytes_read);
          assert(r == 0);
        }

        if (bytes_read == 0)
        {
          lastChunk = chunk;
          break;
        }
        //Memcpy to GPU will go here(async)
        //Generates the batch of offsets
        std::vector<int> lengths;
        std::vector<int> beginOffsets;

        std::vector<unsigned char *> dataCompressed;
        std::vector<std::vector<unsigned char>> allDataCompressed = std::vector<std::vector<unsigned char>>();
        
        std::vector<int> dataLength;
        currentBatch++;

          
        //Last chunk will generate only one item
        int maxLength = 0;
        int offset = 0;
        int increasedPointer = 0;
        while (true)
        {
          beginOffsets.push_back(increasedPointer);
          offset = rabinseg(chunk->uncompressed_data.ptr + increasedPointer, chunk->uncompressed_data.n - increasedPointer, rf_win_dataprocess, rabintab, rabinwintab);
          increasedPointer += offset;
          if (offset > maxLength)
          {
            maxLength = offset;
          }
          if (increasedPointer < chunk->uncompressed_data.n)
          {
            lengths.push_back(offset);
          }
          else
          {

            //treat left overs in next interaction
            int lastBeginOffset = beginOffsets[beginOffsets.size() - 1];
            int sizeTempChunk = chunk->uncompressed_data.n - lastBeginOffset;
            temp = (chunk_t *)malloc(sizeof(chunk_t));
            if (temp == NULL)
              EXIT_TRACE("Memory allocation failed.\n");
            r = mbuffer_create(&temp->uncompressed_data, sizeTempChunk);
            if (r != 0)
            {
              EXIT_TRACE("Unable to initialize temp memory buffer.\n");
            }
            memcpy(temp->uncompressed_data.ptr, chunk->uncompressed_data.ptr + lastBeginOffset, sizeTempChunk);
            break;
          }
        }

        assert(lengths.size() == beginOffsets.size() - 1);

        Sha1OCLDeviceObjects* currentDeviceOpenCl = devices[currentBatch % deviceIds.size()];
         cl_context context = currentDeviceOpenCl->device->context;

        cl_int status;
        cl_command_queue queue = currentDeviceOpenCl->queue;
        OpenCLCheckError(status);
        cl_int err;
        
        cl_mem uncompressed_data_gpu = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(unsigned char) * chunk->uncompressed_data.n, NULL, &err);
        OpenCLCheckError(err);
        OpenCLCheckError( clEnqueueWriteBuffer(queue, uncompressed_data_gpu, CL_FALSE, 0, sizeof(unsigned char) * chunk->uncompressed_data.n, chunk->uncompressed_data.ptr, 0, NULL, NULL) );

        cl_mem d_beginOffsets = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeof(int) * beginOffsets.size(), NULL, &err);
        OpenCLCheckError(err);
        OpenCLCheckError( clEnqueueWriteBuffer(queue, d_beginOffsets, CL_FALSE, 0, sizeof(int) * beginOffsets.size(), beginOffsets.data(), 0, NULL, NULL) );

        err = clFinish(queue);

        OpenCLCheckError(err );
        
        //Move offsets to GPU
        unsigned char *sha1s = new unsigned char[lengths.size() * SHA1_LEN]; //GPU friendly way of holding sha1
        bool *deduplicateStatus = new bool[lengths.size()];
        [[ spar::Stage(), spar::Input(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd, fd_out, maxLength,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl), spar::Output(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd, fd_out, maxLength,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl), spar::Replicate(deviceSize) ]] 
        {
          // auto beginOffsets1 = beginOffsets.data();
          auto lengths1data = lengths.data();
          auto lengths1size = lengths.size();
          sha1DigestBatchWrapperClWithDevice(uncompressed_data_gpu, lengths1size, d_beginOffsets, lengths1data, sha1s, currentDeviceOpenCl, queue);
        }

        [[ spar::Stage(), spar::Input(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd_out, maxLength, r, fd, fd_out,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl), spar::Output(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd, fd_out, maxLength,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl) ]]
        {
          for (int i = 0; i < lengths.size(); i++)
          {
            // printf("I %i \n",i);
            int beginOffset = (int)hashtable_search(cache, (void *)(sha1s + (i * SHA1_LEN)));
            bool isDeduplicate = beginOffset == 1;
            deduplicateStatus[i] = isDeduplicate;
            if (!isDeduplicate)
            {
              unsigned char *hash = new unsigned char[SHA1_LEN]; // = sha1s + (i * SHA1_LEN);
              memcpy(hash, sha1s + (i * SHA1_LEN), SHA1_LEN * sizeof(unsigned char));
              if (hashtable_insert(cache, (void *)(hash), 1) == 0)
              {
                EXIT_TRACE("hashtable_insert failed");
              }
            }
          }
        }
        [[ spar::Stage(), spar::Input(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd, fd_out, maxLength,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl), spar::Output(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd, fd_out, maxLength,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl), spar::Replicate(deviceSize) ]] 
        {
          //Check if is deduplicated
       
          LzssEncodeMemoryGpuBatch(chunk->uncompressed_data.ptr,uncompressed_data_gpu,chunk->uncompressed_data.n,beginOffsets.data(),d_beginOffsets,beginOffsets.size(),&allDataCompressed,currentDeviceOpenCl->device,queue);
        }

        [[ spar::Stage(), spar::Input(sha1s, uncompressed_data_gpu, lengths, beginOffsets, chunk, fd, fd_out, maxLength,  deduplicateStatus, dataCompressed, dataLength,allDataCompressed, d_beginOffsets, queue, currentDeviceOpenCl) ]]
        {
          for (int i = 0; i < lengths.size(); i++)
          {
            int r;
            bool isDeduplicate = deduplicateStatus[i];

            if (isDeduplicate){
              chunksFingerprint++;
              write_file(fd_out, (unsigned char)TYPE_FINGERPRINT, SHA1_LEN, (unsigned char *)(sha1s + i * SHA1_LEN));

            }
            else{
              r = write_file(fd_out, (unsigned char)TYPE_COMPRESS, allDataCompressed[i].size(), allDataCompressed[i].data());
              if (r != 0)
                EXIT_TRACE("Error in write_file %i", r);
            }
          }
          delete deduplicateStatus;
          delete sha1s;
          // delete compressedData;
          clReleaseMemObject(uncompressed_data_gpu);
          clReleaseMemObject(d_beginOffsets);
          // clReleaseCommandQueue(queue);
          mbuffer_free(&chunk->uncompressed_data);
        }

        
  }
  

  //Last chunk has only one item, do it on CPU
  if (lastChunk != NULL)
  {
    int isDuplicate = sub_Deduplicate(lastChunk);
    if (!isDuplicate)
    {
      sub_Compress(lastChunk);
    }
    write_chunk_to_file(fd_out, lastChunk);
  }
  free(rabintab);
  free(rabinwintab);

  close(fd_out);

  return NULL;
}

// #endif


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
 
#include <zlib.h>
 
#include <bzlib.h>
 
#include <lzss.h>
 
#include "gpu/memutils.h"
 
#include "gpu/sha1_gpu.h"
 
#define INITIAL_SEARCH_TREE_SIZE 4096
 
extern config_t * conf; 
extern struct hashtable * cache; 
extern int rf_win; 
extern int rf_win_dataprocess; 
int deviceSize; 
chunk_t * lastChunk = NULL; 
void * SerialIntegratedPipelineGpuCuda(struct thread_args targs,int deviceSize); 
void * SerialIntegratedPipelineGpu(struct thread_args targs) {
	SerialIntegratedPipelineGpuCuda(targs,getDeviceIds().size());
} 
#include <ff/pipeline.hpp>
 
#include <ff/farm.hpp>
 
using namespace ff; 
namespace spar{
	static inline ssize_t get_mac_core() {
		ssize_t n = 1; 
		FILE * f; 
		f = popen("cat /proc/cpuinfo |grep processor | wc -l","r"); 
		if(fscanf(f,"%ld",& n) == EOF)
		{
			pclose (f); 
			return n;
		} 
		pclose (f); 
		return n;
	} 
	static inline ssize_t get_env_num_workers() {
		ssize_t n = 1; 
		FILE * f; 
		f = popen("echo $SPAR_NUM_WORKERS","r"); 
		if(fscanf(f,"%ld",& n) == EOF)
		{
			pclose (f); 
			return n;
		} 
		pclose (f); 
		return n;
	} 
	static inline ssize_t get_Num_Workers() {
		ssize_t w_size = get_env_num_workers(); 
		if(w_size > 0)
		{
			return w_size;
		} 
		return get_mac_core();
	}
} 
struct _struct_spar0{
	_struct_spar0(unsigned char * sha1s,unsigned char * uncompressed_data_gpu,std::vector < int > lengths,std::vector < int > beginOffsets,chunk_t * chunk,int fd,int fd_out,int maxLength,int deviceIdThread,bool * deduplicateStatus,std::vector < unsigned char * > data,std::vector < int > dataLength) : sha1s(sha1s),uncompressed_data_gpu(uncompressed_data_gpu),lengths(lengths),beginOffsets(beginOffsets),chunk(chunk),fd(fd),fd_out(fd_out),maxLength(maxLength),deviceIdThread(deviceIdThread),deduplicateStatus(deduplicateStatus),data(data),dataLength(dataLength) {
	} 
	; 
	unsigned char * sha1s; 
	unsigned char * uncompressed_data_gpu; 
	std::vector < int > lengths; 
	std::vector < int > beginOffsets; 
	chunk_t * chunk; 
	int fd; 
	int fd_out; 
	int maxLength; 
	int deviceIdThread; 
	bool * deduplicateStatus; 
	std::vector < unsigned char * > data; 
	std::vector < int > dataLength; 
	int r;
}; 
_struct_spar0 * _Stage_spar00(_struct_spar0 * _Input_spar,ff_node *const) {
	{
		setDevice(_Input_spar -> deviceIdThread); 
		auto beginOffsets1 = _Input_spar -> beginOffsets._Input_spar -> data(); 
		auto lengths1data = _Input_spar -> lengths._Input_spar -> data(); 
		auto lengths1size = _Input_spar -> lengths.size(); 
		printf("a"); 
		sha1DigestBatchWrapper(_Input_spar -> uncompressed_data_gpu,lengths1size,beginOffsets1,lengths1data,_Input_spar -> sha1s);
	} 
	return _Input_spar;
} 
_struct_spar0 * _Stage_spar01(_struct_spar0 * _Input_spar,ff_node *const) {
	{
		
		for(int i = 0; i < _Input_spar -> lengths.size();i++)
		{
			int beginOffset = (int)hashtable_search(cache,(void *)(_Input_spar -> sha1s+(i*SHA1_LEN))); 
			bool isDeduplicate = beginOffset == 1; 
			_Input_spar -> deduplicateStatus[i] = isDeduplicate; 
			if(! isDeduplicate)
			{
				unsigned char * hash = new unsigned char [SHA1_LEN]; 
				memcpy(hash,_Input_spar -> sha1s+(i*SHA1_LEN),SHA1_LEN*sizeof(unsigned char)); 
				if(hashtable_insert(cache,(void *)(hash),1) == 0)
				{
					EXIT_TRACE("hashtable_insert failed");
				}
			}
		}
	} 
	return _Input_spar;
} 
_struct_spar0 * _Stage_spar02(_struct_spar0 * _Input_spar,ff_node *const) {
	{
		int n = _Input_spar -> maxLength+(_Input_spar -> maxLength / 8)+100; 
		int r; 
		unsigned char * compressedDataTmp = new unsigned char [n]; 
		
		for(int i = 0; i < _Input_spar -> lengths.size();i++)
		{
			bool isDeduplicate = _Input_spar -> deduplicateStatus[i]; 
			if(! isDeduplicate)
			{
				switch(conf -> compress_type)
				{
					case COMPRESS_LZSS : 
					int compressedSize; 
					if(_Input_spar -> lengths[i] < 1024)
					{
						r = LzssEncodeMemory((unsigned char *)(_Input_spar -> chunk -> uncompressed_data.ptr+_Input_spar -> beginOffsets[i]),_Input_spar -> lengths[i],compressedDataTmp,n,& compressedSize);
					} else 
					{
						r = LzssEncodeMemoryGpu((unsigned char *)(_Input_spar -> chunk -> uncompressed_data.ptr+_Input_spar -> beginOffsets[i]),_Input_spar -> uncompressed_data_gpu+_Input_spar -> beginOffsets[i],_Input_spar -> lengths[i],compressedDataTmp,n,& compressedSize,_Input_spar -> deviceIdThread);
					} 
					if(r != 0)
					{
						EXIT_TRACE("Error in lzss encoding %i",r);
					} 
					unsigned char * compressedData = new unsigned char [compressedSize]; 
					memcpy(compressedData,compressedDataTmp,compressedSize*sizeof(unsigned char)); 
					_Input_spar -> data.push_back(compressedData); 
					_Input_spar -> dataLength.push_back(compressedSize); 
					break; 
					case COMPRESS_NONE : 
					if(r != 0)
					{
						EXIT_TRACE("Error in write_file %i",r);
					} 
					_Input_spar -> data.push_back((unsigned char *)(_Input_spar -> chunk -> uncompressed_data.ptr+_Input_spar -> beginOffsets[i])); 
					_Input_spar -> dataLength.push_back(_Input_spar -> lengths[i]); 
					break; 
					default : 
					EXIT_TRACE("Compression  not supported on GPU"); 
					break;
				}
			}
		}
	} 
	return _Input_spar;
} 
_struct_spar0 * _Stage_spar03(_struct_spar0 * _Input_spar,ff_node *const) {
	{
		int currentDataIndex = 0; 
		
		for(int i = 0; i < _Input_spar -> lengths.size();i++)
		{
			bool isDeduplicate = _Input_spar -> deduplicateStatus[i]; 
			if(isDeduplicate)
			{
				printf("Printing fingerprint %i\n",i); 
				write_file(_Input_spar -> fd_out,(unsigned char)TYPE_FINGERPRINT,SHA1_LEN,(unsigned char *)(_Input_spar -> sha1s+i*SHA1_LEN));
			} else 
			{
				r = write_file(_Input_spar -> fd_out,(unsigned char)TYPE_COMPRESS,_Input_spar -> dataLength[currentDataIndex],_Input_spar -> data[currentDataIndex]); 
				if(r != 0)
				{
					EXIT_TRACE("Error in write_file %i",r);
				} 
				currentDataIndex++; 
				if(conf -> compress_type == COMPRESS_LZSS)
				{
					delete _Input_spar -> data[currentDataIndex];
				}
			}
		} 
		delete _Input_spar -> deduplicateStatus; 
		delete _Input_spar -> sha1s; 
		freeGpu(_Input_spar -> uncompressed_data_gpu); 
		mbuffer_free(& _Input_spar -> chunk -> uncompressed_data);
	} 
	delete _Input_spar; 
	return (_struct_spar0 *)GO_ON;
} 
struct _ToStream_spar0 : ff_node_t < _struct_spar0 >{
	chunk_t * temp; 
	chunk_t * chunk; 
	u32int * rabintab; 
	u32int * rabinwintab; 
	struct thread_args * args; 
	size_t preloading_buffer_seek; 
	int r; 
	int fd; 
	int fd_out; 
	int currentBatch; 
	std::vector < int > deviceIds; 
	int deviceSize; 
	_struct_spar0 * svc(_struct_spar0 * _Input_spar) {
		
		while(1)
		{
			size_t bytes_left; 
			if(temp != NULL)
			{
				bytes_left = temp -> uncompressed_data.n;
			} else 
			{
				bytes_left = 0;
			} 
			if(MAXBUF+bytes_left > SSIZE_MAX)
			{
				EXIT_TRACE("Input buffer size exceeds system maximum.\n");
			} 
			chunk = (chunk_t *)malloc(sizeof(chunk_t)); 
			if(chunk == NULL)
			EXIT_TRACE("Memory allocation failed.\n"); 
			r = mbuffer_create(& chunk -> uncompressed_data,MAXBUF+bytes_left); 
			if(r != 0)
			{
				EXIT_TRACE("Unable to initialize memory buffer.\n");
			} 
			chunk -> header.state = CHUNK_STATE_UNCOMPRESSED; 
			if(bytes_left > 0)
			{
				memcpy(chunk -> uncompressed_data.ptr,temp -> uncompressed_data.ptr,temp -> uncompressed_data.n); 
				mbuffer_free(& temp -> uncompressed_data); 
				free (temp); 
				temp = NULL;
			} 
			size_t bytes_read = 0; 
			if(conf -> preloading)
			{
				size_t max_read = MIN(MAXBUF,args -> input_file.size-preloading_buffer_seek); 
				memcpy(chunk -> uncompressed_data.ptr+bytes_left,args -> input_file.buffer+preloading_buffer_seek,max_read); 
				bytes_read = max_read; 
				preloading_buffer_seek += max_read;
			} else 
			{
				
				while(bytes_read < MAXBUF)
				{
					r = read(fd,chunk -> uncompressed_data.ptr+bytes_left+bytes_read,MAXBUF-bytes_read); 
					if(r < 0)
					switch(errno)
					{
						case EAGAIN : 
						EXIT_TRACE("I/O error: No data available\n"); 
						break; 
						case EBADF : 
						EXIT_TRACE("I/O error: Invalid file descriptor\n"); 
						break; 
						case EFAULT : 
						EXIT_TRACE("I/O error: Buffer out of range\n"); 
						break; 
						case EINTR : 
						EXIT_TRACE("I/O error: Interruption\n"); 
						break; 
						case EINVAL : 
						EXIT_TRACE("I/O error: Unable to read from file descriptor\n"); 
						break; 
						case EIO : 
						EXIT_TRACE("I/O error: Generic I/O error\n"); 
						break; 
						case EISDIR : 
						EXIT_TRACE("I/O error: Cannot read from a directory\n"); 
						break; 
						default : 
						EXIT_TRACE("I/O error: Unrecognized error\n"); 
						break;
					} 
					if(r == 0)
					break; 
					bytes_read += r;
				}
			} 
			if(bytes_left+bytes_read == 0)
			{
				mbuffer_free(& chunk -> uncompressed_data); 
				free (chunk); 
				chunk = NULL; 
				break;
			} 
			if(bytes_left+bytes_read < chunk -> uncompressed_data.n)
			{
				r = mbuffer_realloc(& chunk -> uncompressed_data,bytes_left+bytes_read); 
				assert(r == 0);
			} 
			if(bytes_read == 0)
			{
				lastChunk = chunk; 
				break;
			} 
			std::vector < int > lengths; 
			std::vector < int > beginOffsets; 
			std::vector < unsigned char * > data; 
			std::vector < int > dataLength; 
			int deviceIdThread = 0; 
			deviceIdThread = deviceIds[currentBatch%deviceIds.size()]; 
			currentBatch++; 
			unsigned char * uncompressed_data_gpu; 
			setDevice (deviceIdThread); 
			allocateGpu((void * *)& uncompressed_data_gpu,sizeof(unsigned char)*chunk -> uncompressed_data.n); 
			moveToGpu(uncompressed_data_gpu,chunk -> uncompressed_data.ptr,sizeof(unsigned char)*chunk -> uncompressed_data.n); 
			int maxLength = 0; 
			int offset = 0; 
			int increasedPointer = 0; 
			
			while(true)
			{
				beginOffsets.push_back(increasedPointer); 
				offset = rabinseg(chunk -> uncompressed_data.ptr+increasedPointer,chunk -> uncompressed_data.n-increasedPointer,rf_win_dataprocess,rabintab,rabinwintab); 
				increasedPointer += offset; 
				if(offset > maxLength)
				{
					maxLength = offset;
				} 
				if(increasedPointer < chunk -> uncompressed_data.n)
				{
					lengths.push_back(offset);
				} else 
				{
					int lastBeginOffset = beginOffsets[beginOffsets.size()-1]; 
					int sizeTempChunk = chunk -> uncompressed_data.n-lastBeginOffset; 
					temp = (chunk_t *)malloc(sizeof(chunk_t)); 
					if(temp == NULL)
					EXIT_TRACE("Memory allocation failed.\n"); 
					r = mbuffer_create(& temp -> uncompressed_data,sizeTempChunk); 
					if(r != 0)
					{
						EXIT_TRACE("Unable to initialize temp memory buffer.\n");
					} 
					memcpy(temp -> uncompressed_data.ptr,chunk -> uncompressed_data.ptr+lastBeginOffset,sizeTempChunk); 
					break;
				}
			} 
			assert(lengths.size() == beginOffsets.size()-1); 
			unsigned char * sha1s = new unsigned char [lengths.size()*SHA1_LEN]; 
			bool * deduplicateStatus = new bool [lengths.size()]; 
			_struct_spar0 * stream_spar = new _struct_spar0 (sha1s,uncompressed_data_gpu,lengths,beginOffsets,chunk,fd,fd_out,maxLength,deviceIdThread,deduplicateStatus,data,dataLength); 
			ff_send_out (stream_spar); 
			; 
			; 
			;
		} 
		return EOS;
	}
}; 
void * SerialIntegratedPipelineGpuCuda(struct thread_args targs,int deviceSize) {
	_ToStream_spar0 _ToStream_spar0_call; 
	ff_node_F < _struct_spar0 > _Stage_spar00_call (_Stage_spar00); 
	ff_node_F < _struct_spar0 > _Stage_spar01_call (_Stage_spar01); 
	ff_node_F < _struct_spar0 > _Stage_spar02_call (_Stage_spar02); 
	ff_node_F < _struct_spar0 > _Stage_spar03_call (_Stage_spar03); 
	ff_Pipe < _struct_spar0 > pipeline0(_ToStream_spar0_call,_Stage_spar00_call,_Stage_spar01_call,_Stage_spar02_call,_Stage_spar03_call); 
	struct thread_args * args = & targs; 
	size_t preloading_buffer_seek = 0; 
	int fd_out; 
	int fd; 
	fd = args -> fd; 
	fd_out = create_output_file(conf -> outfile); 
	int r; 
	chunk_t * temp = NULL; 
	chunk_t * chunk = NULL; 
	lastChunk = NULL; 
	u32int * rabintab = malloc(256*sizeof rabintab[0]); 
	u32int * rabinwintab = malloc(256*sizeof rabintab[0]); 
	if(rabintab == NULL || rabinwintab == NULL)
	{
		EXIT_TRACE("Memory allocation failed.\n");
	} 
	rf_win_dataprocess = 0; 
	rabininit(rf_win_dataprocess,rabintab,rabinwintab); 
	if(MAXBUF < 8*ANCHOR_JUMP)
	{
		printf("WARNING: I/O buffer size is very small. Performance degraded.\n"); 
		fflush (NULL);
	} 
	int currentBatch = 0; 
	std::vector < int > deviceIds = getDeviceIds(); 
	_ToStream_spar0_call.temp = temp; 
	_ToStream_spar0_call.chunk = chunk; 
	_ToStream_spar0_call.rabintab = rabintab; 
	_ToStream_spar0_call.rabinwintab = rabinwintab; 
	_ToStream_spar0_call.args = args; 
	_ToStream_spar0_call.preloading_buffer_seek = preloading_buffer_seek; 
	_ToStream_spar0_call.r = r; 
	_ToStream_spar0_call.fd = fd; 
	_ToStream_spar0_call.fd_out = fd_out; 
	_ToStream_spar0_call.currentBatch = currentBatch; 
	_ToStream_spar0_call.deviceIds = deviceIds; 
	_ToStream_spar0_call.deviceSize = deviceSize; 
	if(pipeline0.run_and_wait_end() < 0)
	{
		error("Running pipeline\n"); 
		exit(1);
	} 
	if(lastChunk != NULL)
	{
		int isDuplicate = sub_Deduplicate(lastChunk); 
		if(! isDuplicate)
		{
			sub_Compress (lastChunk);
		} 
		write_chunk_to_file(fd_out,lastChunk);
	} 
	free (rabintab); 
	free (rabinwintab); 
	close (fd_out); 
	return NULL;
}

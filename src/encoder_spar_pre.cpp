
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
chunk_t * lastChunk = NULL; 
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
struct struct_spar0{
	struct_spar0(unsigned char * sha1s,unsigned char * uncompressed_data_gpu,std::vector < int > lengths,std::vector < int > beginOffsets,chunk_t * chunk,int fd,int fd_out,int maxLength) : sha1s(sha1s),uncompressed_data_gpu(uncompressed_data_gpu),lengths(lengths),beginOffsets(beginOffsets),chunk(chunk),fd(fd),fd_out(fd_out),maxLength(maxLength) {
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
	int r;
}; 
struct_spar0 * Stage_spar00(struct_spar0 * Input_spar,ff_node *const) {
	{
		sha1DigestBatchWrapper(Input_spar -> uncompressed_data_gpu,Input_spar -> lengths.size(),Input_spar -> beginOffsets.data(),Input_spar -> lengths.data(),Input_spar -> sha1s);
	} 
	return Input_spar;
} 
struct_spar0 * Stage_spar01(struct_spar0 * Input_spar,ff_node *const) {
	{
		int n = Input_spar -> maxLength+(Input_spar -> maxLength / 8)+100; 
		unsigned char * compressedData = new unsigned char [n]; 
		
		for(int i = 0; i < Input_spar -> lengths.size();i++)
		{
			int beginOffset = (int)hashtable_search(cache,(void *)(Input_spar -> sha1s+(i*SHA1_LEN))); 
			bool isDeduplicate = beginOffset == 1; 
			if(isDeduplicate)
			{
				write_file(Input_spar -> fd_out,(unsigned char)TYPE_FINGERPRINT,SHA1_LEN,(unsigned char *)(Input_spar -> sha1s+i*SHA1_LEN));
			} else 
			{
				switch(conf -> compress_type)
				{
					case COMPRESS_LZSS : 
					int compressedSize; 
					if(Input_spar -> lengths[i] < 1024)
					{
						Input_spar -> r = LzssEncodeMemory((unsigned char *)(Input_spar -> chunk -> uncompressed_data.ptr+Input_spar -> beginOffsets[i]),Input_spar -> lengths[i],compressedData,n,& compressedSize);
					} else 
					{
						Input_spar -> r = LzssEncodeMemoryGpu((unsigned char *)(Input_spar -> chunk -> uncompressed_data.ptr+Input_spar -> beginOffsets[i]),Input_spar -> uncompressed_data_gpu+Input_spar -> beginOffsets[i],Input_spar -> lengths[i],compressedData,n,& compressedSize);
					} 
					if(Input_spar -> r != 0)
					{
						EXIT_TRACE("Error in lzss encoding %i",Input_spar -> r);
					} 
					Input_spar -> r = write_file(Input_spar -> fd_out,(unsigned char)TYPE_COMPRESS,compressedSize,compressedData); 
					if(Input_spar -> r != 0)
					{
						EXIT_TRACE("Error in write_file %i",Input_spar -> r);
					} 
					break; 
					case COMPRESS_NONE : 
					Input_spar -> r = write_file(Input_spar -> fd_out,(unsigned char)TYPE_COMPRESS,Input_spar -> lengths[i],(unsigned char *)(Input_spar -> chunk -> uncompressed_data.ptr+Input_spar -> beginOffsets[i])); 
					if(Input_spar -> r != 0)
					{
						EXIT_TRACE("Error in write_file %i",Input_spar -> r);
					} 
					break; 
					default : 
					EXIT_TRACE("Compression  not supported on GPU"); 
					break;
				} 
				unsigned char * hash = new unsigned char [SHA1_LEN]; 
				memcpy(hash,Input_spar -> sha1s+(i*SHA1_LEN),SHA1_LEN*sizeof(unsigned char)); 
				if(hashtable_insert(cache,(void *)(hash),1) == 0)
				{
					EXIT_TRACE("hashtable_insert failed");
				}
			}
		} 
		delete Input_spar -> sha1s; 
		delete compressedData; 
		freeGpu(Input_spar -> uncompressed_data_gpu); 
		mbuffer_free(& Input_spar -> chunk -> uncompressed_data);
	} 
	delete Input_spar; 
	return (struct_spar0 *)GO_ON;
} 
struct ToStream_spar0 : ff_node_t < struct_spar0 >{
	chunk_t * temp; 
	chunk_t * chunk; 
	u32int * rabintab; 
	u32int * rabinwintab; 
	struct thread_args * args; 
	size_t preloading_buffer_seek; 
	int r; 
	int fd; 
	int fd_out; 
	struct_spar0 * svc(struct_spar0 * Input_spar) {
		
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
			int split; 
			std::vector < int > lengths; 
			std::vector < int > beginOffsets; 
			unsigned char * uncompressed_data_gpu; 
			allocateGpu((void * *)& uncompressed_data_gpu,sizeof(unsigned char)*chunk -> uncompressed_data.n); 
			moveToGpu(uncompressed_data_gpu,chunk -> uncompressed_data.ptr,sizeof(unsigned char)*chunk -> uncompressed_data.n); 
			int maxLength = 0; 
			if(bytes_read == 0)
			{
				lastChunk = chunk; 
				break;
			} else 
			{
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
			} 
			unsigned char * sha1s = new unsigned char [lengths.size()*SHA1_LEN]; 
			struct_spar0 * stream_spar = new struct_spar0 (sha1s,uncompressed_data_gpu,lengths,beginOffsets,chunk,fd,fd_out,maxLength); 
			ff_send_out (stream_spar); 
			;
		} 
		return EOS;
	}
}; 
void * SerialIntegratedPipelineGpu(struct thread_args targs) {
	ToStream_spar0 ToStream_spar0_call; 
	ff_node_F < struct_spar0 > Stage_spar00_call (Stage_spar00); 
	ff_node_F < struct_spar0 > Stage_spar01_call (Stage_spar01); 
	ff_Pipe < struct_spar0 > pipeline0(ToStream_spar0_call,Stage_spar00_call,Stage_spar01_call); 
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
	ToStream_spar0_call.temp = temp; 
	ToStream_spar0_call.chunk = chunk; 
	ToStream_spar0_call.rabintab = rabintab; 
	ToStream_spar0_call.rabinwintab = rabinwintab; 
	ToStream_spar0_call.args = args; 
	ToStream_spar0_call.preloading_buffer_seek = preloading_buffer_seek; 
	ToStream_spar0_call.r = r; 
	ToStream_spar0_call.fd = fd; 
	ToStream_spar0_call.fd_out = fd_out; 
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

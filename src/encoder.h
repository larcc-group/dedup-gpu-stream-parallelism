#ifndef _ENCODER_H_
#define _ENCODER_H_ 1


//Arguments to pass to each thread
struct thread_args {
  //thread id, unique within a thread pool (i.e. unique for a pipeline stage)
  int tid;
  //number of queues available, first and last pipeline stage only
  int nqueues;
  //file descriptor, first pipeline stage only
  int fd;
  //input file buffer, first pipeline stage & preloading only
  struct {
    void *buffer;
    size_t size;
  } input_file;
};


void Encode(config_t * conf);
#ifdef ENABLE_SPAR
void EncodeSPar(config_t * _conf);
#endif

#ifndef ENABLE_SPAR

#ifdef ENABLE_SERIAL_GPU
void *SerialIntegratedPipelineGpu(struct thread_args targs) ;
void *SerialIntegratedPipelineGpuCl(struct thread_args targs) ;
#endif
int create_output_file(char *outfile) ;
int sub_Deduplicate(chunk_t *chunk) ;


#ifdef ENABLE_STATISTICS
//Keep track of block granularity with 2^CHUNK_GRANULARITY_POW resolution (for statistics)
#define CHUNK_GRANULARITY_POW (7)
//Number of blocks to distinguish, CHUNK_MAX_NUM * 2^CHUNK_GRANULARITY_POW is biggest block being recognized (for statistics)
#define CHUNK_MAX_NUM (8*32)
//Map a chunk size to a statistics array slot
#define CHUNK_SIZE_TO_SLOT(s) ( ((s)>>(CHUNK_GRANULARITY_POW)) >= (CHUNK_MAX_NUM) ? (CHUNK_MAX_NUM)-1 : ((s)>>(CHUNK_GRANULARITY_POW)) )
//Get the average size of a chunk from a statistics array slot
#define SLOT_TO_CHUNK_SIZE(s) ( (s)*(1<<(CHUNK_GRANULARITY_POW)) + (1<<((CHUNK_GRANULARITY_POW)-1)) )
//Deduplication statistics (only used if ENABLE_STATISTICS is defined)
typedef struct {
  /* Cumulative sizes */
  size_t total_input; //Total size of input in bytes
  size_t total_dedup; //Total size of input without duplicate blocks (after global compression) in bytes
  size_t total_compressed; //Total size of input stream after local compression in bytes
  size_t total_output; //Total size of output in bytes (with overhead) in bytes

  /* Size distribution & other properties */
  unsigned int nChunks[CHUNK_MAX_NUM]; //Coarse-granular size distribution of data chunks
  unsigned int nDuplicates; //Total number of duplicate blocks
} stats_t;

void merge_stats(stats_t *s1, stats_t *s2);
int write_file(int fd, u_char type, u_long len, u_char * content) ;
void print_stats(stats_t *s) ;
extern stats_t stats;
#endif

void write_chunk_to_file(int fd, chunk_t *chunk) ;
void sub_Compress(chunk_t *chunk);

#endif
#endif /* !_ENCODER_H_ */

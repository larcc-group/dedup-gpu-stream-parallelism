#include <unistd.h>
#include <string.h>

#include "util.h"
#include "debug.h"
#include "dedupdef.h"
#include "encoder.h"
#include "decoder.h"
#include "config.h"
#include "queue.h"

#include "gpu_util.h"
#include <iostream>
#include <sstream>

#ifdef ENABLE_DMALLOC
#include <dmalloc.h>
#endif //ENABLE_DMALLOC

#ifdef ENABLE_PTHREADS
#include <pthread.h>
#endif //ENABLE_PTHREADS

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif //ENABLE_PARSEC_HOOKS


extern config_t * conf;



/*--------------------------------------------------------------------------*/
static void
usage(char* prog)
{
  printf("usage: %s [-cusfvh] [-w gzip/bzip2/lzss/none] [-i file] [-o file] [-t number_of_threads]\n",prog);
  printf("-c \t\t\tcompress\n");
  printf("-u \t\t\tuncompress\n");
  printf("-p \t\t\tpreloading (for benchmarking purposes)\n");
  printf("-w \t\t\tcompression type: gzip/bzip2/none\n");
  printf("-i file\t\t\tthe input file\n");
  printf("-o file\t\t\tthe output file\n");
  printf("-t \t\t\tnumber of threads per stage \n");
  printf("-v \t\t\tverbose output\n");
  printf("-h \t\t\thelp\n");
}
/*--------------------------------------------------------------------------*/
int main(int argc, char** argv) {
#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
  printf("PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
#else
  printf("PARSEC Benchmark Suite\n");
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
        __parsec_bench_begin(__parsec_dedup);
#endif //ENABLE_PARSEC_HOOKS
    std::vector<int> deviceIds = {};

  int32 compress = TRUE;

  //We force the sha1 sum to be integer-aligned, check that the length of a sha1 sum is a multiple of unsigned int
  assert(SHA1_LEN % sizeof(unsigned int) == 0);

  conf = (config_t *) malloc(sizeof(config_t));
  if (conf == NULL) {
    EXIT_TRACE("Memory allocation failed\n");
  }

  strcpy(conf->outfile, "");
  conf->compress_type = COMPRESS_GZIP;
  conf->preloading = 0;
  conf->nthreads = 1;
  conf->verbose = 0;

  //parse the args
  int ch;
  opterr = 0;
  optind = 1;
  while (-1 != (ch = getopt(argc, argv, "cupvo:i:w:t:hg:"))) {
    
    switch (ch) {
    case 'c':
      compress = TRUE;
      strcpy(conf->infile, "/home/charles/Downloads/parsec-3.0/pkgs/kernels/dedup/inputs/input_simsmall.tar");
      strcpy(conf->outfile, "/home/charles/Downloads/parsec-3.0/pkgs/kernels/dedup/inputs/input_simsmall.tar.ddp");
      break;
    case 'u':
      compress = FALSE;
      strcpy(conf->infile, "/home/charles/projects/stream_applications/lzss/data/silesia.tar.ddp");
      strcpy(conf->outfile, "/home/charles/projects/stream_applications/lzss/data/silesia.tar2");
      break;
    case 'g':
    {

      std::istringstream is(optarg);

      std::string number_as_string;
      while (std::getline(is, number_as_string, ','))
      {
          deviceIds.push_back(std::stoi(number_as_string));
      }
    }
    break;
    case 'w':
	
      if (strcmp(optarg, "gzip") == 0)
        conf->compress_type = COMPRESS_GZIP;
      else if (strcmp(optarg, "bzip2") == 0) 
        conf->compress_type = COMPRESS_BZIP2;
      else if (strcmp(optarg, "lzss") == 0) 
        conf->compress_type = COMPRESS_LZSS;
      else if (strcmp(optarg, "none") == 0)
        conf->compress_type = COMPRESS_NONE;
      else {
        fprintf(stdout, "Unknown compression type `%s'.\n", optarg);
        usage(argv[0]);
        return -1;
      }
      break;
    case 'o':
      strcpy(conf->outfile, optarg);
      break;
    case 'i':
      strcpy(conf->infile, optarg);
      break;
    case 'h':
      usage(argv[0]);
      return -1;
    case 'p':
      conf->preloading = TRUE;
      break;
    case 't':
      conf->nthreads = atoi(optarg);
      break;
    case 'v':
      conf->verbose = TRUE;
      break;
    case '?':
      fprintf(stdout, "Unknown option `-%c'.\n", optopt);
      usage(argv[0]);
      return -1;
    }
  }
  if(deviceIds.size() == 0){
    deviceIds.push_back(0);
  }
  
    setDeviceIds(deviceIds);
#ifndef ENABLE_BZIP2_COMPRESSION
 if (conf->compress_type == COMPRESS_BZIP2){
    printf("Bzip2 compression not supported\n");
    exit(1);
  }
#endif

#ifndef ENABLE_GZIP_COMPRESSION
 if (conf->compress_type == COMPRESS_GZIP){
    printf("Gzip compression not supported\n");
    exit(1);
  }
#endif


#ifndef ENABLE_LZSS_COMPRESSION
 if (conf->compress_type == COMPRESS_LZSS){
    printf("Lzss compression not supported\n");
    exit(1);
  }
#endif

#ifndef ENABLE_STATISTICS
 if (conf->verbose){
    printf("Statistics collection not supported\n");
    exit(1);
  }
#endif

#if !defined(ENABLE_PTHREADS) && !defined(ENABLE_SPAR)
 if (conf->nthreads != 1){
    printf("Number of threads must be 1 (serial version)\n");
    exit(1);
  }
#endif


  if (compress) {
    #ifdef ENABLE_SPAR
      EncodeSPar(conf);
    #else
      Encode(conf);
    #endif
  } else {
    Decode(conf);
  }

  free(conf);

#ifdef ENABLE_PARSEC_HOOKS
  __parsec_bench_end();
#endif

  return 0;
}


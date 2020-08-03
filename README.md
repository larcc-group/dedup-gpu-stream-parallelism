This repository has the parallel and sequential implementations for PARSEC's Dedup benchmark. We introduced stream parallelism for CPU (using SPar) and GPU (using CUDA and OpenCL).

## Citing Dedup on GPU
You can use this code as long as you cite our work in case you publish something related to it.

[DOI](https://doi.org/10.1109/IPDPSW.2019.00137) Rockenbach, Dinei A; Stein, Charles M; Griebler, Dalvan; Mencagli, Gabriele; Torquati, Massimo; Danelutto, Marco; Fernandes, Luiz Gustavo.
**Stream Processing on Multi-Cores with GPUs: Parallel Programming Models` Challenges**. *IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, IEEE, Rio de Janeiro, Brazil, 2019.

```bibtex
@inproceedings{larcc:stream_processing_gpu_challenges:IPDPSW:19,
    title = {Stream Processing on Multi-Cores with GPUs: Parallel Programming Models' Challenges},
    author = {Dinei A Rockenbach and Charles M Stein and Dalvan Griebler and Gabriele Mencagli and Massimo Torquati and Marco Danelutto and Luiz Gustavo Fernandes},
    year = {2019},
    date = {2019-05-01},
    booktitle = {IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
    publisher = {IEEE},
    address = {Rio de Janeiro, Brazil},
    keywords = {},
    pubstate = {forthcoming},
    tppubtype = {inproceedings}
}

```


## Compiling
To compile the project, you need to have the following prerequisites installed.
 - CUDA
 - SPar in your home (https://github.com/dalvangriebler/SPar)

Once you have all the prerequisites installed, run the following command inside *src* folder:

```make```

This will generate the *dedup* runnable files. PS: You wont be able to run it on on Windows, because SPar at the time we are publishing it can only be ran on linux.

The following versions are available on this library:
 - CUDA
 - CUDA + SPar
 - OpenCL
 - OpenCL + SPar
 - Sequential
 - CUDA  + SPar (lzss optimization)
 - OpenCL  + SPar (lzss optimization)
 - CUDA  + SPar (2x memory optimization)
 - OpenCL  + SPar (2x memory optimization)
## Using Dedup on GPU

### Compression
```
    ./src/dedup_{version} -c -i <input_file_name> -o <output_file_name> -w lzss
```
To use multiple GPU, pass in the argument *-g*, indicating the GPU ids separated by comma
```
    ./src/dedup_{version} -c -i <input_file_name> -o <output_file_name> -w lzss -g 0,1
```
For benchmarking purposes, you can pass the *-m* argument to run the test in memory
### Decompression
The decompression is only available on a serial implementation, thus you only need to provide the input file  path and the output path by using this command:
```
    ./src/dedup_sequential -u -i <input_file_name> -o <output_file_name>
```

## Downloading datasets
The datasets used in our work can be downloaded by executing this command:
```
cd data
./download_benchmark.sh
```
## Running benchmarks
To run benchmarks run the following commands:
```
cd benchmark
./all_dedup_datasets.sh
```

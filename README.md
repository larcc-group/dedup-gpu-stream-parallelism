This repository has the parallel and sequential implementations for PARSEC's Dedup benchmark. We introduced stream parallelism for CPU (using SPar) and GPU (using CUDA and OpenCL).

## Citing Dedup on GPU
You can use this code as long as you cite our work in case you publish something related to it.

[DOI](https://doi.org/10.1109/IPDPSW.2019.00137) Rockenbach, Dinei A; Stein, Charles M; Griebler, Dalvan; Mencagli, Gabriele; Torquati, Massimo; Danelutto, Marco; Fernandes, Luiz Gustavo.
**Stream Processing on Multi-Cores with GPUs: Parallel Programming Models` Challenges**. *IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, IEEE, Rio de Janeiro, Brazil, 2019.

```bibtex
@inproceedings{ROCKENBACH:stream-multigpus:IPDPSW:19,
	author={Dinei A. Rockenbach and Charles Michael Stein and Dalvan Griebler and Gabriele Mencagli and Massimo Torquati and Marco Danelutto and Luiz Gustavo Fernandes},
	title={{Stream Processing on Multi-cores with GPUs: Parallel Programming Models' Challenges}},
	booktitle={International Parallel and Distributed Processing Symposium Workshops (IPDPSW)},
	series={IPDPSW'19},	
	pages={834-841},
	publisher={IEEE},	
	address={Rio de Janeiro, Brazil},
	month={May},
	year={2019},
	doi={10.1109/IPDPSW.2019.00137},
	url={https://doi.org/10.1109/IPDPSW.2019.00137},
	abstract={The stream processing paradigm is used in several scientific and enterprise applications in order to continuously compute results out of data items coming from data sources such as sensors. The full exploitation of the potential parallelism offered by current heterogeneous multi-cores equipped with one or more GPUs is still a challenge in the context of stream processing applications. In this work, our main goal is to present the parallel programming challenges that the programmer has to face when exploiting CPUs and GPUs' parallelism at the same time using traditional programming models. We highlight the parallelization methodology in two use-cases (the Mandelbrot Streaming benchmark and the PARSEC's Dedup application) to demonstrate the issues and benefits of using heterogeneous parallel hardware. The experiments conducted demonstrate how a high-level parallel programming model targeting stream processing like the one offered by SPar can be used to reduce the programming effort still offering a good level of performance if compared with state-of-the-art programming models.},
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

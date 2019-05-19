#!/bin/bash
sleep 3 
for (( i = 0; i < 10; i++ )); do
	#parsecmgmt -a run -p ferret -i native -n 4
	./amd64-linux.gcc-pthreads/bin/dedup -c -v -t 2 -i ../inputs/input_native.tar -o native.ddp >> bensp.log
	./amd64-linux.gcc-pthreads/bin/dedup -c -v -t 2 -i /input2_native.tar -o native2.ddp >> bensp2.log
	rm *.ddp
done

#rm -f /home/carlos/source_benchmarks/parsec-3.0/pkgs/kernels/dedup/src/rabin.h
#mv /home/carlos/source_benchmarks/parsec-3.0/pkgs/kernels/dedup/src/1rabin.h /home/carlos/source_benchmarks/parsec-3.0/pkgs/kernels/dedup/src/rabin.h


#	CHECK_DIR=$(ls "amd64-linux.gcc-pthreads")
	#RETURN=$(echo $?)
	#if [[ $RETURN -eq 0 ]]; then
		#echo APAGANDO
	#	rm -rf amd64-linux.gcc-pthreads
	#fi

	#parsecmgmt -a build -p dedup -c gcc-pthreads

#for (( i = 0; i < 10; i++ )); do
#	parsecmgmt -a run -p ferret -i native -n 4
#	./amd64-linux.gcc-pthreads/bin/dedup -c -v -t 4 -i ../inputs/input_native.tar -o native.ddp >> bensp_custom.log
#done
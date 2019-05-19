#!/bin/bash
if [[ $1 == 'pt' ]]; then
	CHECK_DIR=$(ls "amd64-linux.gcc-pthreads")
	RETURN=$(echo $?)
	if [[ $RETURN -eq 0 ]]; then
		#echo APAGANDO
		rm -rf amd64-linux.gcc-pthreads
	fi
	parsecmgmt -a build -p dedup -c gcc-pthreads
fi

if [[ $1 == 'seq' ]]; then
	echo "install serial"
fi
#!/bin/sh

REPS=5
PARSECINPUT="../data/input_simlarge.tar"
LINUX="../data/linux-4.20.tar"
SILESIA="../data/silesia.tar"
NOW=`date`
eval "mv benchmark.db \"benchmark_${NOW}.db\""
eval "mv benchmark.log \"benchmark_${NOW}.log\""

eval "python3 benchmark.py $PARSECINPUT $REPS"
eval "python3 benchmark.py $LINUX $REPS"
eval "python3 benchmark.py $SILESIA $REPS"

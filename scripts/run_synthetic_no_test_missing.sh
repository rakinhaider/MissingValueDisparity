#!/bin/bash

dt=$1 # Use corr
deltas=$2 # Use 0, -3 or if necessary 3

dir="outputs/synthetic/tables/"
mkdir -p ${dir}
file=${dir}/${dt}_${deltas}.tsv
echo -n "" >${file}
for method in baseline drop mean mice knn softimpute; do
	python -m experiment_synthetic -dt ${dt}\
	  --method ${method} -gs ${deltas} >>${file};
done

#!/bin/bash

dt=corr # Use corr
deltas=-3 # Use 0, -3 or if necessary 3

dir="outputs/synthetic/tables/"
mkdir -p ${dir}
file=${dir}/${dt}_${deltas}_tm.tsv
echo -n "" >${file}
for method in baseline drop mean mice knn softimpute; do
	python -m experiment_synthetic -dt ${dt}\
	  --method ${method} -gs ${deltas} -tm train >>${file};
done
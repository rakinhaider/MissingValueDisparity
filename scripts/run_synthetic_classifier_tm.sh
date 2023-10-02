#!/bin/bash


echo -n -e "\\multirow{6}{*}{0}"
for method in baseline drop mean mice knn softimpute; do
	echo -n -e '\t'${method}'\t'
	python -m experiment_synthetic -dt corr --method ${method}\
	  -gs -3 -tm train 2>/dev/null
done
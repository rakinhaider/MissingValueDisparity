#!/bin/bash


echo -n -e "\\multirow{6}{*}{0}"
for method in baseline drop mean mice knn softimpute; do
	echo -n -e '\t'${method}'\t'
	python -m experiment_synthetic -dt corr --method ${method} 2>/dev/null
done

echo -n -e "\\multirow{6}{*}{-3}"
for method in baseline drop mean mice knn softimpute; do
	echo -n -e '\t'${method}'\t'
	python -m experiment_synthetic -dt corr --method ${method} -gs -3 2>/dev/null
done
#!/bin/bash

dataset=$1
test_method=$2
estimator='nb'
strategy='3'

methods=('baseline' 'drop' 'simple_imputer.mean'
'iterative_imputer.mice'
'knn_imputer')
is_reduce=0
estimators=('nb' 'lr' 'pr')

for estimator in ${estimators[@]}; do
	fname=${dataset}_${test_method}_${estimator}_${strategy}_${is_reduce}.tsv
	python -m experiment_standard_dataset --header-only >outputs/standard/${dataset}/$fname;

	reduce_cmd='';
	if [ ${is_reduce} == 1 ]; then
		reduce_cmd='--reduce'
	fi

	for m in ${methods[@]}; do
		echo ${estimator} ${m};
		python -m experiment_standard_dataset\
			  --dataset ${dataset} --method ${m}\
			  --estimator ${estimator} ${reduce} --strategy ${strategy}\
			  --priv-ic-prob 0.1 --unpriv-ic-prob 0.4\
			  >>outputs/standard/${dataset}/$fname;
	done
	printf "\\midrule\n" >>outputs/standard/${dataset}/$fname;
done
#!/bin/bash

type=$1
test_method=$2

methods=('baseline' 'drop' 'simple_imputer.mean'
'iterative_imputer.mice'
'iterative_imputer.missForest'
'knn_imputer'
'group_imputer')

for is_kip in ''; do
	echo "Keep protected param:" $is_kip;
	if [[ "$is_kip" == '' ]]
	then
		fname="${type}_${test_method}_nop.tsv";
	else
		fname="${type}_${test_method}_kip.tsv";
	fi
	python -m experiment_synthetic --header-only >outputs/synthetic/$fname;
	for alpha in 0.25 0.5 0.75; do
		for m in ${methods[@]}; do
			python -m experiment_synthetic\
			--alpha $alpha --distype $type\
			--method $m $is_kip -tm ${test_method}\
			"${@:3:99}" >>outputs/synthetic/$fname;
		done
		printf "\\midrule\n" >>outputs/synthetic/$fname;
	done
done



#!/bin/bash

type=$1
test_method=$2

methods=('baseline' 'drop' 'simple_imputer.mean'
'iterative_imputer.mice'
'knn_imputer')

if [ $1 == "syn" ]; then
	python -m experiment_synthetic --header-only;
	for alpha in 0.5; do
		for upic in 0.1 0.2 0.3 0.4 0.5 0.6; do
			echo "upic" ${upic};
			python -m experiment_synthetic\
			--alpha $alpha --distype corr\
			--delta 10 -gs -3 -pic 0.1 -upic ${upic}\
			--method "knn_imputer" -tm none;
		done
	done
	exit
elif [ $1 == "std" ]; then
	python -m experiment_standard_dataset --header-only;
	for dataset in compas; do
		python -m experiment_standard_dataset\
		--dataset ${dataset}\
		-pic 0 -upic 0\
		--method "simple_imputer.mean";

		for upic in 0.1 0.2 0.3 0.4 0.5 0.6; do
			echo "upic" ${upic};
			python -m experiment_standard_dataset\
			--dataset ${dataset}\
			-pic 0.1 -upic ${upic}\
			--method "simple_imputer.mean";
		done
	done
	exit
fi

for is_kip in ''; do
	echo "Keep protected param:" $is_kip;
	for gs in -3 3; do
		if [[ "$is_kip" == '' ]]
		then
			fname="${type}_${test_method}_nop_${gs}.tsv";
		else
			fname="${type}_${test_method}_kip_${gs}.tsv";
		fi
		python -m experiment_synthetic --header-only >outputs/synthetic/$fname;
		for alpha in 0.25 0.5 0.75; do
			for m in ${methods[@]}; do
				python -m experiment_synthetic\
				--alpha $alpha --distype $type\
				--method $m $is_kip -tm ${test_method} -gs ${gs}\
				>>outputs/synthetic/$fname;
			done
		done
		printf "\\midrule\n" >>outputs/synthetic/$fname;
	done
done



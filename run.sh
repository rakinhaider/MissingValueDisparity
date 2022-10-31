#!/bin/bash

methods=('baseline' 'drop' 'simple_imputer.mean'
'iterative_imputer.mice'
'iterative_imputer.missForest'
'knn_imputer'
'group_imputer')

for is_kip in '' '-kip'; do
	echo "Keep protected param:" $is_kip;
	for type in ds_ccd ccd; do
		if [[ "$is_kip" == '' ]]
		then
			fname="${type}_nop.tsv";
		else
			fname="${type}_kip.tsv";
		fi
		python -m experiment --header-only >outputs/$fname;
		for m in ${methods[@]}; do
			for alpha in 0.25 0.5 0.75; do
				python -m experiment\
				--alpha $alpha --distype $type\
				 --method $m $is_kip >>outputs/$fname;
			done
		done
	done
done



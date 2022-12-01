#!/bin/bash

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
fi



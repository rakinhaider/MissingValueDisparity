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
	for dataset in compas; do
		estimator='lr'
		# calibration="-c isotonic -ccv 10"
		calibration=""
		strategy=3
		file_name=${dataset}_by_upic.tsv
		python -m experiment_standard_dataset --header-only >${file_name};
		python -m experiment_standard_dataset\
		--dataset ${dataset}\
		-pic 0 -upic 0 -e ${estimator} ${calibration} -s ${strategy}\
		--method "simple_imputer.mode" >>${file_name};

		for upic in 0.1 0.2 0.3 0.4 0.5 0.6; do
			echo "upic" ${upic};
			python -m experiment_standard_dataset\
			--dataset ${dataset}\
			-pic 0.1 -upic ${upic} -e ${estimator} ${calibration}\
			-s ${strategy} --method "simple_imputer.mode"\
			-ll DEBUG >>${file_name};
		done
	done
fi



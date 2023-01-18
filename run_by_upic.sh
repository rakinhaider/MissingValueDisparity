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
	for dataset in pima; do
		estimator='nb'
		# calibration="-c isotonic -ccv 10"
		calibration=""
		reduce=""
		strategy=3
		method="simple_imputer.mean"
		file_name=${dataset}_by_upic.tsv
		python -m experiment_standard_dataset --header-only >${file_name};
		python -m experiment_standard_dataset\
		--dataset ${dataset}\
		-pic 0 -upic 0 -e ${estimator} ${calibration} -s ${strategy}\
		--method ${method} ${reduce} >>${file_name};

		for pic in 0.1 0.2 0.3 0.4 0.5 0.6; do
			echo "pic" ${pic};
			python -m experiment_standard_dataset\
			--dataset ${dataset}\
			-upic 0 -pic ${pic} -e ${estimator} ${calibration}\
			-s ${strategy} --method ${method} ${reduce}\
			-ll DEBUG >>${file_name};
		done
	done
fi
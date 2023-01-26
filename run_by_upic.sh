#!/bin/bash

methods=('simple_imputer.mean' 'iterative_imputer.mice' 'knn_imputer')
estimators=('nb' 'lr' 'dt' 'pr')
datasets=('compas' 'bank')
xvalid=''
logging='-ll DEBUG'

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
	methods=('simple_imputer.mean')
	estimators=('nb' 'lr' 'dt' 'pr')
	datasets=('pima')
	xvalid='-x'
	logging='-ll DEBUG'
	for dataset in ${datasets[@]}; do
		for estimator in ${estimators[@]}; do
			for is_cali in 0 1; do
				calibration=""
				if [ ${is_cali} == 1 ]
				then
					calibration="-c isotonic -ccv 10"
				fi
				for strategy in 0 2 3; do
					xval_dir=''
					if [ ${xvalid} == '-x' ]
					then
						xval_dir='xval/'
						is_xval=1
					fi
					dir=outputs/standard/${dataset}/${xval_dir}results/${estimator}/
					echo ${dir};
					mkdir -p $dir;
					file_name=${dataset}_${estimator}_${is_cali}_${strategy}.tsv
					echo ${file_name};
					python -m experiment_standard_dataset --header-only >${dir}/${file_name};
					for method in ${methods[@]}; do
						reduce=''
						if [ $estimator = 'reduce' ]
						then
							estimator='lr'
							reduce='--reduce'
						fi
						python -m experiment_standard_dataset\
						--dataset ${dataset}\
						-pic 0 -upic 0 -e ${estimator} ${calibration}\
						-s ${strategy} --method ${method} ${reduce} \
						${xvalid} >>${dir}/${file_name};

						for upic in 0.1 0.2 0.3 0.4 0.5 0.6; do
							echo "upic" ${upic};
							python -m experiment_standard_dataset\
							--dataset ${dataset}\
							-pic 0 -upic ${upic} -e ${estimator} ${calibration}\
							-s ${strategy} --method ${method} ${reduce}\
							${xvalid} ${logging} >>${dir}/${file_name};
						done
						echo "####################################################################################";
						for pic in 0.1 0.2 0.3 0.4 0.5 0.6; do
							echo "pic" ${pic};
							python -m experiment_standard_dataset\
							--dataset ${dataset}\
							-upic 0 -pic ${pic} -e ${estimator} ${calibration}\
							-s ${strategy} --method ${method} ${reduce}\
							${xvalid} ${logging} >>${dir}/${file_name};
						done
					done
				done
			done
		done
	done
elif [ $1 == "balanced" ]; then
	for dataset in pima; do
		estimator='nb'
		is_cali=0
		strategy=3
		method='simple_imputer.mean'
		dir=outputs/standard/${dataset}/balanced/results/${estimator}/
		echo ${dir};
		mkdir -p $dir;
		file_name=${dataset}_${estimator}_${is_cali}_${strategy}.tsv
		echo ${file_name};
		# python -m balanced --header-only >${dir}/${file_name};
		reduce=''
		if [ $estimator = 'reduce' ]
		then
			estimator='lr'
			reduce='--reduce'
		fi
		python -m balanced\
		-d ${dataset}\
		--pic 0 --uic 0 -m ${estimator} --method ${method} \
		>>${dir}/${file_name};
		echo "####################################################################################";
		for upic in 0.1 0.2 0.3 0.4 0.5 0.6; do
			echo "upic" ${upic};
			python -m balanced\
			-d ${dataset}\
			--pic 0 --uic ${upic} -m ${estimator} --method ${method}\
			>>${dir}/${file_name};
		done
		echo "####################################################################################";
		for pic in 0.1 0.2 0.3 0.4 0.5 0.6; do
			echo "pic" ${pic};
			python -m balanced\
			-d ${dataset}\
			--uic 0 --pic ${pic} -m ${estimator} --method ${method}\
			>>${dir}/${file_name};
		done
		echo "####################################################################################";
	done
fi
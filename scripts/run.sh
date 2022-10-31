#!/bin/bash

source myvenv/Scripts/activate
echo `pip --version`
out_dir='outputs'
mkdir ${out_dir} 2>/dev/null

export PYTHONPATH="$(pwd)"

printf "Table 1: \n"

python -m experiment --header-only >${out_dir}/ds_ccd_0.txt
methods=(baseline drop simple_imputer.mean iterative_imputer.mice iterative_imputer.missForest knn_imputer)
alphas=(0.25 0.50 0.75)
for method in ${methods[@]}; do
	for alpha in ${alphas[@]}; do
		python -m experiment --distype ds_ccd --method ${method}\
			--alpha ${alpha} >>${out_dir}/ds_ccd_0.txt
	done
done

printf "Table 2: \n"

python -m experiment --header-only >${out_dir}/ds_ccd_kip.txt
alphas=(0.25 0.50 0.75)
for method in ${methods[@]}; do
	for alpha in ${alphas[@]}; do
		python -m experiment --distype ds_ccd --method ${method}\
			--alpha ${alpha} -kip >>${out_dir}/ds_ccd_kip.txt
	done
done

printf "Table 3: \n"

python -m experiment --header-only >${out_dir}/ccd_0.txt
alphas=(0.25 0.50 0.75)
for method in ${methods[@]}; do
	for alpha in ${alphas[@]}; do
		python -m experiment --distype ccd --method ${method}\
			--alpha ${alpha} >>${out_dir}/ccd_0.txt
	done
done

printf "Table 4: \n"

python -m experiment --header-only >${out_dir}/ccd_kip.txt
alphas=(0.25 0.50 0.75)
for method in ${methods[@]}; do
	for alpha in ${alphas[@]}; do
		python -m experiment --distype ccd --method ${method}\
			--alpha ${alpha} -kip >>${out_dir}/ccd_kip.txt
	done
done


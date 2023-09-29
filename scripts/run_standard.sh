#!/bin/bash

dataset=$1

dir_name=outputs/standard/${dataset}/tables
mkdir -p ${dir_name};
file_name=${dir_name}/sota_cls_perf.tsv
echo ${file_name}
echo "Table: Group-wise performance of NBC classifier after
each missing value handling mechanism on ${dataset} dataset." >${file_name}

for estimator in nb pr; do
  echo ${estimator}
  echo "\midrule" >>${file_name}
  echo -n  "\multirow{6}{*}{${estimator}}" >>${file_name}
  for method in baseline drop mean mice knn softimpute; do
    echo ${method}
    echo -n -e "\t${method}" >>${file_name}
    python -m experiment_standard_dataset\
        --dataset ${dataset} --method ${method}\
        --estimator ${estimator} --strategy 3\
        --priv-ic-prob 0.1 --unpriv-ic-prob 0.4 >>${file_name};
  done
done
echo "\midrule" >>${file_name}
echo -n  "\multirow{6}{*}{RBC}" >>${file_name}
echo RBC;
for method in baseline drop mean mice knn softimpute; do
    echo ${method}
    echo -n -e "\t${method}" >>${file_name}
    python -m experiment_standard_dataset\
      --dataset ${dataset} --method ${method}\
      --estimator lr --reduce --strategy 3\
      --priv-ic-prob 0.1 --unpriv-ic-prob 0.4 >>${file_name};
done



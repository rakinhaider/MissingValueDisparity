#!/bin/bash

dataset=pima
dir="outputs/standard/${dataset}/tables"
mkdir -p ${dir}
file=${dir}/strateies.tsv
echo -n "" >${file}

for strategy in 1 2 3; do
  echo -n ${strategy}
  echo -n -e "\\multirow{6}{*}{${strategy}}" >>${file}
  for method in baseline drop mean mice knn softimpute; do
    echo -n ' '${method}
    echo -n -e "\t${method}" >>${file}
    python -m experiment_standard_dataset\
        --dataset ${dataset} --method ${method}\
        --estimator nb --strategy ${strategy}\
        --priv-ic-prob 0.1 --unpriv-ic-prob 0.4 >>${file};
  done
  echo '';
done
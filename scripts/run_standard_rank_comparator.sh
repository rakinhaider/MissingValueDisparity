#!/bin/bash

dataset=$1

echo -n -e "\\multirow{4}{*}{${dataset}}\t"
python -m rank_comparator_standard -d ${dataset} --strategy 1

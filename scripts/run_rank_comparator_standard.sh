#!/bin/bash

dataset=$1

echo "\midrule"
echo -n "\multirow{4}{*}{\texttt{${dataset}}}"
python -m rank_comparator_standard -d ${dataset} --strategy 1

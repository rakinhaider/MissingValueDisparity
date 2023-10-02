#!/bin/bash

echo -e "\\multirow{4}{*}{0}\t\\multirow{4}{*}{Mean}\t"
python -m rank_comparator -dt ccd --method mean

echo -e "\\multirow{4}{*}{-3 (<0)}\t\\multirow{4}{*}{Mean}\t"
python -m rank_comparator -dt ccd --method mean -gs -3

echo -e "\\multirow{4}{*}{-3 (<0)}\t\\multirow{4}{*}{k-NN}\t"
python -m rank_comparator -dt ccd --method knn -gs -3

echo -e "\\multirow{4}{*}{-3 (<0)}\t\\multirow{4}{*}{MICE}\t"
python -m rank_comparator -dt ccd --method mice -gs -3

echo -e "\\multirow{4}{*}{3 (>0)}\t\\multirow{4}{*}{Mean}\t"
python -m rank_comparator -dt ccd --method mean -gs 3
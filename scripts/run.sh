#!/bin/bash

export PYTHONPATH="$(pwd)"

printf "Table 1: \n"


python3 -m experiment --distype ds_ccd --method baseline --print-header
methods=(drop simple_imputer.mean iterative_imputer.mice iterative_imputer.missForest)
for method in ${methods}; do
	python3 -m experiment --distype ds_ccd --method ${method}
done


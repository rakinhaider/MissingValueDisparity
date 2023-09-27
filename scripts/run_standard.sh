#!/bin/bash

echo "Table 10: Group-wise performance of NBC classifier after
each missing value handling mechanism on PIMA dataset."
for estimator in nb pr; do
	echo 'Classifier' ${estimator}
	for method in baseline mean mice knn; do
		python -m experiment_standard_dataset\
			  --dataset pima --method ${method}\
			  --estimator ${estimator} --strategy 3\
			  --priv-ic-prob 0.1 --unpriv-ic-prob 0.4;
	done
done
echo 'Classifier' RBC
for method in baseline mean mice knn; do
	python -m experiment_standard_dataset\
		  --dataset pima --method ${method}\
		  --estimator lr --reduce --strategy 3\
		  --priv-ic-prob 0.1 --unpriv-ic-prob 0.4;
done


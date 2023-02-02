echo "Table 1: Changes in positive prediction probabilities and
rankings of identically distributed privileged and unpriv-
ileged group when moving to 𝜃′ from 𝜃."

python -m rank_comparator -dt ccd --method mean

echo "Table 2: Changes in positive prediction probabilities and
rankings of non-identically distributed (𝛿𝑠 > 0) privileged
and unprivileged group after mean imputation."

python -m rank_comparator -dt ccd --method mean -gs -3

echo "Table 3: Changes in positive prediction probabilities and
rankings of non-identically distributed (𝛿𝑠 < 0) privileged
and unprivileged group after KNN imputation."

python -m rank_comparator -dt ccd --method mean -gs -3 --method knn

echo "Table 4: Changes in positive prediction probabilities and
rankings of non-identically distributed (𝛿𝑠 > 0) privileged
and unprivileged group after mean imputation."

python -m rank_comparator -dt ccd --method mean -gs 3 --method knn

echo "Table 5: Group-wise performance of NBC classifier after each
missing value handling mechanism when the groups are
identically distributed and 𝑥1 and 𝑥2 are correlated."

for method in baseline drop mean mice knn; do
	python -m experiment_synthetic -dt corr --method ${method}
done

echo "Table 6: Group-wise performance of NBC classifier after each
missing value handling mechanism when the groups are non-
identically distributed and 𝑥1 and 𝑥2 are correlated."

for method in baseline drop mean mice knn; do
	python -m experiment_synthetic -dt corr --method ${method} -gs -3
done

# TODO: The FPR graph need re-do.

echo "Table 7: Group-wise performance of NBC classifier where
missing value in both train and test samples were imputed."
for method in baseline mean mice knn; do
	python -m experiment_synthetic -dt corr --method ${method} -gs -3 -tm train
done
#!/bin/bash

aif360folder=$1

./scripts/run_download.sh ${aif360folder}

echo "Table 1: Group-wise summaries of changes in positive prediction
probabilities and relative rankings of the individuals due to
imputation mechanisms."
./scripts/run_synthetic_rank_comparator.sh

echo "Table 2: Disparities in group-wise accuracies, base prediction
rates and false positive rates of NBC classifiers on SFBD with
correlated features, missing value disparity, and different
values of 𝛿𝑠."
./scripts/run_synthetic_classifier.sh

echo "Table 3: Disparities in group-wise accuracies, base prediction
rates and false positive rates of NBC classifiers on SFBD with
correlated features, missing value disparity and incomplete
test samples."
./scripts/run_synthetic_classifier_tm.sh

echo "Table 4: Benchmark datasets, number of samples and features
in each of them, and their corresponding sensitive attributes.
Bold sensitive attributes are used in our experiments."

echo "###################### Skipped ################################"


echo "Table 5: Group-wise summaries of changes in positive prediction
probabilities and relative rankings of the individuals due to
imputation mechanism. on real-world benchmark datasets."
for d in compas folkincome german pima heart; do
  ./scripts/run_standard_rank_comparator.sh ${d}
done

echo "Table 6: Disparities in group-wise accuracies, base prediction
rates and false positive rates of NBC classifiers on PIMA with
missing value disparity induced using different strategies."
./scripts/run_standard_strategy_experiment.sh 2>/dev/null
python -m tsv_to_tex_converter.py\
  --path outputs/standard/tables/pima/strategies.tsv

echo "Table 7: Disparities in group-wise accuracies, base prediction rates
and false positive rates of different classifiers on FolkIncome with
missing value disparity induced using strategy 3."

echo "###################### Skipped ########################################"
echo "######## Uncomment the following commands to generate results. ########"

# Takes quite long time. Uncomment if desired.
# Progress bars are printed.
# ./scripts/run_standard.sh folkincome
# If progress not desired, use the following redirection.
# ./scripts/run_standard.sh folkincome 2>/dev/null
# Then run the following command.
# python -m tsv_to_tex_converter \
#   --path outputs/standard/pima/tables/sota_cls_perf.tsv


echo "Table 8: Disparities in group-wise accuracies, base prediction
rates and false positive rates of different classifiers on PIMA
with missing value disparity induced using strategy 3."
./scripts/run_standard.sh pima &>/dev/null
python -m tsv_to_tex_converter \
  --path outputs/standard/pima/tables/sota_cls_perf.tsv


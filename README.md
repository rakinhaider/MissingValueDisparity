# Introduced Bias From Missing Value Disparity

## Description

In this project, we demonstrate disparity in rates of missing values tend
to introduce bias in machine learning systems even if the training data is free
from historical biases. We train Gaussian Naive Bayes classifiers on
synthetic fair balanced dataset *(SFBD)* of 10000
samples with 2 or 10 features. Results show that missing value disparity
skew the model closer to one group than the other. We also experiment
with real-world dataset, namely, COMPAS and PIMA and observe similar
discriminating patterns. Missing value disparity introduces bias by inducing
changes in prediction probabilities. In cases where the change is too low to
alter the original prediction (obtained from baseline without any missing
values), it can unfairly re-order the relative ranking of the individuals.

## Getting Started

### Pre-requisites

The codes are built and tested with the following
system configurations.

|  Item   |  Version                      |
| ------- | ------------------------------|
| OS      | Linux (Ubuntu 20.04.2 64bit)  |
| Python  | 3.8.10                        |
| pip     |  20.0.2                             |

### Randomization Seeds
The following seeds where used to ensure reproducibility of the results. Further experiments can be carried out by trying out different randomization seeds.

|  Case                         |  Seed    |
| ------- | ------------------------------|
| Training SFBD Generation      |   47  |
| Test SFBD Generation          |   41  |
| COMPAS train test split       |   23  |

### Getting Started

Create a virtual environment and activate it with the following
commands.

```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

### Installing Dependencies

Install all the dependencies with the following command.

```bash
pip3 install -r requirements.txt
```

After installation, you need to download the COMPAS dataset
and put it in **data/raw/compas/** inside the aif360 folder. We provide
*dowload.py* to perform this task. Run the following program
by replacing *<aif360-folder>* with the aif360 folder location.

```bash
python -m download --aif360-folder <aif360-folder>
```
In my case it was,
```bash
python -m download --aif360-folder myvenv/lib/python3.8/site-packages/aif360/
```
Now download the PIMA dataset with the following command.
```bash
python -m download --dataset pima
```

### Executing program

* To generate all the results use *run.sh*
* The following command will generate all the tables in the paper
```bash
chmod +x run.sh
./run.sh
```

* You can also run the script individually to obtain the results.
For example,
```bash
python3 -m rank_comparator --help
python3 -m rank_comparator_standard --help
python3 -m experiment_synthetic --help
python3 -m experiment_standard_dataset --help
```
## Help

Please ignore **tensorflow** related warnings.

## Authors

Anonymous Authors
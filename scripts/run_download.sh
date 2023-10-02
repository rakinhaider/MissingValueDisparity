#!/bin/bash

aif360folder=$1

python -m download --aif360-folder ${aif360folder} --dataset compas
python -m download --aif360-folder ${aif360folder} --dataset adult
python -m download --aif360-folder ${aif360folder} --dataset german
python -m download --aif360-folder ${aif360folder} --dataset pima
python -m download --aif360-folder ${aif360folder} --dataset heart

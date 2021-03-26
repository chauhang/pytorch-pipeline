#!/bin/bash

python ../src/ax_hpo.py  --max_epochs 3 --total_trials 3 --model_file ./input/model_file.py --data_module_file ./input/data_module_file.py --params_file ./input/parameters.json --output_dir ./output

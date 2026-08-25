#!/bin/bash
conda activate marlin23
python experiment_manager.py sac schema_update
python experiment_manager.py sac run
source run_evaluate.sh
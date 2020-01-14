#!/bin/bash
cd '/home/npg/projects/IDERMS/voltvar_package/CSAC'
export PYTHONPATH=$PYTHONPATH:'/home/npg/projects/IDERMS/voltvar_package/CSAC/rllab'
export PYTHONPATH=$PYTHONPATH:'/home/npg/projects/IDERMS/voltvar_package/CSAC/sac_safe_exp_ordinal'
/home/npg/.conda/envs/sac2/bin/python sac_safe_exp_ordinal/examples/CSAC_voltvar_bus34.py
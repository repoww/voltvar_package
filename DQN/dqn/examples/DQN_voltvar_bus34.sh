#!/bin/bash
cd '/home/npg/projects/IDERMS/voltvar_package/DQN'
export PYTHONPATH=$PYTHONPATH:'/home/npg/projects/IDERMS/voltvar_package/DQN/rllab'
export PYTHONPATH=$PYTHONPATH:'/home/npg/projects/IDERMS/voltvar_package/DQN/dqn'
/home/npg/.conda/envs/dqn/bin/python dqn/examples/DQN_voltvar_bus34.py

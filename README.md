# Whole-body PID + QP for Stanford Woofer

## Overview
The goal of this project is to simulate a controller that uses whole-body PID and QP. The wholy-body PID generates desired body accelerations. A QP solver then allocates foot forces to achieve the desired body accelerations. Then, multiplication by the jacobian transpose will transform the foot forces into joint torques.

## Install
1. Activate a python3 conda environment
2. Install cvxpy
```
conda install -c conda-forge lapack
conda install -c cvxgrp cvxpy
```
3. Install mujoco-py: https://github.com/openai/mujoco-py

## Test
```
python3 qp_test.py
```

## Details 
### Implementation 
* QP solver: CVXPY
* Simulator: Mujoco

### Lessons from "High-slope terrain locomotion for torque-controlled quadruped robots"
* Minimize joint torques using W=JcS⊤WτSJc⊤, where W is the positive-definite used to regularize f in equation (6).
* Regularize foot forces to be normal to surface, however, regularizing joint torques was more practical




#!/bin/bash

# Set script to exit on errors
set -e

# Command 1: Run with WATER DEPTH
echo "Running with calibration_quantities: WATER DEPTH, complete_bal_mode: True, only_bal_mode: False"
python bal_telemac.py --calibration_quantities "WATER DEPTH" 

# Command 2: Run with SCALAR VELOCITY
echo "Running with calibration_quantities: SCALAR VELOCITY, complete_bal_mode: True, only_bal_mode: True"
python bal_telemac.py --calibration_quantities "SCALAR VELOCITY" --complete_bal_mode True --only_bal_mode True

# Command 3: Run with SCALAR VELOCITY and WATER DEPTH
echo "Running with calibration_quantities: SCALAR VELOCITY and WATER DEPTH, complete_bal_mode: True, only_bal_mode: True"
python bal_telemac.py --calibration_quantities "SCALAR VELOCITY" "WATER DEPTH" --complete_bal_mode True --only_bal_mode True

# Command 4: Run with SCALAR VELOCITY and TKE
# echo "Running with calibration_quantities: SCALAR VELOCITY and TURBULENT ENERGY, complete_bal_mode: True, only_bal_mode: True"
# python bal_telemac.py --calibration_quantities "SCALAR VELOCITY" "TURBULENT ENERG" --complete_bal_mode True --only_bal_mode True

# Command 5: Run with SCALAR VELOCITY and TKE and WATER DEPTH
# echo "Running with calibration_quantities: SCALAR VELOCITY and TURBULENT ENERGY and WATER DEPTH, complete_bal_mode: True, only_bal_mode: True"
# python bal_telemac.py --calibration_quantities "SCALAR VELOCITY" "TURBULENT ENERG" "WATER DEPTH" --complete_bal_mode True --only_bal_mode True

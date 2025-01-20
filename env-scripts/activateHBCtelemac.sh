#!/bin/bash
# User definitions
TELEMAC_CONFIG_DIR=/home/modelling/telemac/v8p5r0/configs
TELEMAC_CONFIG_NAME=pysource.gfortranHPC.sh
HBCenv_DIR=/home/IWS/hidalgo/Documents/hydrobayescal/HBCenv/

# SCRIPT ACTIONS - DO NOT MODIFY BELOW

# get current working directory
ACT_DIR=$PWD

# enable error checks
feedback () {
  # check if any error occurred and give feedback
  if [[ $? -eq 0 ]]; then
    echo "  **Success**"
  else
    echo "! ERROR: failed to source environment (check path)."
  fi
}

# function to load TELEMAC config
activate_tm () {
  echo "> Loading TELEMAC config..."
  command cd $TELEMAC_CONFIG_DIR
  source $TELEMAC_CONFIG_NAME || return -1
}

# function to activate HBCenv and its packages
activate_hbc () {
  echo "> Loading HBCenv..."
  cd $HBCenv_DIR
  source bin/activate || return -1
}

# run functions
activate_hbc
feedback
activate_tm
feedback

# go back to act dir
cd $ACT_DIR

#!/bin/bash
# User definitions
TELEMAC_CONFIG_DIR=/home/amintvm/modeling/telemac-mascaret/configs
TELEMAC_CONFIG_NAME=pysource.gfortranHPC.sh

# SCRIPT ACTIONS - DO NOT MODIFY BELOW

# get current working directory
ACT_DIR=$PWD

# List of items


# enable error checks
feedback () {
  # check if any error occurred and give feedback
    if [[ $? -eq 0 ]]; then
        echo "  **Success**"
    else
        echo "! ERROR: failed to source environment (check path)."
        cd $ACT_DIR
    fi
}

# function to load TELEMAC config
activate_tm () {
    echo "> Loading TELEMAC config..."
    command cd $TELEMAC_CONFIG_DIR
    source $TELEMAC_CONFIG_NAME || return -1
}


# run functions

    activate_tm
    feedback


# go back to act dir
    cd $ACT_DIR
#done


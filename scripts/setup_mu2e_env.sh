#!/bin/bash

# Setup script for Mu2e environment
# This script should be sourced, not executed directly
# Usage: source setup_mu2e_env.sh

echo "Setting up..."

# Setup commands - these need to be sourced
kinit ${USER}@FNAL.GOV
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
export -n X509_USER_PROXY
setup dhtools
# muse setup SimJob

echo "Done"
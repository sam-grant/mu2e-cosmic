export -n X509_USER_PROXY
kinit ${USER}@FNAL.GOV
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
/cvmfs/mu2e.opensciencegrid.org/bin/getToken

#############################################
# TODO: add python envionment setup handling
#
# pyenv ana # VMS 
# mambe activate mu2e_env # EAF
#############################################

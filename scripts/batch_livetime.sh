#!/bin/bash
# muse setup SimJob for Offline
# muse setup ops for mdh

DEFNAME=$1 # needs to be the parent dataset...

parallel --progress -j 48 \
     'file={}; xrpath=$(mdh print-url "${file}" -l tape -s root); mu2e -c Offline/Print/fcl/printCosmicLivetime.fcl -s "${xrpath}" | grep "Livetime:" | awk -F: "{print \$NF}"' \
     < <(samweb list-definition-files $DEFNAME) > ${DEFNAME}.livetime

awk '{sum += $1} END {print "Total livetime:", sum, "sec"}' "${DATASET}.livetime"

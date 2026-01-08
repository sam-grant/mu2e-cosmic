#!/bin/bash

DEFNAME=$1
parallel --progress -j 48 \
    'file={}; xrpath=$(mdh print-url "${file}" -l tape -s root); mu2e -c Offline/Print/fcl/printCosmicLivetime.fcl -s "${xrpath}" | grep "Livetime:" | awk -F: "{print \$NF}"' \
    < <(samweb list-definition-files $DEFNAME) > ${DEFNAME}.livetime


#!/bin/bash

logfolder=/GW/D5data-13/ghazaleh/ranking_outputs/logs_test/
domain=book
pipeline=ENTITY_CONCEPT_JOINT_LINKING
querytype=chatprofile

echo "sbatch -a 1-1 -p gpu20 -o ${logfolder}KNRM_%a_${domain}_${querytype}_${pipeline}_%j.log run_knrm_single_test.sh $domain $pipeline $querytype;"
sbatch  -a 1-1 -p gpu20 -o ${logfolder}KNRM_%a_${domain}_${querytype}_${pipeline}_%j.log run_knrm_single_test.sh  $domain $pipeline $querytype;

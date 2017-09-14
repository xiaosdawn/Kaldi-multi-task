#!/bin/bash

# This is the latest version of training that combines RM and WSJ, in a setup where
# there are no shared phones (so it's like a multilingual setup).
# Before running this script, go to ../../wsj/s5, and after running
# the earlier stages in the run.sh (so the baseline SAT system is built),
# run the following:
# 
# local/online/run_nnet2.sh --stage 8 --dir exp/nnet2_online/nnet_ms_a_partial --exit-trai    n-stage 15    
#
# (you may want to keep --stage 8 on the above command line after run_nnet2.sh,
# in case you already ran some scripts in local/online/ in ../../wsj/s5/ and
# the earlier stages are finished, otherwise remove it).
 
 
stage=0
train_stage=-10
am_weight="90 1"
 
i=$(echo $am_weight | awk -F " " '{print $1}')
j=$(echo $am_weight | awk -F " " '{print $2}')
echo " the weight of 0 task is $i , the weight of 1 task is $j "
echo
echo
 
###=========== language 0 : well-trained nnet2 model ================================

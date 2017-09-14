#!/bin/bash

# This refers to ../../rm/s5/local/online/run_nnet2_wsj_joint.sh, 
# This is multi-task training that combines primary and auxiliary tasks, in a setup where
# there are no shared phones (so it's like a multilingual setup).
# Before running this script, go to ../../wsj/s5, and after running
# the earlier stages in the run.sh (so the baseline SAT system is built),
# run the following:
# 
# local/online/run_nnet2.sh --stage 8 --dir exp/nnet2_Code-Switch-Inf-Ph --exit-train-stage 15    
#
# (you may want to keep --stage 8 on the above command line after run_nnet2.sh,
# in case you already ran some scripts in local/online/ in ../../wsj/s5/ and
# the earlier stages are finished, otherwise remove it).
 
 
stage=0
train_stage=-10
# task weight can be setting
am_weight="0.6 0.4"
 
task_0=$(echo $am_weight | awk -F " " '{print $1}')
task_1=$(echo $am_weight | awk -F " " '{print $2}')
echo " the weight of 0 task is $task_0 , the weight of 1 task is $task_1 "
echo
echo
 
###=========== Primary task (task 0) : well-trained nnet2 model ================================
##-- Inf-Ph nnet2 system
srcdir=exp/nnet2_Code-Switch-Inf-Ph
src_dir_graph=exp/tri5a_Code-Switch-Inf-Ph/graph
src_alidir=exp/tri5a_Code-Switch-Inf-Ph_ali
src_lang=data_code-switch-Inf-Ph/lang
 
###========== Auxiliary task (task 1) : for multi-task learning ========================
##-- Ph-Ph tri5a system
tardir=exp/tri5a_Code-Switch-Ph-Ph

###========== destination file path ==================================================
dir=exp/nnet2_Code-Switch-Inf-Ph_joint_Ph-Ph_${task_0}-${task_1}-weight
 
 
use_gpu=true
set -e
 
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
 if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1"
  num_threads=1
  minibatch_size=512
else
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads"
fi

# Check inputs.
for f in $srcdir/egs/egs.1.ark $srcdir/egs/info/egs_per_archive \
    ${srcdir}_online/final.mdl $src_alidir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

if ! cmp $srcdir/tree $src_alidir/tree; then
  echo "$0: trees in $srcdir and $src_alidir do not match"
  exit 1;
fi

if [ $stage -le 0 ]; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train data/train_max2
fi
 
if [ $stage -le 1 ]; then
  echo "$0: dumping egs for tar data"
  steps/online/nnet2/get_egs2.sh --cmd "$train_cmd" \
    data/train_max2 ${tardir}_ali ${srcdir}_online ${dir}/egs
fi

if [ $stage -le 2 ]; then
  echo "$0: doing the multil-task training."
  # note: the arguments to the --mix-up option are (number of mixtures for task_0,
  # number of mixtures for task_1).  We just use fairly typical numbers for each
  # (although a bit fewer for task_0, since we're not so concerned about the
  # performance of that system).

  local/nnet2/train_multilang2.sh --num-jobs-nnet "1 1" \
    --stage $train_stage \
    --mix-up "0 4000" \
    --cleanup true --num-epochs 8 \
    --initial-learning-rate 0.005 --final-learning-rate 0.0005 \
    --cmd "$train_cmd" --parallel-opts "$parallel_opts" --num-threads "$num_threads" \
    --am-weight "$task_0 $task_1" \
   $src_alidir $srcdir/egs ${tardir}_ali $dir/egs ${srcdir}_online/final.mdl $dir
fi
 
if [ $stage -le 3 ]; then
  # Prepare the task_0 and task_1 setups for decoding, with config files

  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir}_online $src_lang $dir/0 ${dir}_0_online

  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir}_online data/lang $dir/1 ${dir}_1_online
fi

if [ $stage -le 4 ]; then
  # Decoding
  
  for file in dev test; do
  
     # Decoding for task_0
     local/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
        $src_dir_graph data_code-switch-Inf-Ph/${file} ${dir}_0_online/decode_0_${file}
     
     # Decoding for task_1
     local/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
        $tardir/graph data/${file} ${dir}_1_online/decode_1_${file}
  done
  wait
fi


 

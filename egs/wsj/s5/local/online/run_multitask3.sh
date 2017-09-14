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
am_weight="1.0 0.6 0.4 "
 
task_0=$(echo $am_weight | awk -F " " '{print $1}')
task_1=$(echo $am_weight | awk -F " " '{print $2}')
task_2=$(echo $am_weight | awk -F " " '{print $3}')
echo " the weight of 0 task is $task_0 , the weight of 1 task is $task_1 , the weight of 2 task is $task_2"
echo
echo
 
###=========== Primary task (task 0) : well-trained nnet2 model ================================
##-- Inf-Ph nnet2 system
srcdir0=exp/nnet2_Code-Switch-Inf-Ph
src_dir0_graph=exp/tri5a_Code-Switch-Inf-Ph/graph
src_alidir0=exp/tri5a_Code-Switch-Inf-Ph_ali
src0_data=data_code-switch-Inf-Ph
src0_lang=${src0_data}/lang

###=========== Auxiliary task 1 (task 1) : well-trained nnet2 model ================================
##-- Ph-Ph nnet2 system
srcdir1=exp/nnet2_Code-Switch-Ph-Ph
src_dir1_graph=exp/tri5a_Code-Switch-Ph-Ph/graph
src_alidir1=exp/tri5a_Code-Switch-Ph-Ph_ali
src1_data=data_code-switch-Ph-Ph
src1_lang=${src1_data}/lang

###=========== Auxiliary task2 (task 2) : for multi-task learning ========================
##-- LID tri5a system
tardir=exp/tri5a_Code-Switch-LID

###========== destination file path ==================================================
dir=exp/nnet2_Code-Switch-Inf-Ph_joint_Ph-Ph_joint_LID_${task_0}-${task_1}-${task_2}-weight
 
 
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
for f in $srcdir0/egs/egs.1.ark $srcdir0/egs/info/egs_per_archive \
    ${srcdir0}_online/final.mdl $src_alidir0/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done
for f in $srcdir1/egs/egs.1.ark $srcdir1/egs/info/egs_per_archive \
     ${srcdir1}_online/final.mdl $src_alidir1/ali.1.gz; do
   [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done


if ! cmp $srcdir0/tree $src_alidir0/tree; then
  echo "$0: trees in $srcdir0 and $src_alidir0 do not match"
  exit 1;
fi
 if ! cmp $srcdir1/tree $src_alidir1/tree; then
  echo "$0: trees in $srcdir1 and $src_alidir1 do not match"
  exit 1;
fi

if [ $stage -le 0 ]; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train data/train_max2
fi
 
if [ $stage -le 1 ]; then
  echo "$0: dumping egs for tar data"
  steps/online/nnet2/get_egs2.sh --cmd "$train_cmd" \
    data/train_max2 ${tardir}_ali ${srcdir0}_online ${dir}/egs
  steps/online/nnet2/get_egs2.sh --cmd "$train_cmd" \
    data/train_max2 ${tardir}_ali ${srcdir1}_online ${dir}/egs
fi

if [ $stage -le 2 ]; then
  echo "$0: doing the multil-task training."
  # note: the arguments to the --mix-up option are (number of mixtures for task_0,
  # number of mixtures for task_1).  We just use fairly typical numbers for each
  # (although a bit fewer for task_0, since we're not so concerned about the
  # performance of that system).

  local/nnet2/train_multilang2.sh --num-jobs-nnet "1 1 1" \
    --stage $train_stage \
    --mix-up "0 0 4000" \
    --cleanup true --num-epochs 8 \
    --initial-learning-rate 0.005 --final-learning-rate 0.0005 \
    --cmd "$train_cmd" --parallel-opts "$parallel_opts" --num-threads "$num_threads" \
    --am-weight "$task_0 $task_1 $task_2" \
   $src_alidir0 $srcdir0/egs $src_alidir1 $srcdir1/egs ${tardir}_ali $dir/egs ${srcdir0}_online/final.mdl $dir

fi
 
if [ $stage -le 3 ]; then
  # Prepare the task_i setups for decoding, with config files
 
  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir0}_online $src0_lang $dir/0 ${dir}_0_online

  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir1}_online $src1_lang $dir/1 ${dir}_1_online


  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir0}_online $tarlang $dir/2 ${dir}_2_0_online

  steps/online/nnet2/prepare_online_decoding_transfer.sh \
    ${srcdir1}_online $tarlang $dir/2 ${dir}_2_1_online
fi

if [ $stage -le 4 ]; then
  # Decoding
  
  for file in dev test; do
     # Decoding for task_0 
     local/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
        $src_dir0_graph $src0_data/${file} ${dir}_0_online/decode_0_${file} || exit 1;
     # Decodeing for task_1
     local/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
        $src_dir1_graph $src1_data/${file} ${dir}_1_online/decode_1_${file} || exit 1;
     # Decoding for task_2
     local/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
        $tardir/graph $tardata/${file} ${dir}_2_0_online/decode_2_0_${file} || exit 1;
     local/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 10 \
        $tardir/graph $tardata/${file} ${dir}_2_1_online/decode_2_1_${file} || exit 1;
  done
  wait
fi

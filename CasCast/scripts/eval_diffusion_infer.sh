#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=8
export CUDA_VISIBLE_DEVICES=0
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL


PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))

echo $PORT

torchrun --nproc_per_node=$gpus --master_port=$PORT evaluation.py \
--per_cpus $cpus \
--batch_size 2 \
--num_workers 2 \
--cfgdir ./experiments/cascast_diffusion/world_size1-ckpt \
--pred_len 12 \
--test_name test \
--ens_member 1 \
--cfg_weight 2 \
--metrics_type SEVIRSkillScore

sleep 2
rm -f batchscript-*
#!/bin/bash

gpus=1
node_num=1
single_gpus=`expr $gpus / $node_num`

cpus=13

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

while true
do 
  PORT=$((((RANDOM<<15)|RANDOM)%49152 + 10000))
  break
done
echo $PORT

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

python -u latent_preprocess.py \
--init_method 'tcp://127.0.0.1:'$PORT \
-c ./configs/sevir_used/compress_gt.yaml \
--world_size $gpus \
--per_cpus $cpus \
--tensor_model_parallel_size 1 \
--outdir './experiments' \
--desc  'gt_48x48x4'

#
sleep 2
rm -f batchscript-*
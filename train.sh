echo "set up environment..."
# for IB
export NCCL_IB_DISABLE=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_NET_GDR_LEVEL=5
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
# for others
export MKL_THREADING_LAYER=GNU
GPU_PER_NODE_COUNT=`nvidia-smi -L | wc -l`

[[ -z "$NODE_COUNT" ]] && NODE_COUNT=1 # || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCHAI_TASK_INDEX" ]] && RANK=0 || RANK=$AZ_BATCHAI_TASK_INDEX
[[ -z "$MASTER_ADDR" ]] && MASTER_ADDR=$MASTER_IP
[[ -z "$MASTER_ADDR" ]] && MASTER_ADDR=192.168.1.30

base=$1
logdir=$2
resume=${3:-""}
echo "base: $base"
echo "logdir: $logdir"
if [[ -n "$resume" && "$resume" == */* ]]; then
    resume_arg="--resume $resume"
    echo "resume: $resume"
else
    resume_arg=""
fi

torchrun --nproc_per_node=${GPU_PER_NODE_COUNT} \
    --node_rank=${NODE_RANK} \
    --nnodes=${NODE_COUNT} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    main.py --base ${base} --logdir ${logdir} ${resume_arg} --train --scale_lr False --wandb True

#!/bin/bash
#PBS -A WYOM0221
#PBS -N accfiy_v2
#PBS -l walltime=01:00:00
#PBS -l select=2:ncpus=64:ngpus=4:mem=480GB
#PBS -q develop
#PBS -o accfiy_v2.out
#PBS -e accfiy_v2.err
#PBS -k oe

set -ex

module purge
module load gcc cuda cray-mpich conda 
conda activate vllm

nodes=( $( cat $PBS_NODEFILE ) )
head_node=${nodes[0]}
head_node_ip=$(ip addr show bond0 | grep -E 'inet ' | awk '{print $2}' | cut -d/ -f1)  # More reliable IP retrieval

export LSCRATCH=/glade/derecho/scratch/ssuresh/
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export MPICH_GPU_MANAGED_MEMORY_SUPPORT_ENABLED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_SOCKET_IFNAME=hsn
export MPICH_OFI_NIC_POLICY=GPU
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_IB_DISABLE=1
export NCCL_CROSS_NIC=1
export NCCL_NCHANNELS_PER_NET_PEER=4
export MPICH_RDMA_ENABLED_CUDA=1
export NCCL_NET_GDR_LEVEL=PBH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_OPTIMIZED_MRS=false
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Calculate resources with error checking
num_nodes=$(sort -u $PBS_NODEFILE | wc -l)
num_gpus=$(nvidia-smi -L | wc -l)  # Direct GPU count from system
total_gpus=$((num_nodes * num_gpus))

echo "Cluster Configuration:"
echo "----------------------"
echo "Number of nodes: $num_nodes"
echo "GPUs per node: $num_gpus"
echo "Total GPUs: $total_gpus"
echo "Head node: $head_node ($head_node_ip)"

# Create hostfile with validation
hostfile="$PBS_JOBDIR/hostfile"
rm -f $hostfile
for node in $(sort -u $PBS_NODEFILE); do
    echo "$node slots=$num_gpus" >> $hostfile
done

echo -e "\nHostfile Contents:"
cat $hostfile && echo

export CODE_HOME=/glade/work/ssuresh/aiml/accfiy_v2/
cd $CODE_HOME || { echo "Failed to cd to $CODE_HOME"; exit 1; }
export ACCELERATE_HOSTFILE=$hostfile

# Launch command with improved arguments
accelerate launch \
    --config_file "$CODE_HOME/run_config/accel.yaml" \
    --num_processes "$total_gpus" \
    --num_machines "$num_nodes" \
    --main_process_ip "$head_node_ip" \
    --main_process_port 29502 \
    --mixed_precision "fp16" \
    --use_deepspeed \
    --deepspeed_config_file "$CODE_HOME/run_config/ds_config.json" \
    --gradient_accumulation_steps 2 \
    --zero3_init_flag True \
    --deepspeed_hostfile $hostfile \
    --dynamo_backend=no \
    "$CODE_HOME/train.py"

# Add final validation
if ! command -v accelerate >/dev/null 2>&1; then
    echo "Error: accelerate command not found!"
    exit 1
fi
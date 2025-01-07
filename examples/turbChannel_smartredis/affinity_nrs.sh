#!/bin/bash
num_gpus=$1
shift
gpu=$((${PALS_LOCAL_RANKID} % ${num_gpus} ))
export ZE_AFFINITY_MASK=$gpu
exec "$@"

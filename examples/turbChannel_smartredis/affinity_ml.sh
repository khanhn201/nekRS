#!/bin/bash
num_gpus=$1
offset=$2
shift
shift
gpu=$((${PALS_LOCAL_RANKID} % ${num_gpus} + ${offset} ))
export ZE_AFFINITY_MASK=$gpu
exec "$@"

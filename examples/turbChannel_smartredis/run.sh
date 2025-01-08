#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'
echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module load frameworks
source /gila/Aurora_deployment/balin/Nek/nekRSv24/venv/_ssim_env/bin/activate
module list
export NEKRS_HOME=/flare/Aurora_deployment/balin/Nek/nekRSv24/exe/nekrsv24_simai_frameworks/
export NEKRS_GPU_MPI=0
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RDZV_THRESHOLD=16384
export FI_CXI_RDZV_EAGER_SIZE=2048
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_OFLOW_BUF_SIZE=12582912
export FI_CXI_OFLOW_BUF_COUNT=3
export FI_CXI_REQ_BUF_MIN_POSTED=6
export FI_CXI_REQ_BUF_SIZE=12582912
export FI_MR_CACHE_MAX_SIZE=-1
export FI_MR_CACHE_MAX_COUNT=524288
export FI_CXI_REQ_BUF_MAX_CACHED=0
export FI_CXI_REQ_BUF_MIN_POSTED=6
export FI_CXI_RX_MATCH_MODE=hybrid

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

# Set the correct .udf file
#cp turbChannel_train.udf turbChannel.udf
#cp turbChannel_train.par turbChannel.par

# precompilation
date
case_tmp=turbChannel
ntasks_tmp=12
mpiexec -n 12 -ppn 12 --cpu-bind=list:0-7,104-111:8-15,112-119:16-23,120-127:24-31,128-135:32-39,136-143:40-47,144-151:52-59,156-163:60-67,164-171:68-75,172-179:76-83,180-187:84-91,188-195:92-99,196-203 -- ./.lhelper /flare/Aurora_deployment/balin/Nek/nekRSv24/exe/nekrsv24_simai_frameworks//bin/nekrs --setup ${case_tmp} --backend dpcpp --device-id 0  --build-only ${ntasks_tmp}
if [ $? -ne 0 ]; then
  exit
fi
date

# actual run
sim_arguments="--setup turbChannel.par --backend DPCPP --device-id 0"
python ssim_driver_polaris.py \
  sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=6 run_args.simprocs_pn=6 \
  sim.arguments="${sim_arguments}" sim.affinity=./affinity_nrs.sh \
  train.executable=${PWD}/trainer.py run_args.mlprocs=6 run_args.mlprocs_pn=6 \
  train.device=xpu train.affinity=./affinity_ml.sh

#rm turbChannel.udf
#rm turbChannel.par


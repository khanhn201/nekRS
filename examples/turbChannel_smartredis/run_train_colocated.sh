#!/bin/bash

# Set run env
nodes=`wc -l < $PBS_NODEFILE`

export TZ='/usr/share/zoneinfo/US/Central'
echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
module reload
#module load oneapi/eng-compiler/2024.07.30.002
#module load cmake
module load frameworks
source /gila/Aurora_deployment/balin/Nek/nekRSv24/venv/_ssim_env/bin/activate
module list
#export NEKRS_HOME=/gila/Aurora_deployment/balin/Nek/nekRSv24/exe/nekrsv24_simai
export NEKRS_HOME=/gila/Aurora_deployment/balin/Nek/nekRSv24/exe/nekrsv24_simai_frameworks
#export NEKRS_GPU_MPI=0
export MPICH_GPU_SUPPORT_ENABLED=0
export OCCA_DPCPP_COMPILER_FLAGS="-O3 -fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"
export FI_CXI_RX_MATCH_MODE=hybrid
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE

export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH

# Set the correct .udf file
cp turbChannel_train.udf turbChannel.udf
cp turbChannel_train.par turbChannel.par

# Run the driver script
sim_arguments="--setup turbChannel.par --backend DPCPP --device-id 0"
python ssim_driver_polaris.py \
  sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=6 run_args.simprocs_pn=6 \
  sim.arguments="${sim_arguments}" sim.affinity=./affinity_nrs.sh \
  train.executable=${PWD}/trainer.py run_args.mlprocs=6 run_args.mlprocs_pn=6 \
  train.device=xpu train.affinity=./affinity_ml.sh

rm turbChannel.udf
rm turbChannel.par

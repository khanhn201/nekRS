#!/bin/bash

export TZ='/usr/share/zoneinfo/US/Central'

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`
#module load frameworks
#source /flare/Aurora_deployment/balin/Nek/nekRSv24/nekRS/examples/tgv_gnn_traj_offline/_pyg/bin/activate
module list

export NEKRS_HOME=/flare/Aurora_deployment/balin/Nek/nekRSv24/exe/nekrsv24_simai_frameworks
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

# Run nekRS
mpiexec -n 4 -ppn 4 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 -- /flare/Aurora_deployment/balin/Nek/nekRSv24/exe/nekrsv24_simai_frameworks/bin/nekrs --setup tgv --backend dpcpp

# Generate the halo_info, edge_weights and node_degree files
mpiexec -n 4 -ppn 4 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 python ../../3rd_party/gnn/create_halo_info.py --SIZE 4 --POLY 7 --PATH ./gnn_outputs_poly_7

# Check the GNN input files
python ../../3rd_party/gnn/check_input_files.py --REF /flare/Aurora_deployment/balin/Nek/nekRSv24/tgv_traj_data/gnn_outputs_poly_7 --PATH ./gnn_outputs_poly_7

# Check the GNN trajectory files
for RANK in {0..3};
do
  python ../../3rd_party/gnn/check_input_files.py --REF /flare/Aurora_deployment/balin/Nek/nekRSv24/tgv_traj_data/traj_poly_7/tinit_0.000000_dtfactor_10/data_rank_${RANK}_size_4 --PATH ./traj_poly_7/tinit_0.000000_dtfactor_10/data_rank_${RANK}_size_4
done

# Train the GNN
#mpiexec -n 2 -ppn 2 --cpu-bind=list:1:8:16:24:32:40:53:60:68:76:84:92 python ../../3rd_party/gnn/main.py backend=ccl halo_swap_mode=all_to_all_opt gnn_outputs_path=/flare/Aurora_deployment/balin/Nek/nekRSv24/nekRS/examples/tgv_gnn_traj_offline/gnn_outputs_poly_3

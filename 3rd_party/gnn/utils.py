"""
Utilities for training and inferencing
"""

import sys
from typing import Optional, Union, Callable
import time
import logging
import numpy as np
import torch
import torch.distributed as dist

Tensor = torch.Tensor
log = logging.getLogger(__name__)

try:
    import mpi4py
    mpi4py.rc.initialize = False
    from mpi4py import MPI
    if not MPI.Is_initialized():
        MPI.Init()
    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
    WITH_DDP = True
except ModuleNotFoundError as e:
    WITH_DDP = False
    pass

try:
    WITH_CUDA = torch.cuda.is_available()
except:
    WITH_CUDA = False
    pass

try:
    WITH_XPU = torch.xpu.is_available()
except:
    WITH_XPU = False
    pass

def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)
    elif WITH_XPU:
        backend = 'ccl' if backend is None else str(backend)
    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )

def cleanup():
    dist.destroy_process_group()

def force_abort():
    time.sleep(2)
    if WITH_DDP:
        COMM.Abort()
    else:
        sys.exit("Exiting...")

def metric_average(val: Tensor):
    if (WITH_DDP):
        #dist.all_reduce(val, op=dist.ReduceOp.SUM)
        dist.reduce(val, 0, op=dist.ReduceOp.SUM)
        return val / SIZE
    return val

def metric_min(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.MIN)
    return val

def metric_max(val: Tensor):
    if (WITH_DDP):
        dist.all_reduce(val, op=dist.ReduceOp.MAX)
    return val

def all_gather_tensor(tensor_list: list[Tensor], tensor_local: Tensor):
    if (WITH_DDP):
        dist.all_gather(tensor_list, tensor_local)
        return tensor_list
    return [tensor_local]

def mpi_all_gather(local_obj):
    if (WITH_DDP):
        obj_list = COMM.allgather(local_obj)
        return obj_list
    return [local_obj]

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time", row_limit=20)
    print(output)
    p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

def collect_list_times(a_list):
    collected_arr = np.zeros((len(a_list)*SIZE))
    COMM.Gather(np.array(a_list),collected_arr,root=0)
    avg = np.mean(collected_arr)
    std = np.std(collected_arr)
    minn = np.amin(collected_arr); min_loc = [minn, 0]
    maxx = np.amax(collected_arr); max_loc = [maxx, 0]
    summ = np.sum(collected_arr)
    stats = {
                "avg": avg,
                "std": std,
                "sum": summ,
                "min": [min_loc[0],min_loc[1]],
                "max": [max_loc[0],max_loc[1]]
    }
    return stats

def average_list_times(a_list):
    sum_across_ranks = np.zeros((len(a_list)))
    COMM.Reduce(np.array(a_list),sum_across_ranks,op=MPI.SUM)
    avg = np.mean(sum_across_ranks)
    return avg

def print_fom(n_nodes_local: int, local_times: list, local_throughputs: list):
    n_nodes_global = COMM.reduce(n_nodes_local)
    global_times = COMM.gather(local_times, root=0)
    global_throughputs = COMM.gather(local_throughputs, root=0)
    if RANK == 0:
        global_parallel_throughputs = np.zeros(len(local_throughputs))
    else:
        global_parallel_throughputs = None
    COMM.Reduce(np.array(local_throughputs),
                         global_parallel_throughputs, 
                         op=MPI.SUM, root=0
    )
    if RANK == 0:
        log.info('Performance metrics:')
        log.info(f'Total number of graph nodes: {n_nodes_global}')
        log.info(f'Step time [sec]: min={min(global_times[0]):.4g}, max={max(global_times[0]):.4g}, mean={sum(global_times[0])/len(global_times[0]):.4g}')
        log.info(f'Step throughput [million nodes/sec]: min={min(global_throughputs[0]):.4g}, max={max(global_throughputs[0]):.4g}, mean={sum(global_throughputs[0])/len(global_throughputs[0]):.4g}')
        log.info(f'Parallel throughput [million nodes/sec]: min={np.amin(global_parallel_throughputs):.4g}, max={np.amax(global_parallel_throughputs):.4g}, mean={np.mean(global_parallel_throughputs):.4g}')

"""
Create halo swap info.
"""
import argparse 
import numpy as np
from typing import Tuple

import mpi4py
#from mpi4py.util import pkl5
#COMM = pkl5.Intracomm(MPI.COMM_WORLD)
mpi4py.rc.initialize = False
from mpi4py import MPI
if not MPI.Is_initialized():
    MPI.Init()
COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

import torch
from torch_geometric.data import Data
import torch_geometric.utils as utils
import torch_geometric.nn as tgnn


def make_reduced_graph() -> Tuple[Data, Data, torch.Tensor]:
    if RANK == 0: print('Loading data from file ...', flush=True)
    path_to_pos_full = main_path + 'pos_node_rank_%d_size_%d' %(RANK,SIZE)
    path_to_ei = main_path + 'edge_index_rank_%d_size_%d' %(RANK,SIZE)
    path_to_glob_ids = main_path + 'global_ids_rank_%d_size_%d' %(RANK,SIZE)
    path_to_unique = main_path + 'local_unique_mask_rank_%d_size_%d' %(RANK,SIZE)
    path_to_halo_ids = None 
    if SIZE > 1:
        path_to_halo_ids =  main_path + 'halo_ids_rank_%d_size_%d' %(RANK,SIZE)
        path_to_unique_halo = main_path + 'halo_unique_mask_rank_%d_size_%d' %(RANK,SIZE)

    # ~~~~ Get positions and global node index  
    #if args.LOG=='debug': print('[RANK %d]: Loading positions and global node index' %(RANK), flush=True)
    pos = np.fromfile(path_to_pos_full + ".bin", dtype=np.float64).reshape((-1,3))
    gli = np.fromfile(path_to_glob_ids + ".bin", dtype=np.int64).reshape((-1,1))
     
    # ~~~~ Back-out number of elements
    Ne = int(pos.shape[0]/Np)
    #if args.LOG=='debug': print('[RANK %d]: Number of elements is %d' %(RANK, Ne), flush=True)

    # ~~~~ Get edge index
    #if args.LOG=='debug': print('[RANK %d]: Loading edge index' %(RANK), flush=True)
    ei = np.fromfile(path_to_ei + ".bin", dtype=np.int32).reshape((-1,2)).T
    ei = ei.astype(np.int64)

    # ~~~~ Get local unique mask 
    #if args.LOG=='debug': print('[RANK %d]: Loading local unique mask' %(RANK), flush=True)
    local_unique_mask = np.fromfile(path_to_unique + ".bin", dtype=np.int32)

    # ~~~~ Get halo unique mask 
    halo_unique_mask = np.array([])
    if SIZE > 1:
        halo_unique_mask = np.fromfile(path_to_unique_halo + ".bin", dtype=np.int32)
    COMM.Barrier()
    if RANK == 0: print('Done data loading from file \n', flush=True)
    
    # ~~~~ Make graph:
    if RANK == 0: print('Making graph ...', flush=True)
    #if args.LOG=='debug': print('[RANK %d]: Making graph' %(RANK), flush=True)
    data = Data(x = torch.tensor(pos), edge_index = torch.tensor(ei), pos = torch.tensor(pos), global_ids = torch.tensor(gli.squeeze()), local_unique_mask = torch.tensor(local_unique_mask), halo_unique_mask = torch.tensor(halo_unique_mask))
    data.edge_index = utils.remove_self_loops(data.edge_index)[0]
    data.edge_index = utils.coalesce(data.edge_index)
    data.edge_index = utils.to_undirected(data.edge_index)

    # ~~~~ Append list of graphs
    #graph_list.append(data)
    COMM.Barrier()
    if RANK == 0: print('Done making graph \n', flush=True)

    # ~~~~ Reduce size of graph 
    if RANK == 0: print('Making reduced graph ...', flush=True)
    #if args.LOG=='debug': print('[RANK %d]: Reduced size of edge_index based on unique node ids' %(RANK), flush=True)
    # X: [First isolate local nodes] 
    idx_local_unique = torch.nonzero(data.local_unique_mask).squeeze(-1)
    idx_halo_unique = torch.tensor([], dtype=idx_local_unique.dtype)
    if SIZE > 1:
        idx_halo_unique = torch.nonzero(data.halo_unique_mask).squeeze(-1)
    idx_keep = torch.cat((idx_local_unique, idx_halo_unique))   
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PYGEOM FUNCTION -- this gets the reduced edge_index 
    num_nodes = data.x.shape[0]
    perm = idx_keep 
    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = data.edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]
    edge_index_reduced = torch.stack([row, col], dim=0)
    edge_index_reduced = utils.coalesce(edge_index_reduced)
    edge_index_reduced = utils.to_undirected(edge_index_reduced)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    pos_reduced = data.pos[idx_keep]
    gid_reduced = data.global_ids[idx_keep]
    data_reduced = Data(x = pos_reduced, pos = pos_reduced, edge_index = edge_index_reduced, global_ids = gid_reduced)
    n_not_halo = len(idx_local_unique)
    n_halo = len(idx_halo_unique)
    data_reduced.local_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.local_unique_mask[:n_not_halo] = 1
    data_reduced.halo_unique_mask = torch.zeros(n_not_halo + n_halo, dtype=torch.int64)
    data_reduced.halo_unique_mask[n_not_halo:] = 1
    gid = data.global_ids 
    zero_indices = torch.where(gid == 0)[0]
    consecutive_negatives = -1 * torch.arange(1, len(zero_indices) + 1)
    gid[zero_indices] = consecutive_negatives
    data.global_ids = gid
    data_reduced.global_ids = gid[idx_keep]
    if RANK == 0: print('Done making reduced graph \n', flush=True)
    return data, data_reduced, idx_keep

    #graph_reduced_list.append(data_reduced)

# ~~~~ Get the new halo_ids
def get_reduced_halo_ids(data_reduced) -> torch.Tensor:
    idx_halo_unique = torch.tensor([], dtype=torch.int64)
    halo_ids = torch.tensor([], dtype=torch.int64)
    halo_ids_full = torch.tensor([], dtype=torch.int64)
    if SIZE > 1:
        #gid = data.global_ids

        # What are the local ids of the halo nodes ? 
        n_local = data_reduced.local_unique_mask.sum().item()
        n_halo = data_reduced.halo_unique_mask.sum().item()
        idx_halo_unique = torch.tensor(list(range(n_local, n_local+n_halo)))

        # What are the corresponding global ids? 
        gid_halo_unique = data_reduced.global_ids[idx_halo_unique] 

        # What is the current rank? 
        rank_array = torch.ones_like(gid_halo_unique, dtype=torch.int64) * RANK 

        # [Local ids, global ids, rank] 
        halo_ids = torch.concat((idx_halo_unique.view(-1,1),
                                 gid_halo_unique.view(-1,1),
                                 rank_array.view(-1,1)), dim=1)
        
        halo_ids_shape_list = COMM.allgather(halo_ids.shape[0])
        halo_ids_full_length = sum(halo_ids_shape_list)
        halo_ids_full_width = halo_ids.shape[1]
        halo_ids_full_type = halo_ids.dtype
        halo_ids_full = torch.zeros(halo_ids_full_length,halo_ids_full_width, dtype=halo_ids_full_type)

        count = [halo_ids_shape_list[i]*halo_ids_full_width for i in range(SIZE)]
        displ = [sum(count[:i]) for i in range(SIZE)]
        if args.LOG == 'debug' and RANK==0:
            print(f'count={count}',flush=True)
            print(f'displ={displ}',flush=True)
        COMM.Allgatherv([halo_ids,MPI.LONG],[halo_ids_full,count,displ,MPI.LONG])
    return halo_ids_full

# Prepares the halo_info matrix for halo swap 
def get_halo_info(data_reduced, halo_ids_full) -> list:
    if SIZE == 1:
        halo_info_glob = [torch.tensor([], dtype=torch.int64)]
    else:
        # Collect number of nodes
        n_nodes = []
        n_nodes.append(data_reduced.pos.shape[0])
        n_nodes_glob = COMM.allgather(n_nodes[0])

        # concatenate 
        #halo_ids_full = torch.cat(halo_ids_list)
        #halo_ids_full = torch.cat(halo_ids_list_glob)
        #del halo_ids_list_glob

        # take absolute value of global id 
        halo_ids_full[:,1] = torch.abs(halo_ids_full[:,1])

        # sort in ascending order of global id 
        global_ids = halo_ids_full[:,1] 
        _, idx_sort = torch.sort(global_ids)
        halo_ids_full = halo_ids_full[idx_sort]

        # get the frequency of nodes 
        global_ids = halo_ids_full[:,1] 
        output = torch.unique_consecutive(global_ids, return_inverse=True, return_counts=True)
        counts_unique = output[2]
        counts = output[2][output[1]]
        counts = counts.reshape((-1,1))

        # append the counts to halo_ids_full
        halo_ids_full = torch.cat([halo_ids_full, counts], dim=1)

        # Get the number of halo nodes for each rank
        #halo_info = []
        halo_ids_rank = halo_ids_full[halo_ids_full[:,2] == RANK]
        Nhalo_rank = torch.sum(halo_ids_rank[:,3] - 1)
        #halo_info.append(torch.zeros((Nhalo_rank,4), dtype=torch.int64))
        #halo_info_glob = COMM.allgather(halo_info[0])
        Nhalo_rank_glob = COMM.allgather(Nhalo_rank)
        halo_info_glob = [torch.zeros((Nhalo_rank_glob[i],4), dtype=torch.int64) for i in range(SIZE)]
        if args.LOG == 'debug': print('[RANK %d]: halo info shape is ' %(RANK), halo_info_glob[RANK].shape, flush=True)

        # Loop through counts 
        halo_counts = [0]*SIZE
        idx = 0 
        for i in range(len(counts_unique)):
            count = counts_unique[i].item()
            halo_temp = halo_ids_full[idx:idx+count]
            #for j in range(count): 
            #    a = halo_ids_full[idx]
        
            rank_list = halo_temp[:,2]
            for j in range(len(rank_list)):
                rank = rank_list[j].item()

                # get the current rank info 
                node_local_id = halo_temp[j,0] # local node id of sender on "rank"
                node_global_id = halo_temp[j,1] # global node id of sender on "rank" 

                # loop through the same nodes not on this rank index
                halo_temp_nbrs = halo_temp[torch.arange(len(halo_temp)) != j]
                for k in range(len(halo_temp_nbrs)):
                    neighbor_rank = halo_temp_nbrs[k,2] # neighboring rank for this halo node 
                    node_halo_id = n_nodes_glob[rank] + halo_counts[rank] # local node id of halo node on "rank"  

                    # update the halo info matrix
                    halo_info_glob[rank][halo_counts[rank]][0] = node_local_id
                    halo_info_glob[rank][halo_counts[rank]][1] = node_halo_id
                    halo_info_glob[rank][halo_counts[rank]][2] = node_global_id
                    halo_info_glob[rank][halo_counts[rank]][3] = neighbor_rank

                    # update the count 
                    halo_counts[rank] += 1

                    # print('[RANK %d] \t %d \t %d \t %d \n' %(rank, node_local_id, node_halo_id, neighbor_rank))

            #print('count = %d, idx = %d' %(count, idx))
            #print(a)
            #print('\n')
            idx+=count
    return halo_info_glob


# ~~~~ Get node degree from halo_info
def get_node_degree(data_reduced, halo_info_rank) -> torch.Tensor:
    if SIZE == 1:
        node_degree = torch.ones(data_reduced.pos.shape[0])
    else:
        sample = data_reduced
        n_nodes_local = sample.pos.shape[0]
        node_degree = torch.ones(n_nodes_local)
        #halo_info_rank = halo_info_glob[RANK]
        unique_local_indices, counts = torch.unique(halo_info_rank[:,0], return_counts=True)
        node_degree[unique_local_indices] += counts
    return node_degree

# ~~~~ Get edge weights to account for duplicate edges 
def get_edge_weights(data_reduced, halo_info_glob) -> torch.Tensor:
    if SIZE == 1:
        edge_weights = torch.ones(data_reduced.edge_index.shape[1])
    else: 
        # Collect edge_index shape
        edge_index_shape_list = COMM.allgather(data_reduced.edge_index.shape)

        # Collect global_id shape
        global_ids_shape_list = COMM.allgather(data_reduced.global_ids.shape)

        sample = data_reduced
        halo_info_rank = halo_info_glob[RANK]

        # Get neighboring procs for this rank
        neighboring_procs = np.unique(halo_info_rank[:,3])
        if args.LOG == 'debug': 
            print(f'[RANK {RANK}]: Found {len(neighboring_procs)} neighboring procs.: {neighboring_procs}',flush=True)
        
        # Initialize edge weights 
        num_edges_own = sample.edge_index.shape[1]
        edge_weights = torch.ones(num_edges_own)

        # Send/receive the edge index
        for j in neighboring_procs:
            COMM.Isend([data_reduced.edge_index,MPI.INT],dest=j)
        edge_index_nei_list = [] 
        for j in neighboring_procs:
            tmp = torch.zeros(edge_index_shape_list[j], dtype=torch.int64) 
            COMM.Recv([tmp,MPI.INT],source=j)
            edge_index_nei_list.append(tmp)
        COMM.Barrier()
        if RANK == 0: print('Communicated the edge_index arrays', flush=True)
        
        # Send/receive the global ids
        for j in neighboring_procs:
            COMM.Isend([data_reduced.global_ids,MPI.INT],dest=j)
        global_ids_nei_list = [] 
        for j in neighboring_procs:
            tmp = torch.zeros(global_ids_shape_list[j], dtype=torch.int64) 
            COMM.Recv([tmp,MPI.INT],source=j)
            global_ids_nei_list.append(tmp)
        COMM.Barrier()
        if RANK == 0: print('Communicated the global_ids arrays', flush=True)

        for i in range(len(neighboring_procs)):
            rank_own = RANK
            rank_nei = neighboring_procs[i]

            halo_info_own = halo_info_rank
            halo_info_nei = halo_info_glob[rank_nei]

            halo_info_own = halo_info_own[halo_info_own[:,3] == rank_nei, :] 
            halo_info_nei = halo_info_nei[halo_info_nei[:,3] == rank_own, :] 

            # check the global id ordering 
            if not torch.equal(halo_info_own[:,2], halo_info_nei[:,2]):
                raise AssertionError("halo_info tensors are not properly ordered by global id")
            
            # Get connectivities 
            edge_index_own = data_reduced.edge_index
            edge_index_nei = edge_index_nei_list[i]

            #num_edges_nei = edge_index_nei.shape[1]
            #ew_nei = torch.ones(num_edges_nei)
            #edge_id_own = torch.arange(num_edges_own, dtype=torch.int64) 
            #edge_id_nei = torch.arange(num_edges_nei, dtype=torch.int64) 

            # Get the edges of owner rank nodes 
            local_id_own = halo_info_own[:,0]
            mask_own = torch.isin(edge_index_own[1], local_id_own)
            edge_index_own_subset = edge_index_own[:, mask_own]

            # Get the edges of neighbor rank nodes 
            local_id_nei = halo_info_nei[:,0]
            mask_nei = torch.isin(edge_index_nei[1], local_id_nei)
            edge_index_nei_subset = edge_index_nei[:, mask_nei]

            # Convert to global ids
            gli_own = data_reduced.global_ids
            send, recv = edge_index_own_subset
            send_global = gli_own[send]
            recv_global = gli_own[recv]
            edge_index_own_subset_global = torch.stack((send_global, recv_global))  

            gli_nei = global_ids_nei_list[i]
            send, recv = edge_index_nei_subset
            send_global = gli_nei[send]
            recv_global = gli_nei[recv]
            edge_index_nei_subset_global = torch.stack((send_global, recv_global))  

            # Compute pairing functions: Cantor pair = 0.5 * (k1 + k2) * (k1 + k2 + 1) + k2 
            k1, k2 = edge_index_own_subset_global
            k1 = k1.to(torch.float64)
            k2 = k2.to(torch.float64)
            cpair_own = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).to(torch.int64)

            k1, k2 = edge_index_nei_subset_global
            k1 = k1.to(torch.float64)
            k2 = k2.to(torch.float64)
            cpair_nei = (0.5 * (k1 + k2) * (k1 + k2 + 1) + k2).to(torch.int64)

            # Find which owner edges are duplicated in neighbor edges 
            duplicates_count = torch.zeros_like(cpair_own)
            for eid in range(len(cpair_own)):
                edge = cpair_own[eid]
                duplicates_count[eid] = (cpair_nei == edge).sum().item()

            # place in edge weighst 
            edge_weights[mask_own] += duplicates_count 
    return edge_weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--POLY', type=int, required=True, help='Specify the polynomial order.')
    parser.add_argument('--PATH', type=str, required=True, help='Specify the gnn_outputs folder path.')
    parser.add_argument('--LOG', type=str, default='info', required=False, help='Logging verbosity')
    args = parser.parse_args()

    POLY = args.POLY
    DIM = 3
    Np = (POLY+1)**DIM
    main_path = args.PATH + '/' 

    # Make graph and reduced graph
    data, data_reduced, idx_keep = make_reduced_graph()

    # Get halo_ids for reduced graph
    halo_ids_full = get_reduced_halo_ids(data_reduced)

    # Compute the halo_info
    if RANK == 0: print('Computing halo_info ...', flush=True)
    halo_info_glob = get_halo_info(data_reduced, halo_ids_full)
    if RANK == 0: print('Done\n', flush=True)

    # Compute the node_degree
    if RANK == 0: print('Computing node_degree ...', flush=True)
    node_degree = get_node_degree(data_reduced, halo_info_glob[RANK])
    if RANK == 0: print('Done\n', flush=True)

    # Compute the edge_weights
    if RANK == 0: print('Computing edge_weights ...', flush=True)
    edge_weights = get_edge_weights(data_reduced, halo_info_glob)
    if RANK == 0: print('Done\n', flush=True)

    # Write files
    if RANK == 0: print('Writing halo_info, edge_weights, node_degree ...', flush=True)
    np.save(main_path + 'halo_info_rank_%d_size_%d.npy' %(RANK,SIZE), halo_info_glob[RANK].numpy())
    np.save(main_path + 'node_degree_rank_%d_size_%d.npy' %(RANK,SIZE), node_degree.numpy())
    np.save(main_path + 'edge_weights_rank_%d_size_%d.npy' %(RANK,SIZE), edge_weights.numpy())
    COMM.Barrier()
    if RANK == 0: print('Done \n', flush=True)





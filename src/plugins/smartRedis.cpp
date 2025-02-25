#ifdef NEKRS_ENABLE_SMARTREDIS

#include "nrs.hpp"
#include "nekInterfaceAdapter.hpp"
#include "smartRedis.hpp"
#include "client.h"
#include <string>
#include <vector>

// Initialize the SmartRedis client
smartredis_client_t::smartredis_client_t(nrs_t *nrs)
{
  // MPI
  _rank = platform->comm.mpiRank;
  _size = platform->comm.mpiCommSize;

  // nekRS object
  _nrs = nrs;

  // Set up SmartRedis parameters
  platform->options.getArgs("SSIM DB DEPLOYMENT",_deployment);
  _num_tot_tensors = _size;
  if (_deployment == "colocated") {
    _num_db_tensors = std::stoi(getenv("PALS_LOCAL_SIZE"));
    if (_rank%_num_db_tensors == 0)
      _head_rank = _rank;
    _db_nodes = 1;
  } else if (_deployment == "clustered") {
    _num_db_tensors = _size;
    _head_rank = 0;
    platform->options.getArgs("SSIM DB NODES",_db_nodes);
  }

  // Initialize SR client
  if (_rank == 0)
    printf("\nInitializing client ...\n");
  bool cluster_mode;
  if (_db_nodes > 1)
    cluster_mode = true;
  else
    cluster_mode = false;
  std::string logger_name("Client");
  _client = new SmartRedis::Client(cluster_mode, logger_name); // allocates on heap
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("All done\n");
  fflush(stdout);
}
 // destructor
smartredis_client_t::~smartredis_client_t()
{
    if (_rank == 0) printf("Taking down smartredis_client_t\n");
    delete[] _client;
}

// Initialize the check-run tensor on DB
void smartredis_client_t::init_check_run()
{
  std::vector<int> check_run(1);
  check_run[0] = 1;
  std::string run_key = "check-run";

  if (_rank % _num_db_tensors == 0) {
    _client->put_tensor(run_key, check_run.data(), {1},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Put check-run in DB\n\n");
  fflush(stdout);
}

// Check value of check-run variable to know when to quit
int smartredis_client_t::check_run()
{
  int exit_val;
  std::string run_key = "check-run";
  int *check_run = new int[1]();

  // Check value of check-run tensor in DB from head rank
  if (_rank%_num_db_tensors == 0) {
    _client->unpack_tensor(run_key, check_run, {1},
                       SRTensorTypeInt32, SRMemLayoutContiguous);
    exit_val = check_run[0];
  }

  // Broadcast exit value and return it
  MPI_Bcast(&exit_val, 1, MPI_INT, 0, platform->comm.mpiComm);
  if (exit_val==0 && platform->comm.mpiRank==0) {
    printf("\nML training says time to quit ...\n");
  }
  fflush(stdout);
  return exit_val;
}

// Put step number in DB
void smartredis_client_t::put_step_num(int tstep)
{
  // Initialize local variables
  std::string key = "step";
  std::vector<long> step_num(1,0);
  step_num[0] = tstep;

  // Send time step to DB
  if (_rank == 0)
    printf("\nSending time step number ...\n");
  if (_rank%_num_db_tensors == 0) {
    _client->put_tensor(key, step_num.data(), {1},
                    SRTensorTypeInt64, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n\n");
  fflush(stdout);
}

// Append a new DataSet to a list and put in DB
void smartredis_client_t::append_dataset_to_list(const std::string& dataset_name,
  const std::string& tensor_name,
  const std::string& list_name,
  dfloat* data,
  unsigned long int num_rows,
  unsigned long int num_cols) 
{
  if (_rank == 0)
    printf("\nAdding dataset to list ...\n");
  SmartRedis::DataSet dataset(dataset_name);
  dataset.add_tensor(tensor_name,  data, {num_rows,num_cols}, 
                     SRTensorTypeDouble, SRMemLayoutContiguous);
  _client->put_dataset(dataset);
  _client->append_to_list(list_name,dataset);
  if (_rank == 0)
    printf("Done\n\n");
}

// Initialize training for the wall shear stress model
void smartredis_client_t::init_wallModel_train()
{
  std::vector<int> tensor_info(6,0);
  auto mesh = _nrs->mesh;
  auto [xvec, yvec, zvec] = mesh->xyzHost();

  if (_rank == 0) 
    printf("\nSending training metadata for wall shear stress model ...\n");

  // Loop over mesh coordinates to find wall and off-wall node indices
  std::vector<int> ind_owall_nodes_raw;
  for (int i=0; i<mesh->Nlocal; i++) {
    const auto y = yvec[i];
    if (y <= _wall_height + _eps) {
      _ind_wall_nodes.push_back(i);
    }
    else if (y >= _off_wall_height - _eps && y <= _off_wall_height + _eps) {
      ind_owall_nodes_raw.push_back(i);
    }
  }
  int wall_node_ct = _ind_wall_nodes.size();
  int owall_node_ct = ind_owall_nodes_raw.size();
  printf("Found %d wall nodes and %d off-wall nodes\n",wall_node_ct,owall_node_ct);
  fflush(stdout);
  assert(wall_node_ct == owall_node_ct);
  _npts_per_tensor = wall_node_ct;
  _num_samples = wall_node_ct;

  // Pair the wall nodes with the off-wall nodes based on the x and z coordinate
  // to pair the inputs and outputs of the model
  std::vector<int> ind_owall_nodes_tmp(wall_node_ct,0);
  ind_owall_nodes_tmp = ind_owall_nodes_raw;
  int pairs = 0;
  for (int i=0; i<wall_node_ct; i++) {
    int ind_w = _ind_wall_nodes[i];
    const auto x_wall = xvec[ind_w];
    const auto z_wall = zvec[ind_w];
    for (int j=0; j<ind_owall_nodes_tmp.size(); j++) {
      int ind_ow = ind_owall_nodes_tmp[j];
      const auto x_owall = xvec[ind_ow];
      const auto z_owall = zvec[ind_ow];
      if ((x_owall >= x_wall - _eps && x_owall <= x_wall + _eps) && 
          (z_owall >= z_wall - _eps && z_owall <= z_wall + _eps)) {
        _ind_owall_nodes_matched.push_back(ind_ow);
        ind_owall_nodes_tmp.erase(ind_owall_nodes_tmp.begin() + j);
        pairs++;
      }
    }
  }
  //printf("Found %d pairs of wall nodes and off-wall nodes\n",pairs);

  // Create and send metadata for training
  tensor_info[0] = _npts_per_tensor;
  tensor_info[1] = _num_tot_tensors;
  tensor_info[2] = _num_db_tensors;
  tensor_info[3] = _head_rank;
  tensor_info[4] = _num_inputs;
  tensor_info[5] = _num_outputs;
  std::string info_key = "tensorInfo";
  if (_rank%_num_db_tensors == 0) {
    _client->put_tensor(info_key, tensor_info.data(), {6},
                    SRTensorTypeInt32, SRMemLayoutContiguous);
  }
  MPI_Barrier(platform->comm.mpiComm);

  if (_rank == 0)
    printf("Done\n\n");
  fflush(stdout);
}

// Put training data for wall shear stress model in DB
void smartredis_client_t::put_wallModel_data(int tstep)
{
  unsigned long int num_cols = _num_inputs+_num_outputs;
  int size_U = _nrs->fieldOffset * _nrs->mesh->dim;
  dfloat mue;
  std::string key = "x." + std::to_string(_rank) + "." + std::to_string(tstep);
  dfloat *U = new dfloat[size_U]();
  dfloat *train_data = new dfloat[_num_samples*num_cols]();
  dfloat *vel_data = new dfloat[_num_samples]();
  dfloat *shear_data = new dfloat[_num_samples]();

  // Extract velocity at off-wall nodes (inputs)
  _nrs->o_U.copyTo(U, size_U);
  for (int i=0; i<_num_samples; i++) {
    int ind = _ind_owall_nodes_matched[i];
    vel_data[i] = U[ind+0*_nrs->fieldOffset];
  }

  // Extract strain rate at wall and multiply by viscosity to obtain stress
  platform->options.getArgs("VISCOSITY",mue);
  for (int i=0; i<_num_samples; i++) {
    int ind = _ind_wall_nodes[i];
    //shear_data[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
    shear_data[i] = 0.055;
  }

  // Concatenate inputs and outputs
  for (int i=0; i<_num_samples; i++) {
    train_data[i*num_cols] = vel_data[i];
    train_data[i*num_cols+1] = shear_data[i];
  }

  // Send training data to DB
  if (_rank == 0)
    printf("\nSending field with key %s \n",key.c_str());
  _client->put_tensor(key, train_data, {_num_samples,num_cols},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n\n");
  fflush(stdout);
}

// Run ML model for inference
void smartredis_client_t::run_wallModel(int tstep)
{
  int size_U = _nrs->fieldOffset * _nrs->mesh->dim;
  dfloat mue;
  std::string in_key = "x." + std::to_string(_rank);
  std::string out_key = "y." + std::to_string(_rank);
  dfloat *U = new dfloat[size_U]();
  dfloat *outputs = new dfloat[_num_samples]();
  dfloat *inputs = new dfloat[_num_samples]();
  dfloat *targets = new dfloat[_num_samples]();

  // Extract velocity at off-wall nodes (inputs)
  _nrs->o_U.copyTo(U, size_U);
  for (int i=0; i<_num_samples; i++) {
    int ind = _ind_owall_nodes_matched[i];
    inputs[i] = U[ind+0*_nrs->fieldOffset];
  }

  // Extract strain rate at wall and multiply by viscosity to obtain stress
  platform->options.getArgs("VISCOSITY",mue);
  for (int i=0; i<_num_samples; i++) {
    int ind = _ind_wall_nodes[i];
    //targets[i] = mue * nrs->cds->S[ind+0*nrs->fieldOffset];
    targets[i] = 0.055;
  }

  // Send input data
  if (_rank == 0)
    printf("\nSending field with key %s \n",in_key.c_str());
  _client->put_tensor(in_key, inputs, {_num_samples,1},
                    SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n\n");

  // Run ML model on input data
  if (_rank == 0)
    printf("\nRunning ML model ...\n");
  _client->run_model("model", {in_key}, {out_key});
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n\n");

  // Retrieve model pedictions
  if (_rank == 0)
    printf("\nRetrieving field with key %s \n",out_key.c_str());
  _client->unpack_tensor(out_key, outputs, {_num_samples},
                       SRTensorTypeDouble, SRMemLayoutContiguous);
  MPI_Barrier(platform->comm.mpiComm);
  if (_rank == 0)
    printf("Done\n\n");

  // Compute error in prediction
  double error = 0.0;
  for (int i=0; i<_num_samples; i++) {
    error = error + (outputs[i] - targets[i])*(outputs[i] - targets[i]);
    //printf("True, Pred, Error: %f, %f, %f \n",nrs->P[n],outputs[n],error);
  }
  error = error / _num_samples;
  printf("[%d]: Mean Squared Error in wall shear stress field = %E\n\n",_rank,error);
  fflush(stdout);
  MPI_Barrier(platform->comm.mpiComm);
}

#endif

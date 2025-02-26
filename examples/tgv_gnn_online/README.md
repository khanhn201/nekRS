# Online training of a time independent GNN surrogate model

This example demonstrates how the `gnn` and the `smartRedis` plugins can be used to create a distributed graph from the nekRS mesh and online train a GNN from a series of solution fields.
The online workflow is set up using SmartSim and SmartRedis, as in the [turbChannel_smartredis](../turbChannel_smartredis/) example.
The example flow is based off of the [Taylor-Green-Vortex flow](../tgv/README.md), however on a slightly smaller mesh. 
In this example, the model takes as inputs the three components of velocity and learns to predict the pressure field at every graph (mesh) node.
It is a time independent modeling task, since no information regarding the time dependency of the solution stepshots is given to the GNN.

Specifically, in `UDF_Setup()`, the `graph` class is instantiated from the mesh, followed by calls to `graph->gnnSetup();` and `graph->gnnWriteDB();` to setup and write the GNN input files to the SmartSim database, respectively. Here, the SmartRedis client is also initialized connecting nekRS to the SmartSim Orchestrator (the database).
In `UDF_ExecuteStep()`, the `append_dataset_to_list()` method of the SmartRedis client class is called to send the velocity and pressure fields to the database using the [DataSet](https://www.craylabs.org/docs/sr_data_structures.html#dataset) data structure. 
These keys-value pairs for the training data are tagged with the time stamp, rank ID, and job size.
For simplicity and reproducibility, nekRS is set up to send training data only at the first time step, thus only using the velocity and pressure for the initial condition to train the model, but `UDF_ExecuteStep()` can be changed to send as many time steps as desired.

## Building nekRS

To build nekRS with the GNN plugin, simply execute the build script [BuildMeOnAurora](../../BuildMeOnAurora) from the top directory
```bash
source BuildMeOnAurora
```

NOTE: you must enable building SmartRedis for this example since it is performing online training with the SmartRedis backend. 

## Runnig the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
Note that a virtual environment with SmartSim and PyTorch Geometric is needed for the GNN.
If you don't specify a virtual environment path, the script will create one for you.
From an interactive session on the compute nodes, first execute
```bash
./gen_run_script
```

taking notice of some of the variables to set. 
Specifically, make sure to set 

```
SYSTEM # the ALCF system to run on (aurora, polaris)
NEKRS_HOME # path to the nekRS install directory
VENV_PATH # path to the Python venv activate script
PROJ_ID # project name for the allocation
QUEUE # name of the queue to run on
```

The script generates the run script, which is executed with
```bash
./run.sh
```

The `run.sh` script is composed of two steps:

- First nekRS is run by itself with the `--build-only` flag. This is done such that the `.cache` directory can be built beforehand instead of during online training.
- The online training workflow driver `ssim_driver.py` is executed with Python, setting up nekRS, the GNN training, and the SmartSim Orchestrator (the database).

The outputs of the nekRS and trainer will be within the `./nekRS` directory created at runtime.

## Known Issues and Tips
- The clustered deployment requires at least 3 nodes for training and 2 nodes for inference since it deploys each component on a distinct set of nodes. Also note that the SmartSim database can be run on a single node or sharded across 3 or more nodes (sharding across 2 nodes is not allowed).

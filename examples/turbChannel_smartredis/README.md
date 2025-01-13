# Online training and inference of a wall-shear stress model from LES of a turbulent channel flow.

For the details on the channel flow LES, see the [turbChannel example](../turbChannel/README.md)

This branch of NekRS-ML includes a plugin that enables communication with a SmartSim database through the use of the SmartRedis API. 
[SmartSim](https://github.com/CrayLabs/SmartSim) and [SmartRedis](https://github.com/CrayLabs/SmartRedis) are open-source libraries developed by HPE that can be used for coupling traditional HPC applications with AI/ML functionality in situ.

This branch uses the turbChannel example to demonstrate online training and inference with SmartSim/SmartRedis and nekRS.
In particular, an MLP which takes the streamwise velocity component at some prescribed location off the wall as inputs is trained to predict the wall-shear stress at the corresponding wall node. 
This can be thought of as a crude example of using ML to train a wall-shear stress model valuable for wall-modeled LES. 
The instructions below detail how to run the example, train the MLP model from an ongoing nekRS simulation, and then perform inference with the trained model from nekRS to compare the ML predictions with the true values. 
Note that the SmartRedis plugin is called from UDF_Setup() and UDF_ExecuteStep() in the .udf file.

## Install SmartSim on Aurora

To install SmartSim on Aurora, follow the instructions below
```bash
#!/bin/bash

BASE=<path/to/venv/>

# Load the frameworks module
module load frameworks/2024.2.1_u1

# Create a Python venv for SmartSim
python -m venv --clear $BASE/_ssim_env --system-site-packages
source $BASE/_ssim_env/bin/activate
pip install --upgrade pip

# Set SmartSim build variables
export SMARTSIM_REDISAI=1.2.7

# Build SmartSim
git clone https://github.com/rickybalin/SmartSim.git
cd SmartSim
git checkout rollback_aurora
pip install -e .
# Note: disregard errors
# - intel-extension-for-tensorflow 2.15.0.0 requires absl-py==1.4.0, but you have absl-py 2.1.0 which is incompatible.
# - intel-extension-for-tensorflow 2.15.0.0 requires numpy>=1.24.0, but you have numpy 1.23.5 which is incompatible.
cd ..

# Install the CPU backend
# NB: GPU backend for RedisAI not supported on Intel PVC
cd SmartSim
export TORCH_CMAKE_PATH=$( python -c 'import torch;print(torch.utils.cmake_prefix_path)' )
export TORCH_PATH=$( python -c 'import torch; print(torch.__path__[0])' )
export LD_LIBRARY_PATH=$TORCH_PATH/lib:$LD_LIBRARY_PATH
smart build -v --device cpu --torch_dir $TORCH_CMAKE_PATH --no_tf | tee build.log
smart validate
cd ..
```

More information on SmartSim and SmartRedis on Aurora can be found on the [ALCF documentation](https://docs.alcf.anl.gov/aurora/workflows/smartsim/).

## Build nekRS with the SmartRedis plugin

To build nekRS with the SmartRedis plugin, execute the build script [BuildMeOnAurora](../../BuildMeOnAurora) from the top directory
```bash
source BuildMeOnAurora
```

## Runnig the example

Scripts are provided to conveniently generate run scripts and config files for the workflow on the different ALCF systems.
From an interactive session on the compute nodes, first execute
```bash
./gen_run_script
```

taking notice of some of the variables to set. 
Specifically, make sure to set 

```
SYSTEM # the ALCF system to run on (aurora, polaris)
DEPLOYMENT # the deployment strategy for the workflow (colocated, clustered)
ML_TASK # the ML task to run (train, inference)
NEKRS_HOME # path to the nekRS install directory
VENV_PATH # path to the Python venv activate script where the SmartSim venv was built (see above)
PROJ_ID # project name for the allocation
QUEUE # name of the queue to run on
```

The script generate the config file for the workflow (`ssim_config.yaml`) and the run script to execute with
```bash
./run.sh
```

The outputs of the nekRS and trainer will be within the `./nekRS` directory created at runtime.

## Known Issues and Tips
- The clustered deployment requires at least 3 nodes for training and 2 nodes for inference since it deploys each component on a distinct set of nodes. Also note that the SmartSim database can be run on a single node or sharded across 3 or more nodes (sharding across 2 nodes is not allowed). 
- On Aurora, inference can only be run on the CPU. nekRS will still run on the GPU, but model inference will be executed on the host. This is due to a limitation of RedisAI on Intel hardware.

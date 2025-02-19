# general imports
import os
import sys 
from omegaconf import DictConfig, OmegaConf
import hydra

# smartsim and smartredis imports
from smartsim import Experiment
from smartsim.settings import RunSettings, PalsMpiexecSettings


class ShootingWorkflow():
    """Class for the solution shooting workflow alternating between 
    fine-tuning a surrogate from an ongoing simulation and deploying
    the surrogate to shoot the solution forward
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.nodelist = []
        self.num_nodes = 1
        self.sim_nodes = ''
        self.db_nodes = ''
        self.dbNodes_list = []
        self.train_nodes = ''
        self.inference_nodes = ''
        self.fine_tune_iter = -1

        # Parse the node list from the scheduler
        self.parseNodeList()

        # Split the nodes between the components
        self.assignNodes()

        # Initialize the SmartSim experiment
        self.db = None
        self.nekrs_model = None
        self.train_model = None
        self.port = cfg.database.port
        self.exp = Experiment(self.cfg.database.exp_name, launcher=self.cfg.database.launcher)

    def parseNodeList(self) -> None:
        """Parse the nodelist from the scheduler
        """
        if (self.cfg.database.launcher=='pals'):
            hostfile = os.getenv('PBS_NODEFILE')
            with open(hostfile) as file:
                self.nodelist = file.readlines()
                self.nodelist = [line.rstrip() for line in self.nodelist]
                self.nodelist = [line.split('.')[0] for line in self.nodelist]
        else:
            sys.exit('Only the pals launcher is implemented')
        self.num_nodes = len(self.nodelist)

    def assignNodes(self) -> None:
        """Assign the total nodes of the job to the different components
        """
        if (self.cfg.database.deployment=='clustered'):
            self.sim_nodes = ','.join(self.nodelist[0: self.cfg.run_args.sim_nodes])
            self.db_nodes = ','.join(self.nodelist[self.cfg.run_args.sim_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes])
            self.dbNodes_list = self.nodelist[self.cfg.run_args.sim_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes]
            self.train_nodes = ','.join(self.nodelist[self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes: \
                                self.cfg.run_args.sim_nodes + self.cfg.run_args.db_nodes + \
                                self.cfg.run_args.ml_nodes])
        else:
            self.sim_nodes = ','.join(self.nodelist)
            self.db_nodes = self.sim_nodes
            self.train_nodes = self.sim_nodes
        print(f"Database running on {self.cfg.run_args.db_nodes} nodes:")
        print(self.db_nodes)
        print(f"Simulatiom running on {self.cfg.run_args.sim_nodes} nodes:")
        print(self.sim_nodes)
        print(f"Training running on {self.cfg.run_args.ml_nodes} nodes:")
        print(self.train_nodes,flush=True)

    def launchClusteredDB(self) -> None:
        """Launch the clustered SmartSim Orchestrator
        """
        runArgs = {"np": 1, "ppn": 1, "cpu-bind": "numa"}
        kwargs = {
            'maxclients': 100000,
            'threads_per_queue': 4, # set to 4 for improved performance
            'inter_op_parallelism': 1,
            'intra_op_parallelism': 4,
            'cluster-node-timeout': 30000,
        }
        if (self.cfg.database.launcher=='pals'): run_command = 'mpiexec'
        network = self.cfg.database.network_interface if type(self.cfg.database.network_interface)==str \
                                             else OmegaConf.to_object(self.cfg.database.network_interface)
        self.db = self.exp.create_database(port=self.port,
                                      batch=False,
                                      db_nodes=self.cfg.run_args.db_nodes,
                                      run_command=run_command,
                                      interface=network,
                                      hosts=self.dbNodes_list,
                                      run_args=runArgs,
                                      single_cmd=True,
                                      **kwargs
        )
        self.exp.generate(self.db)
        print("\nStarting database ...")
        self.exp.start(self.db)
        print("Done\n", flush=True)

    def stopClusteredDB(self) -> None:
        """Stop the clustered SmartSim Orchestrator
        """
        print("Stopping the Orchestrator ...")
        self.exp.stop(self.db)
        print("Done\n", flush=True)

    def launchNekRS(self) -> None:
        """Launch the nekRS simulation
        """
        client_exe = self.cfg.sim.executable
        nrs_settings = PalsMpiexecSettings(client_exe,
                                           exe_args=None,
                                           run_args=None,
                                           env_vars=None
        )
        nrs_settings.set_tasks(self.cfg.run_args.simprocs)
        nrs_settings.set_tasks_per_node(self.cfg.run_args.simprocs_pn)
        nrs_settings.set_hostlist(self.sim_nodes)
        nrs_settings.set_cpu_binding_type(self.cfg.run_args.sim_cpu_bind)
        nrs_settings.add_exe_args(self.cfg.sim.arguments)
        if (self.cfg.sim.affinity):
            nrs_settings.set_gpu_affinity_script(self.cfg.sim.affinity,
                                                 self.cfg.run_args.simprocs_pn)
        
        self.nekrs_model = self.exp.create_model(f"nekrs_{self.fine_tune_iter}", nrs_settings)
        if (self.cfg.database.deployment=='colocated'):
            kwargs = {
                'maxclients': 100000,
                'threads_per_queue': 4, # set to 4 for improved performance
                'inter_op_parallelism': 1,
                'intra_op_parallelism': 1,
                'cluster-node-timeout': 30000,
                }
            db_bind = None if self.cfg.run_args.db_cpu_bind=='None' else self.cfg.run_args.db_cpu_bind
            if (self.cfg.database.network_interface=='uds'):
                self.nekrs_model.colocate_db_uds(
                        db_cpus=self.cfg.run_args.dbprocs_pn,
                        custom_pinning=db_bind,
                        debug=False,
                        **kwargs
                )
            else:
                self.nekrs_model.colocate_db_tcp(
                        port=self.port,
                        ifname=self.cfg.database.network_interface,
                        db_cpus=self.cfg.run_args.dbprocs_pn,
                        custom_pinning=db_bind,
                        debug=False,
                        **kwargs
                )
        
        print("Launching the NekRS ...")
        if len(self.cfg.sim.copy_files)>0 or len(self.cfg.sim.link_files)>0:
            self.nekrs_model.attach_generator_files(
                to_copy=list(self.cfg.sim.copy_files), 
                to_symlink=list(self.cfg.sim.link_files)
            )
        self.exp.generate(self.nekrs_model, overwrite=True)
        self.exp.start(self.nekrs_model, block=False, summary=False)
        print("Done\n", flush=True)

    def launchTrainer(self) -> None:
        """Launch the GNN trainer
        """
        env_vars = None
        if (self.cfg.database.deployment=='colocated'):
            SSDB = self.nekrs_model.run_settings.env_vars['SSDB']
            env_vars = {'SSDB': SSDB}
        ml_exe = self.cfg.train.executable
        ml_exe = ml_exe + f' --dbnodes={self.cfg.run_args.db_nodes}' \
                        + f' --device={self.cfg.train.device}' \
                        + f' --ppn={self.cfg.run_args.mlprocs_pn}' \
                        + f' --logging={self.cfg.train.logging}'
        ml_settings = PalsMpiexecSettings(
                           'python',
                           exe_args=ml_exe,
                           run_args=None,
                           env_vars=env_vars,
        )
        ml_settings.set_tasks(self.cfg.run_args.mlprocs)
        ml_settings.set_tasks_per_node(self.cfg.run_args.mlprocs_pn)
        ml_settings.set_hostlist(self.train_nodes)
        ml_settings.set_cpu_binding_type(self.cfg.run_args.ml_cpu_bind)
        if (self.cfg.train.affinity):
            skip = 0 if self.cfg.database.deployment=='clustered' else self.cfg.run_args.simprocs_pn
            ml_settings.set_gpu_affinity_script(self.cfg.train.affinity,
                                                self.cfg.run_args.mlprocs_pn,
                                                skip
            )

        print("Launching training script ... ")
        self.train_model = self.exp.create_model(f"train_{self.fine_tune_iter}", ml_settings)
        if len(self.cfg.train.copy_files)>0 or len(self.cfg.train.link_files)>0:
            self.train_model.attach_generator_files(to_copy=list(self.cfg.train.copy_files), 
                                            to_symlink=list(self.cfg.train.link_files)
            )
        self.exp.generate(self.train_model, overwrite=True)
        self.exp.start(self.train_model, block=True, summary=False)
        print("Done\n", flush=True)

    def fineTune(self) -> None:
        """Fine-tune the GNN model from the nekRS simulation
        """
        self.fine_tune_iter += 1
        self.launchNekRS()
        self.launchTrainer() # blocks code progress

    def runner(self) -> None:
        """Runner function for the workflow responsible for alternating
        between fine-tuning and inference and deploying the components
        """
        # Launch clustered DB
        if (self.cfg.database.deployment=='clustered'):
            self.launchClusteredDB()

        # Start the workflow loop
        while True:
            # Fine-tune model
            self.fineTune()
            break
        
        # Stop clustered DB
        if (self.cfg.database.deployment=='clustered'):
            self.stopClusteredDB()


## Main function
@hydra.main(version_base=None, config_path="./", config_name="ssim_config")
def main(cfg: DictConfig):
    # Initialize workflow class
    workflow = ShootingWorkflow(cfg)

    # Run the workflow
    workflow.runner()

    # Quit
    print("Quitting")


## Run main
if __name__ == "__main__":
    main()

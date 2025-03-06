import os
import sys
from typing import Optional, Union
from omegaconf import DictConfig
import numpy as np
from time import sleep, perf_counter

# Import SmartRedis
try:
    from smartredis import Client, Dataset
except ModuleNotFoundError:
    pass

# Import ADIOS2
try:
    from adios2 import Stream, Adios, bindings
except ModuleNotFoundError:
    pass

class OnlineClient:
    """Class for the online training client
    """
    def __init__(self, cfg: DictConfig, comm) -> None:
        self.backend = cfg.client.backend
        self.db_nodes = cfg.client.db_nodes
        self.client = None
        self.comm = comm

        # initialize the client backend
        clients = ['smartredis', 'adios']
        if self.cfg.client not in clients:
            sys.exit(f'Client {self.cfg.client} not implemented. '\
                     f'Available options are: {clients}')
        self.init_client()

    def init_client(self) -> None:
        """Initialize the client based on the specified backend
        """
        if self.backend == 'smartredis':
            SSDB = os.getenv('SSDB')
            if (self.db_nodes==1):
                self.client = Client(address=SSDB,cluster=False)
            else:
                self.client = Client(address=SSDB,cluster=True)
        elif self.backend == 'adios':
            # Initialize ADIOS MPI Communicator
            adios = Adios(self.comm)
            self.client = adios.declare_io('nekRS-ML')
            self.client.set_engine(self.cfg.client.adios_engine)
            if self.cfg.client.adios_stream == 'sync':
                parameters = {
                    'RendezvousReaderCount': '1', # producer waits for consumer in Open()
                    'QueueFullPolicy': 'Block', # wait for consumer to get every step
                    'QueueLimit': '1', # only buffer one step
                }
            elif self.cfg.client.adios_stream == 'async': 
                parameters = {
                    'RendezvousReaderCount': '0', # producer does not wait for consumer in Open()
                    'QueueFullPolicy': 'Block', # slow consumer misses out on steps
                    'QueueLimit': '3', # buffer first step
                }
            parameters['DataTransport'] = self.cfg.client.adios_transport # options: MPI, WAN, UCX, RDMA
            parameters['OpenTimeoutSecs'] = '600' # number of seconds producer waits on Open() for consumer
            self.client.set_parameters(parameters)

    def file_exists(self, file_name: str) -> bool:
        """Check if a file (or key) exists
        """
        if self.backend == 'smartredis':
            return self.client.key_exists(file_name)

    def get_array(self, file_name: Union[str, Dataset]) -> np.ndarray:
        """Get an array frpm staging area / simulation
        """
        tic = perf_counter()
        if self.backend == 'smartredis':
            if isinstance(file_name, str):
                while True:
                    if self.file_exists(file_name):
                        array = self.client.get_tensor(file_name)
                        break
                    else:
                        sleep(0.5)
                        t_elapsed = perf_counter() - tic
                        if t_elapsed > 300:
                            sys.exit(f'Could not find {file_name} in DB')
            else:
                array = file_name.get_tensor('data')
        return array
    
    def put_array(self, file_name: str, array: np.ndarray) -> None:
        """Put/send an array to staging area / simulation
        """
        if self.backend == 'smartredis':
            self.client.put_tensor(file_name, array)

    def get_file_list(self, list_name: str) -> list:
        """Get the list of files to read
        """
        if self.backend == 'smartredis':
            # Ensure the list of DataSets is available
            while True:
                list_length = self.client.get_list_length(list_name)
                if list_length == 0:
                    sleep(1)
                    continue
                else:
                    break
            
            # Grab list of datasets
            file_list = self.client.get_datasets_from_list(list_name)
        return file_list
    
    def get_file_list_length(self, list_name: str) -> int:
        """Get the length of the file list
        """
        if self.backend == 'smartredis':
            return self.client.get_list_length(list_name)


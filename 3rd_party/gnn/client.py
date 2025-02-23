import os
from typing import Optional, Union
from omegaconf import DictConfig
import numpy as np
from time import sleep

from smartredis import Client, Dataset

class OnlineClient:
    """Class for the online training client
    """
    def __init__(self, cfg: DictConfig) -> None:
        self.backend = cfg.client.backend
        self.db_nodes = cfg.client.db_nodes
        self.client = None

        # initialize the client backend
        self.init_client()

    def init_client(self) -> None:
        """Initialize the client based on the specified backend
        """
        if self.backend == "smartredis":
            SSDB = os.getenv('SSDB')
            if (self.db_nodes==1):
                self.client = Client(address=SSDB,cluster=False)
            else:
                self.client = Client(address=SSDB,cluster=True)

    def get_array(self, file_name: Union[str, Dataset]) -> np.ndarray:
        """Get an array frpm staging area / simulation
        """
        if self.backend == "smartredis":
            if isinstance(file_name, str):
                array = self.client.get_tensor(file_name)
            else:
                array = file_name.get_tensor('data')
        return array
    
    def put_array(self, file_name: str, array: np.ndarray) -> None:
        """Put/send an array to staging area / simulation
        """
        if self.backend == "smartredis":
            self.client.put_tensor(file_name, array)

    def get_file_list(self, list_name: str) -> list:
        """Get the list of files to read
        """
        if self.backend == "smartredis":
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

import os
from typing import Optional, Union
from omegaconf import DictConfig
import numpy as np

from smartredis import Client

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

    def get_array(self, file_name: str) -> np.ndarray:
        """Get an array frpm staging area / simulation
        """
        if self.backend == "smartredis":
            array = self.client.get_tensor(file_name)
        return array
    
    def put_array(self, file_name: str, array: np.ndarray) -> None:
        """Put/send an array to staging area / simulation
        """
        if self.backend == "smartredis":
            self.client.put_tensor(file_name, array)


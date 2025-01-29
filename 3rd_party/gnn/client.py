import os
from typing import Optional, Union
from omegaconf import DictConfig

from smartredis import Client

class OnlineClient:
    def __init__(self, cfg: DictConfig):
        self.backend = cfg.client.backend
        self.db_nodes = cfg.client.db_nodes
        self.client = None

        # initialize the client backend
        self.init_client()

    def init_client(self):
        if self.backend == "smartredis":
            address = os.getenv('SSDB')
            if (self.db_nodes==1):
                self.client = Client(address=SSDB,cluster=False)
            else:
                self.client = Client(address=SSDB,cluster=True)



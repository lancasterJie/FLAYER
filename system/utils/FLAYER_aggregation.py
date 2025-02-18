import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class LocalAggregation:
    def __init__(self, layer_idx: int = 2) -> None:
        """

        Args:
            layer_idx: Control the local aggregation weight range. Default: 2
        Returns:
            None.
        """

        self.layer_idx = layer_idx

    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module,
                            acc: float) -> None:
        """
        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        for param, param_g in zip(params[-self.layer_idx:], params_g[-self.layer_idx:]):
            param.data = acc * param.data + (1 - acc) * param_g.data


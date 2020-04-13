from torch.utils.tensorboard import SummaryWriter
from typing import Union, Optional, Any
import numpy as np
import torch
import os
import time
from flare.kindling.mpi_tools import proc_id
from flare.kindling import utils


class TensorBoardWriter:
    def __init__(self, fpath: Optional[str] = "flare_runs"):
        if fpath is None:
            self.fpath = "flare_runs/run_at_time_" + str(time.time())
        else:
            self.fpath = fpath

        if os.path.exists(self.fpath):
           print(utils.colorize(f"Warning path at {self.fpath} already exists, storing info there anyway.", "yellow"))

        self.full_logdir = self.fpath

        if proc_id() == 0:
            os.makedirs(self.fpath, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.fpath, flush_secs=30)
            print(utils.colorize(f"TensorBoard Logdir: {self.full_logdir}", "green"))


    def add_plot(
        self, key: str, val: Union[torch.Tensor, np.array, list, float, int], step: int
    ):
        if proc_id() == 0:
            if isinstance(val, torch.Tensor):
                val = val.item()
                self.writer.add_scalar(key, val, global_step=step)
            elif isinstance(val, list):
                if len(val) == 1:
                    val = np.float(val)
                    self.writer.add_scalar(key, val, global_step=step)
            else:
                self.writer.add_scalar(key, val, global_step=step)

    def add_hist(
        self,
        key: str,
        val: Union[torch.Tensor, list, np.array, tuple],
        step: int,
        bins: Optional[str] = "tensorflow",
    ):
        if proc_id() == 0:
            if isinstance(val, torch.Tensor):
                val = val.item()
            val = np.array(val)
            if len(val) > 0:
                self.writer.add_histogram(key, val, global_step=step, bins=bins)

    def add_vals(self, val_dict: dict, step: int):
        for k in val_dict.keys():
            val = val_dict[k]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            else:
                val = np.array(val_dict[k])
            if "Loss" in k:
                k = "Loss/" + k
            elif "Ep" in k:
                k = "Episode/" + k
            elif "Values" in k:
                k = "Values/" + k
            elif "Env" in k:
                k = "Env/" + k
            if len(val) == 1:
                self.add_plot(k, val, step)
            elif len(val) > 1:
                self.add_hist(k, val, step)

    def end(self):
        if proc_id() == 0:
            self.writer.flush()

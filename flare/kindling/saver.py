from pathlib import Path
import time
import pickle as pkl
from typing import Optional
import os
from flare.kindling.mpi_tools import proc_id


class Saver:
    r"""
    A class to pickle generic Python objects saved over a model's training cycle.

    Args:
        out_dir (str): Directory to save to.
        keys (list): Keys to initialize the save dictionary with.
    """
    def __init__(self, out_dir: str, keys: Optional[list] = []):
        self.out_path = Path(out_dir)
        os.makedirs(self.out_path, exist_ok=True)
        self.saver_dict = {k: [] for k in keys} if len(keys) > 0 else {}

    def store(self, **kwargs):
        """Store input kwargs in save dictionary."""
        for k, v in kwargs.items():
            if k not in self.saver_dict.keys():
                self.saver_dict[k] = [v]
            else:
                self.saver_dict[k].append(v)

    def save(self):
        """Write save dictionary to .pkl file."""
        if proc_id() == 0:
            ct = time.time()
            if len(self.saver_dict) > 0:
                pkl.dump(
                    self.saver_dict,
                    open(
                        self.out_path / f"env_states_and_screens_saved_on{ct}.pkl", "wb"
                    ),
                )

from pathlib import Path
import time
import pickle as pkl
from typing import Optional
import os
from flare.kindling.mpi_tools import proc_id


class Saver:
    def __init__(self, out_dir, keys: Optional[list] = []):
        self.out_path = Path(out_dir)
        os.makedirs(self.out_path, exist_ok=True)
        self.saver_dict = {k: [] for k in keys} if len(keys) > 0 else {}

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.saver_dict.keys():
                self.saver_dict[k] = [v]
            else:
                self.saver_dict[k].append(v)

    def save(self):
        if proc_id() == 0:
            ct = time.time()
            if len(self.saver_dict) > 0:
                pkl.dump(
                    self.saver_dict,
                    open(
                        self.out_path / f"env_states_and_screens_saved_on{ct}.pkl", "wb"
                    ),
                )

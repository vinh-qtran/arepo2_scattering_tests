import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

import numpy as np

from scipy.stats import maxwell

from utils import ICs_writer

class UniformICs(ICs_writer.ICsWriter):
    def __init__(self, 
                 box_size, density,
                 part_mass, part_veloc,
                 MB_distributed=False,
                 seed=42):
        self.box_size = box_size
        self.density = density
        self.part_mass = part_mass

        self.N_part = int((box_size ** 3) * density / part_mass)

        self.part_veloc = part_veloc
        self.MB_distributed = MB_distributed

        self.seed = seed
        np.random.seed(seed)

        part_coords = self._get_part_coords()
        part_velocs = self._get_part_velocs()
        part_ids = np.arange(self.N_part, dtype=np.int64)
        part_masses = np.full(self.N_part, part_mass, dtype=np.float64)

        super().__init__(
            box_size = self.box_size,
            part_coords = part_coords,
            part_velocs = part_velocs,
            part_ids = part_ids,
            part_masses = part_masses
        )

    def _get_part_coords(self):
        part_coords = np.random.uniform(
            0, self.box_size, size=(self.N_part, 3)
        ).astype(np.float64)
        return part_coords
    
    def _get_part_velocs(self):
        if self.MB_distributed:
            _part_speeds = maxwell.ppf(
                np.random.uniform(0.0, 1.0, size=self.N_part).astype(np.float64), scale=self.part_veloc / np.sqrt(8/np.pi)
            )
        else:
            _part_speeds = np.full(self.N_part, self.part_veloc, dtype=np.float64)
        _part_thetas = np.arccos(1 - 2 * np.random.uniform(0.0, 1.0, size=self.N_part)).astype(np.float64)
        _part_phis = 2 * np.pi * np.random.uniform(0.0, 1.0, size=self.N_part).astype(np.float64)

        part_velocs = np.zeros((self.N_part, 3), dtype=np.float64)
        part_velocs[:, 0] = _part_speeds * np.sin(_part_thetas) * np.cos(_part_phis)
        part_velocs[:, 1] = _part_speeds * np.sin(_part_thetas) * np.sin(_part_phis)
        part_velocs[:, 2] = _part_speeds * np.cos(_part_thetas)

        return part_velocs
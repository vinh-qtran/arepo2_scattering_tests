import os
import sys
import subprocess
from pathlib import Path

repo_root = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
).stdout.strip()

sys.path.append(repo_root)

import numpy as np

from utils import ICs_writer

class SphereICs(ICs_writer.ICsWriter):
    def __init__(self, 
                 bg_size, bg_density,
                 sph_radius, sph_N_part, sph_veloc,
                 seed=42):
        self.bg_size = bg_size
        self.bg_density = bg_density
        self.bg_N_part = int((bg_size ** 3) * bg_density)

        self.sph_radius = sph_radius
        self.sph_N_part = sph_N_part
        self.sph_veloc = sph_veloc

        self.seed = seed
        np.random.seed(seed)

        self.box_size = bg_size * 2

        part_coords, part_velocs, part_ids, part_masses = self._get_ICs()

        super().__init__(
            box_size = self.box_size,
            part_coords = part_coords,
            part_velocs = part_velocs,
            part_ids = part_ids,
            part_masses = part_masses
        )

    def _get_background(self):
        bg_part_coords = np.random.uniform(
            0, self.bg_size, size=(self.bg_N_part, 3)
        ).astype(np.float64)

        bg_part_velocs = np.zeros((self.bg_N_part, 3), dtype=np.float64)
        return bg_part_coords, bg_part_velocs
    
    def _get_sphere(self):
        _sph_center = np.array([self.bg_size/2, self.bg_size/2, self.sph_radius], dtype=np.float64)

        _sph_part_radii = self.sph_radius * np.cbrt(np.random.uniform(0.0, 1.0, size=self.sph_N_part)).astype(np.float64)
        _sph_part_thetas = np.arccos(1 - 2 * np.random.uniform(0.0, 1.0, size=self.sph_N_part)).astype(np.float64)
        _sph_part_phis = 2 * np.pi * np.random.uniform(0.0, 1.0, size=self.sph_N_part).astype(np.float64)

        sph_part_coords = np.zeros((self.sph_N_part, 3), dtype=np.float64)
        sph_part_coords[:, 0] = _sph_center[0] + _sph_part_radii * np.sin(_sph_part_thetas) * np.cos(_sph_part_phis)
        sph_part_coords[:, 1] = _sph_center[1] + _sph_part_radii * np.sin(_sph_part_thetas) * np.sin(_sph_part_phis)
        sph_part_coords[:, 2] = _sph_center[2] + _sph_part_radii * np.cos(_sph_part_thetas)

        sph_part_velocs = np.zeros((self.sph_N_part, 3), dtype=np.float64)
        sph_part_velocs[:, 2] = self.sph_veloc

        return sph_part_coords, sph_part_velocs
    
    def _get_ICs(self):
        _bg_part_coords, _bg_part_velocs = self._get_background()
        _sph_part_coords, _sph_part_velocs = self._get_sphere()
        part_coords = np.vstack((_bg_part_coords, _sph_part_coords)).astype(np.float64)
        part_velocs = np.vstack((_bg_part_velocs, _sph_part_velocs)).astype(np.float64)

        part_ids = np.arange(0, part_coords.shape[0], 1, dtype=np.int32)
        part_masses = np.ones(part_coords.shape[0], dtype=np.float64)

        return part_coords, part_velocs, part_ids, part_masses
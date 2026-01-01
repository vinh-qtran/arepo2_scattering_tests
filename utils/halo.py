import numpy as np
import h5py

import matplotlib.pyplot as plt

class HaloProjection:
    def __init__(self, 
                 halo_file,
                 halo_center=np.array([0.0, 0.0, 0.0])):
        _data = h5py.File(halo_file, 'r')
        self.part_coord = np.array(_data['PartType1']['Coordinates']) - halo_center
        self.mass_scaler = _data['Header'].attrs['MassTable'][1]
        if self.mass_scaler == 0:
            self.mass_scaler = _data['PartType1']['Masses'][0]

    def _get_2d_projection(self, x_bins, y_bins, axis):
        x_axis = (axis + 1) % 3
        y_axis = (axis + 2) % 3

        x_proj = self.part_coord[:,x_axis]
        y_proj = self.part_coord[:,y_axis]

        counts, _, _ = np.histogram2d(x_proj, y_proj, bins=[x_bins, y_bins])

        column_density = counts * self.mass_scaler / ((x_bins[1]-x_bins[0]) * (y_bins[1]-y_bins[0]))

        return column_density

    def show_2d_projection(self, 
                           fig, ax, 
                           box_size, num_bins=201,
                           axis=2,
                           log_scale=False,
                           vmin=None, vmax=None):

        x_bins = np.linspace(- box_size/2, box_size/2, num_bins)
        y_bins = np.linspace(- box_size/2, box_size/2, num_bins)

        column_density = self._get_2d_projection(x_bins, y_bins, axis)
        if log_scale:
            column_density = np.log10(column_density)

        c = ax.pcolormesh(
            x_bins, y_bins,
            column_density.T,
            cmap='RdBu',
            vmin=vmin, vmax=vmax,
            shading='auto'
        )
        cb = fig.colorbar(c, ax=ax)

        ax.set_xlim(- box_size/2, box_size/2)
        ax.set_ylim(- box_size/2, box_size/2)
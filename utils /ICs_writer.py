import numpy as np
import h5py

class ICsWriter:
    def __init__(self, 
                 filename, 
                 box_size, 
                 part_coords, 
                 part_velocs,
                 part_ids,
                 part_masses):
        self._filename = filename
        self._box_size = box_size

        self._part_coords = part_coords
        self._part_velocs = part_velocs
        self._part_ids = part_ids
        self._part_masses = part_masses

        self._N_parts = self._get_N_parts()

    def _get_N_parts(self):
        N_parts = self._part_coords.shape[0]
        
        for _arr in [self._part_velocs, self._part_ids, self._part_masses]:
            if _arr.shape[0] != N_parts:
                raise ValueError("All particle arrays must have the same length.")
        
        return N_parts

    def _write_header(self, ICs):
        _header = ICs.create_group("Header")

        _NumPart = np.array([0, self._N_parts, 0, 0, 0, 0], dtype = np.int32)

        _header.attrs.create("NumPart_ThisFile", _NumPart)
        _header.attrs.create("NumPart_Total", _NumPart)
        _header.attrs.create("NumPart_Total_HighWord", np.zeros(6, dtype = np.int32) )
        _header.attrs.create("MassTable", np.zeros(6, dtype = np.int32) )

        _header.attrs.create("Time", 0.0)
        _header.attrs.create("Redshift", 0.0)
        _header.attrs.create("BoxSize", np.float64(self._box_size))
        _header.attrs.create("NumFilesPerSnapshot", 1)

        _header.attrs.create("Omega0", 0.0)
        _header.attrs.create("OmegaB", 0.0)
        _header.attrs.create("OmegaLambda", 0.0)
        _header.attrs.create("HubbleParam", 1.0)

        _header.attrs.create("Flag_Sfr", 0)
        _header.attrs.create("Flag_Cooling", 0)
        _header.attrs.create("Flag_StellarAge", 0)
        _header.attrs.create("Flag_Metals", 0)
        _header.attrs.create("Flag_Feedback", 0)

        _header.attrs.create("Flag_DoublePrecision", 1)

    def _write_particles(self, ICs):
        _part1 = ICs.create_group("PartType1")

        _part1.create_dataset("ParticleIDs", data = self._part_ids)
        _part1.create_dataset("Coordinates", data = self._part_coords)
        _part1.create_dataset("Masses", data = self._part_masses)
        _part1.create_dataset("Velocities", data = self._part_velocs)

    def write(self):
        with h5py.File(self._filename, "w") as ICs:
            self._write_header(ICs)
            self._write_particles(ICs)
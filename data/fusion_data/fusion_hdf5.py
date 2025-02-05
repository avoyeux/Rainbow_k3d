"""
To add both the real and fake data together in the same HDF5 file.
"""

# IMPORTS
import os
import h5py

# IMPPORTs personal
from common import root_path


class fusionHdf5:

    def __init__(self) -> None:
        pass

    def paths_setup(self) -> dict[str, str]:

        paths= {
            'codes': root_path,

        }

        return paths
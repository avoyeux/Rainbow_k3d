"""
To create the HDF5 data files for the cubes.
"""

# IMPORTS
import os
import re
import h5py
import sparse

import numpy as np

from scipy.io import readsav
from datetime import datetime



class Cubes:
    """
    To create cubes with feet and without feet in an HDF5 format.
    """

    def __init__(self, filename: str) -> None:

        # Initial attributes
        self.filename = filename

        # Constants
        self.solar_r = 6.96e5

        # Attributes setup
        self.dx: float
        self.x_min: float
        self.y_min: float
        self.z_min: float

        # Created attributes
        self.paths = self.setup_path()
        self.cubes = self.fetch_cubes()

        # Run
        self.create()

    def setup_path(self) -> dict[str, str]:

        main = '/home/avoyeux/Documents/avoyeux/python_codes'
        paths = {
            'main': main,
            'cubes': os.path.join(main, '..', 'Cubes_karine'),
            'save': os.path.join(main, 'Data'),
        }
        return paths
    
    def create(self) -> None:
        """
        Creates the HDF5 file.
        """

        with h5py.File(os.path.join(self.paths['save'],self.filename), 'w') as H5PYFile:

            # Main metadata
            file_info = self.foundation()
            H5PYFile = self.add_dataset(H5PYFile, file_info)

            # Raw metadata
            raw_info = self.init_group()
            H5PYFile = self.add_group(H5PYFile, raw_info, 'InitCubes')

    def foundation(self) -> dict[str, str]:
        """
        For the main information before getting to the HDF5 datasets and groups.

        Returns:
            dict[str, str]: the main information.
        """

        description = ("Contains the data cubes for the Solar Rainbow event gotten from the intersection of masks gotten from SDO and STEREO images. The SDO masks were created from an automatic "
        "code created by Dr. Elie Soubrie, while the STEREO masks where manually created by Dr. Karine [...] by visual interpretation of the [...] nm monochromatic STEREO [...] images.\n"
        "New values for the feet where added to help for a curve fitting of the filament. These were added by looking at the STEREO [...] nm images as the ends of the filament are more visible. "
        "Hence, the feet are not actually visible in the initial masks.\n"
        "The data:\n" \
        + (" " * 4) +  "- the voxel values were coded in bits so that they can be easily separated in different categories. These are")
        # TODO: need to finish the explanation here and also explain that the data is saved as sparse arrays.

        info = {
            'author': 'Voyeux Alfred',
            'creationDate': datetime.now().isoformat(),
            'filename': self.filename,
            'description': description,
        }
        return info
    
    def add_group(self, group: h5py.File | h5py.Group, info: dict[str, str | dict[str, any]], name: str) -> h5py.File | h5py.Group:
        """
        Adds a group with the corresponding DataSets to a HDF5 group like object.

        Args:
            group (h5py.File | h5py.Group): the HDF5 group like object.
            info (dict[str, str  |  dict[str  |  any]]): the information  and data to add in the group.
            name (str): the name of the group.

        Returns:
            h5py.File | h5py.Group: the input group like object with the added group.
        """

        new_group = group.create_group(name)
        for key, item in info.items():
            if isinstance(item, str): 
                new_group.attrs[key] = item
            elif 'data' in item.keys():
                new_group = self.add_dataset(new_group, item, key)
            else:
                new_group = self.add_group(new_group, item, key)
        return group
    
    def add_dataset(self, group: h5py.File | h5py.Group, info: dict[str, any], name: str = '') -> h5py.File | h5py.Group:
        """
        Adds a DataSet to a HDF5 group like object.

        Args:
            group (h5py.File | h5py.Group): the HDF5 group like object.
            info (dict[str, any]): the information to add in the DataSet.
            name (str, optional): the name of the DataSet to add. Defaults to '' (then the attributes are directly added to the group).

        Returns:
            h5py.File | h5py.Group: the input group like object with the added DataSet.
        """
        
        dataset = group.create_dataset(name, data=info['data']) if name != '' else group
        for key, item in info.items():
            if key=='data': continue
            dataset.attrs[key] = item 
        return group
    
    def fetch_cubes(self) -> sparse.COO:
        # Find files
        pattern = re.compile(r'cube(\d{3})\.save')
        names = [name for name in os.listdir(self.paths['cubes']) if pattern.match(name)]

        # Opening the first cube
        cube = readsav(os.path.join(self.paths['cubes'], names[0]))
        self.dx = cube.dx
        self.x_min = cube.xt_min
        self.y_min = cube.yt_min
        self.z_min = cube.zt_min

        # Opening the cubes
        cubes = np.stack([readsav(os.path.join(self.paths['cubes'], name)).cube.astype('uint8') for name in names], axis=0)
        cubes = np.transpose(cubes, (0, 3, 2, 1)) # as readsav reads the axes in a different order.
        cubes = self.sparse_data(cubes)
        return cubes
    
    # def reformat_string(self, text: str) -> str:
    #     """
    #     To reformat a string so that 'verbose' strings in the code look better when printed.
    #     The reformatting works like so: single linebreaks are taken out while double linebreaks are converted to single linebreaks. 

    #     Args:
    #         text (str): initial string to be reformatted for printing and/or saving.

    #     Returns:
    #         str: the reformatted print. 
    #     """

    #     return '\n'.join([line.replace('\n', ' ') for line in text.split('\n\n')])

    def init_group(self) -> dict[str, str | dict[str, any]]:
        
        description = ("The filament voxels in sparse COO format (i.e. with a coords and values arrays) of the initial cubes gotten from Dr. Karine [...]'s work."
        "Furthermore, the necessary information to be able to position the filament relative to the Sun are also available.\n Both cubes, with or without feet, are inside this group.")

        raw = {
            'description': "The raw initial data without the feet for the interpolation.",
            'coords': {
                'data': self.cubes.coords,
                'unit': 'none',
                'description': "The index coordinates of the initial voxels.\nTo get the positions with respect to the Sun, then dx, x_min, y_min, z_min need to be taken into account.",
            },
            'value': {
                'data': self.cubes.data, 
                'unit': 'none',
                'description': "The values for each voxel.",
            },
        }

        with_feet = {
            'description': "The raw initial data with the added feet for the interpolation.",
            'coords': {
                'data': None, # TODO: need to decide how to add the feet data
                'unit': 'none',
                'description': "The index coordinates of the initial voxels.\nTo get the positions with respect to the Sun, then dx, x_min, y_min, z_min need to be taken into account.",
            },
            'value': {
                'data': None, #TODO: need to create that data
                'unit': 'none',
                'description': 'The values for each pixel',
            },
        }

        dx = {
            'data': self.dx,
            'unit': 'km',
            'description': "The length of each voxel.",
        }
        x_min = {
            'data': self.x_min,
            'unit': 'km',
            'description': 'The '
        } # TODO: need to check what the x, y, z axis are actually called to be able to fill up the description.
        y_min = {
            'data': self.y_min,
            'unit': 'km',
            'description': "The kjkjfd",
        }
        z_min = {
            'data': self.z_min,
            'unit': 'km',
            'description': "The klkkllkl",
        }
        info = {
            'description': description,
            'raw': raw,
            'withFeet': None, 
            'dx': dx,
            'x_min': x_min,
            'y_min': y_min,
            'z_min': z_min,
        }
        return info

    def sparse_data(self, cubes: np.ndarray) -> sparse.COO:
        """
        Changes data to a sparse representation.

        Args:
            cubes (np.ndarray): the initial array.

        Returns:
            sparse.COO: the corresponding sparse COO array.
        """

        cubes = sparse.COO(cubes)  # the .to_numpy() method wasn't used as the idx_type argument isn't working properly
        cubes.coords = cubes.coords.astype('uint16')  # to save memory
        return cubes



if __name__=='__main__':

    Cubes('testing3.h5')
    


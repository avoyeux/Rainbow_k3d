"""
To add both the real and fake data together in the same HDF5 file.
"""

# IMPORTS
import os
import h5py

# IMPORTs sub
from typing import Self
from dataclasses import dataclass, field

# IMPORTs personal
from common import config, Decorators
from codes.data.base_hdf5_creator import BaseHdf5Creator



@dataclass(slots=True, repr=False, eq=False)
class FileFusionOpener:
    """
    To open the 3 HDF5 files needed for the fusion.

    Raises:
        ValueError: if the key for getitem is not recognized.
    """

    # ARGUMENTs
    new_hdf5_path: str 
    real_hdf5_path: str
    fake_hdf5_path: str

    # PLACEHOLDERs
    new_hdf5: h5py.File = field(init=False)
    real_hdf5: h5py.File = field(init=False)
    fake_hdf5: h5py.File = field(init=False)

    def __enter__(self) -> Self:
        """
        To open the 3 HDF5 files needed for the fusion.

        Returns:
            Self: the class instance itself.
        """

        # OPEN hdf5
        self.new_hdf5 = h5py.File(self.new_hdf5_path, 'w')
        self.real_hdf5 = h5py.File(self.real_hdf5_path, 'r')
        self.fake_hdf5 = h5py.File(self.fake_hdf5_path, 'r')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        To close the 3 HDF5 files needed for the fusion.
        """

        # CLOSE hdf5
        self.new_hdf5.close()
        self.real_hdf5.close()
        self.fake_hdf5.close()

    def __getitem__(self, key: str) -> h5py.File:
        """
        To get the HDF5 file based on the key.

        Args:
            key (str): can be 'real' or 'fake'.

        Raises:
            ValueError: if the key is not 'real' or 'fake'.

        Returns:
            h5py.File: the corresponding HDF5 file.
        """

        if key=='Real':
            return self.real_hdf5
        elif key=='Fake':
            return self.fake_hdf5
        else:
            raise ValueError(f"Key '{key}' not recognized.")


class FusionHdf5(BaseHdf5Creator):
    """
    To add the fake and real data together in the same HDF5 file.
    """

    @Decorators.running_time
    def __init__(
            self,
            filename: str | None = None,
            compression: bool = True,
            compression_lvl: int = 9,
        ) -> None:
        """
        To add the fake and real data together in the same HDF5 file.

        Args:
            filename (str | None, optional): the filename of the new HDF5 file. If not, it fetches
                the name from the config file. Defaults to None.
            compression (bool, optional): deciding to use gzip compression for the datasets in the
                fusion file. Defaults to True.
            compression_lvl (int, optional): the level of the gzip compression to use.
                Defaults to 9.
        """

        # FILENAME setup
        filename = os.path.basename(config.path.data.fusion) if filename is None else filename

        # PARENT
        super().__init__(filename, compression, compression_lvl)
        
        # ATTRIBUTEs
        self.filepath_real = config.path.data.real
        self.filepath_fake = config.path.data.fake

        # SETUP
        self.paths = self.paths_setup()
        self.group_paths = self.paths_choices()

    def paths_setup(self) -> dict[str, str]:
        """
        Gives the paths to the different needed directories.

        Returns:
            dict[str, str]: the paths to the different directories.
        """

        # PATHs formatting
        paths = {'save': config.path.dir.data.hdf5}
        return paths
    
    def paths_choices(self) -> dict[str, list[str] | dict[str, list[str] | dict[str, list[str]]]]:
        """
        Gives the paths to the different needed HDF5 groups.

        Returns:
            dict[str, dict[str, list[str] | dict[str, list[str]]]]: the paths to the different HDF5
                groups in the real and fake files.
        """

        # PATHs choices
        global_choices = ['Dates', 'SDO positions', 'STEREO B positions', 'dx']
        real_sub_group_choices = [
            'All data', 'No duplicates', 'SDO line of sight', 'STEREO line of sight',
        ]

        # PATHs real and fake
        real_paths = {
            'main': ['Time indexes'],
            'group': {
                'Filtered': real_sub_group_choices,
                'Time integrated': ['All data', 'No duplicates'],
            },
        }
        fake_paths = {
            'main': ['Time indexes'],
            'group': {
                'Filtered': ['All data'],
            },
        }

        # PATHs formatting
        paths = {
            'global': global_choices,
            'Real': real_paths,
            'Fake': fake_paths,
        }
        return paths

    @Decorators.running_time
    def create(self) -> None:
        """
        Creates the new HDF5 file containing both the real and fake data.
        """

        # PATH to new hdf5
        new_hdf5_path = os.path.join(self.paths['save'], self.filename)

        # OPEN setup
        file_opener = FileFusionOpener(new_hdf5_path, self.filepath_real, self.filepath_fake)

        # OPEN hdf5
        with file_opener as data:
            # METADATA add
            metadata = self.main_metadata()
            metadata['description'] = (
                'File containing the fusion between the real protuberance volumetric data and the '
                'fake one created with fake STEREO B png and SDO fits files.'
            )
            self.add_dataset(data.new_hdf5, metadata)

            # DATA global
            global_datasets = self.group_paths['global']
            for dataset_path in global_datasets:
                dataset = data.real_hdf5[dataset_path]
                data.new_hdf5.copy(dataset, dataset_path)

            # DATA specific
            for key in ['Real', 'Fake']:
                
                # CREATE 2 main group
                data.new_hdf5.create_group(key)
                base_path = f'{key}/'
                data_file = data[key]

                # CREATE paths
                path: str
                for path in self.group_paths[key]['main']:

                    # DATA get
                    dataset = data_file[path]

                    # SAVE choice
                    new_path =  base_path + f'{path}'
                    data.new_hdf5.copy(dataset, new_path)

                # CREATE subs
                sub_path: str
                for sub_path in self.group_paths[key]['group'].keys():

                    # CREATE sub group
                    data.new_hdf5.create_group(base_path + sub_path)
                    
                    sub_sub_path: str
                    for sub_sub_path in self.group_paths[key]['group'][sub_path]:

                        # DATA get
                        data_path = f'{sub_path}/{sub_sub_path}'
                        group = data_file[data_path]

                        # SAVE choice
                        data.new_hdf5.copy(group, base_path + data_path)



if __name__ == '__main__':

    # FUSION
    fusion = FusionHdf5(filename=None)
    fusion.create()

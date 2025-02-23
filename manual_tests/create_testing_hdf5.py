"""
To create a completely fake HDF5 file only used to test the testing codes on it.
"""

# IMPORTs
import os
import h5py
import sparse

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import root_path
from data.base_hdf5_creator import BaseHDF5Protuberance



class CreateTestingHDF5(BaseHDF5Protuberance):
    """
    To create a completely fake HDF5 file only used to test the testing codes on it.
    """

    def __init__(self, filename: str, wrong: bool = False) -> None:
        """
        To create a completely fake HDF5 file only used to test the testing codes on it.

        Args:
            filename (str): the name of the file to be created.
            wrong (bool, optional): to create a wrong set of all data. Defaults to False.
        """

        # PARENT initialization
        super().__init__()

        # ARGUMENTs
        self.filename = filename
        self.wrong = wrong

        # CONSTANTs
        self.shape: tuple = (4, 20, 20, 20)

        # RUN
        self.paths = self.setup_paths()
        self.create_hdf5()

    def setup_paths(self) -> dict[str, str]:
        """
        To format the paths needed for the creation of the HDF5 file.

        Returns:
            dict[str, str]: the formatted paths.
        """

        # PATHs formatting
        paths = {
            'save': os.path.join(root_path, 'manual_tests'),
        }
        return paths
    
    def create_hdf5(self) -> None:
        """
        To create the fake HDF5 file for check the test codes.
        """

        with h5py.File(os.path.join(self.paths['save'], self.filename), 'w') as HDF5File:

            # METADATA
            metadata = self.main_metadata()
            metadata['description'] = 'Data created for testing the data testing codes.'
            self.add_dataset(HDF5File, metadata)

            # DATA fake
            group = HDF5File.create_group('Filtered')
            group.attrs['description'] = 'Filtered data for testing the data testing codes.'
            fake_lineofsight_data = self.fake_lineofsight_data()

            for name in ['SDO line of sight', 'STEREO line of sight', 'All data']:
                
                if self.wrong and name == 'All data':
                    fake_lineofsight_data = sparse.COO(np.ones(self.shape, dtype='uint8'))

                group = self.add_cube(
                    group=group,
                    data=fake_lineofsight_data,
                    data_name=name,
                    values=1,
                    borders=None,
                )
                group[name].attrs['description'] = (
                    f'Fake {name} data for testing the data testing codes.'
                )
        print(f"File {self.filename} created.")

    def fake_lineofsight_data(self) -> sparse.COO:
        """
        To create a fake line of sight data.

        Returns:
            sparse.COO: the fake line of sight data.
        """

        # DATA
        data = np.zeros(self.shape, dtype='uint8')
        data[:, 5:10, 10:15, 15:20] = 1
        return sparse.COO(data)



if __name__ == '__main__':

    # FILENAME
    filename = 'testing_tests.h5'

    # CREATE
    CreateTestingHDF5(filename, wrong=False)

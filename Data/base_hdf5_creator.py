"""
To store the parent (base) class storing methods to be able to create an HDF5 file with a default
formatting.
"""

# IMPORTs
import h5py
import datetime

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass



@dataclass
class VolumeInfo:
    """
    To save basic volumetric protuberance data from a .save file.
    """

    # VALUEs in km
    dx: int
    xt_min: float
    yt_min: float
    zt_min: float


class BaseHdf5Creator:
    """
    To store base methods for creating HDF5 files.
    """

    def __init__(self) -> None:
        """
        To type initialise the instance attributes usd in these default hdf5 creator methods and
        the constant instant attributes.
        """

        # PLACEHOLDERs
        self.filename: str

    def add_group( # todo need to change so that the name arg is also optional
            self,
            parent_group: h5py.File | h5py.Group,
            info: dict[str, str | int | float | np.generic | np.ndarray | dict],
            name: str,
        ) -> None:
        """
        Adds group(s) with dataset(s) and attribute(s) to a HDF5 group.
        Group created if dict[str, str | dict]
        Dataset created if not dict[str, dict] and at least one item not being a str.
        If an item is a dict, then .add_group() method is called on the item.
        Any str item becomes an attribute.

        Args:
            parent_group (h5py.File | h5py.Group): the HDF5 file or group where to add the new
                group.
            info (dict[str, str | int | float | np.generic | np.ndarray | dict]):
                the information and data to add in the group.
            name (str): the name of the group to add.
        """
            
        # CHECK name
        if name == '': return 

        if any(isinstance(value, dict) for value in info.values()):
            # GROUP create
            group = parent_group.require_group(name)

            for key, value in info.items():
                if isinstance(value, dict):
                    # GROUP add
                    self.add_group(group, value, key)
                else: 
                    # ATTRIBUTES add
                    group.attrs[key] = value

        elif all(isinstance(value, str) for value in info.values()):
            # GROUP with attributes only
            group = parent_group.require_group(name)
            for key, value in info.items(): group.attrs[key] = value

        else:
            # DATASET as values but no dict
            self.add_dataset(parent_group, info, name) #type: ignore

    def add_dataset(
            self,
            parent_group: h5py.File | h5py.Group,
            info: dict[str, str | int | float | np.generic | np.ndarray],
            name: str | None = None,
        ) -> None:
        """
        Adds a dataset and attributes to a HDF5 group like object.

        Args:
            parent_group (h5py.File | h5py.Group): the HDF5 file or group where to add the dataset.
            info (dict[str, str | int | float | np.generic | np.ndarray]): the information to add
                in the DataSet.
            name (str, optional): the name of the DataSet to add.
        """
        
        # CHECK empty
        if len(info) == 0: return
        key: str = ''  # no need but for the type checker

        # DATASET key for ndarray
        stopped = False
        for key, item in info.items(): 
            if not isinstance(item, str): stopped = True; break
        if not stopped: key = ''  # no dataset. Add attributes to group.

        # CHECK args
        if (key != '') and name is None:
            raise ValueError(
                f"In {self.add_dataset.__qualname__}, if the 'info' input dict is "
                "of not of type dict[str, str] then a 'name' argument for the dataset has to be "
                "given. "
            )
        
        # DATASET create
        if (key != ''):
            dataset = parent_group.create_dataset(name, data=info[key])
        else:
            dataset = parent_group

        # ATTRIBUTEs add
        for k, value in info.items():
            if k == key: continue
            dataset.attrs[k] = value

    def main_metadata(self) -> dict[str, str]:
        """
        Creates a dictionary with some default metadata for an HDF5 file.

        Returns:
            dict[str, str]: the default file metadata.
        """

        # METADATA
        metadata = {
            'author': 'Voyeux Alfred',
            'creationDate': datetime.datetime.now().isoformat(),
            'filename': self.filename,
            'description': (
                "HDF5 file containing data to reconstruct a 3D protuberance in the Sun's Corona.\n"
            ),
        }
        return metadata

    def dataset_dict(
            self,
            data: np.ndarray,
            name: str,
            attrs: dict[str, str] | None = None,
        ) -> dict[str, dict[str, np.ndarray] | dict[str, str | np.ndarray]]:
        # ! most likely useless. Keeping it for now just to make sure.
        data_dict = {'data': data}
        if attrs is not None: data_dict |= attrs

        dataset_dict = {name: data_dict}
        return dataset_dict


class BaseHDF5Protuberance(BaseHdf5Creator):
    """
    To store base methods to create an HDF5 file using the rainbow protuberance data.
    """

    def __init__(self) -> None:
        
        super().__init__()

        # PLACEHOLDERs
        self.volume: VolumeInfo

        # CONSTANTs
        self.solar_r = 6.96e5  # in km

    def create_borders(
            self,
            values: tuple[float, float, float],
        ) -> dict[str, dict[str, str | float]]:
        """
        Gives the border information for the data.

        Args:
            values (tuple[float, float, float]): the xmin, ymin, zmin value in km.

        Returns:
            tuple[dict[str, any], dict[str, dict[str, any]]]: the data and metadata for the data
                borders.
        """

        info = {
            'xt_min': {
                'data': np.array(values[0], dtype='float32'),
                'unit': 'km',
                'description': (
                    "The minimum X-axis Carrington Heliographic Coordinates value for each data "
                    "cube.\nThe X-axis in Carrington Heliographic Coordinates points towards the "
                    "First Point of Aries."
                ),
            }, 
            'yt_min': {
                'data': np.array(values[1], dtype='float32'),
                'unit': 'km',
                'description': (
                    "The minimum Y-axis Carrington Heliographic Coordinates value for each data "
                    "cube.\nThe Y-axis in Carrington Heliographic Coordinates points towards the "
                    "ecliptic's eastern horizon."
                ),
            },
            'zt_min': {
                'data': np.array(values[2], dtype='float32'),
                'unit': 'km',
                'description': (
                    "The minimum Z-axis Carrington Heliographic Coordinates value for each data "
                    "cube.\nThe Z-axis in Carrington Heliographic Coordinates points towards "
                    "Sun's north pole."
                ),
            },
        }
        return info
    
    def to_index_pos(self, coords: np.ndarray) -> tuple[np.ndarray, dict]:
 
        # BORDERs new
        x_min, y_min, z_min = np.min(coords, axis=1)
        x_min = x_min if x_min <= self.volume.xt_min else self.volume.xt_min
        y_min = y_min if y_min <= self.volume.yt_min else self.volume.yt_min
        z_min = z_min if z_min <= self.volume.zt_min else self.volume.zt_min
        new_borders = self.create_borders((x_min, y_min, z_min)) # ? add it to parent class?

        # COORDs indexes
        coords[0, :] -= x_min
        coords[1, :] -= y_min
        coords[2, :] -= z_min
        coords /= self.volume.dx
        coords = np.round(coords).astype(int)
        return coords, new_borders     
    
    def dx_to_dict(self) -> dict[str, str | float]:

        dx_dict = {
            'data': self.volume.dx,
            'unit': 'km',
            'description': "The voxel resolution in kilometres.",
        }
        return dx_dict

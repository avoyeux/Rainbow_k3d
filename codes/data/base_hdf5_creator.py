"""
To store the parent (base) class storing methods to be able to create an HDF5 file with a default
formatting.
"""

# IMPORTs
import h5py
import sparse
import datetime

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass



@dataclass(slots=True, repr=False, eq=False)
class VolumeInfo:
    """
    To save basic volumetric protuberance data from a .save file.
    """

    # VALUEs in km
    dx: float
    xt_min: float
    yt_min: float
    zt_min: float


class BaseHdf5Creator:
    """
    To store base methods for creating HDF5 files.
    """

    def __init__(self, filename: str, compression: bool = False, compression_lvl: int = 9) -> None:
        """
        To type initialise the instance attributes usd in these default hdf5 creator methods and
        the constant instant attributes.

        Args:
            filename (str): the filename of the HDF5 file to create.
            compression (bool, optional): deciding to compress the datasets using gzip.
                Defaults to False.
            compression_lvl (int, optional): the gzip compression level. Defaults to 9.
        """

        # PLACEHOLDERs
        self.filename = filename
        self.compression = 'gzip' if compression else None
        self.compression_lvl = compression_lvl

        # ATTRIBUTEs
        self.compression_min_size = 3 * 1024

    def add_group(
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
            info (dict[str, str | int | float | np.generic | np.ndarray | dict]): the information
                and data to add in the group.
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
                    self.add_group(parent_group=group, info=value, name=key)
                else: 
                    # ATTRIBUTES add
                    group.attrs[key] = value

        elif all(isinstance(value, str) for value in info.values()):
            # GROUP with attributes only
            group = parent_group.require_group(name)
            for key, value in info.items(): group.attrs[key] = value

        else:
            # DATASET as values but no dict
            self.add_dataset(parent_group=parent_group, info=info, name=name)

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
            name (str, optional): the name of the DataSet to add. Defaults to None.
        """

        # SETUP
        key: str = ''  # no need but it is for the type checker
        compression: str | None = self.compression
        compression_lvl: int | None = self.compression_lvl

        # CHECKs
        if len(info) == 0: return
        if compression is None: compression_lvl = None

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
            # SELECT dataset
            data = info[key]

            # NDARRAY change
            if not isinstance(data, np.ndarray): data = np.array(data)

            # CHECK nbytes
            if data.nbytes < self.compression_min_size: compression, compression_lvl = None, None
            
            dataset = parent_group.create_dataset(
                name,
                data=info[key],
                compression=compression,
                compression_opts=compression_lvl,
            )
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


class BaseHDF5Protuberance(BaseHdf5Creator):
    """
    To store base methods to create an HDF5 file using the rainbow protuberance data.
    """

    def __init__(self, filename: str, compression: bool = False, compression_lvl: int = 9) -> None:
        """
        To initialise the instance attributes used when creating a default protuberance HDF5 file.

        Args:
            filename (str): the filename of the HDF5 file to create.
            compression (bool, optional): deciding to compress the datasets using gzip.
                Defaults to False.
            compression_lvl (int, optional): the gzip compression level. Defaults to 9.
        """
        
        # PARENT
        super().__init__(filename, compression, compression_lvl)

        # CONSTANTs
        self.solar_r = 6.96e5  # in km

        # PLACEHOLDERs
        self.volume: VolumeInfo  # todo change this as a placeholder doesn't make sense.

    def create_borders(
            self,
            values: tuple[float, float, float],
        ) -> dict[str, dict[str, str | float]]:
        """
        Gives the border information for the data.

        Args:
            values (tuple[float, float, float]): the xmin, ymin, zmin value in km.

        Returns:
            dict[str, dict[str, str | float]]: the border information.
        """

        info = {
            'xt_min': {
                'data': values[0],
                'unit': 'km',
                'description': (
                    "The minimum X-axis Carrington Heliographic Coordinates value for each data "
                    "cube.\nThe X-axis in Carrington Heliographic Coordinates points towards the "
                    "First Point of Aries."
                ),
            }, 
            'yt_min': {
                'data': values[1],
                'unit': 'km',
                'description': (
                    "The minimum Y-axis Carrington Heliographic Coordinates value for each data "
                    "cube.\nThe Y-axis in Carrington Heliographic Coordinates points towards the "
                    "ecliptic's eastern horizon."
                ),
            },
            'zt_min': {
                'data': values[2],
                'unit': 'km',
                'description': (
                    "The minimum Z-axis Carrington Heliographic Coordinates value for each data "
                    "cube.\nThe Z-axis in Carrington Heliographic Coordinates points towards "
                    "Sun's north pole."
                ),
            },
        }
        return info
    
    def add_cube(
            self,
            group: h5py.File | h5py.Group,
            data: sparse.COO,
            data_name: str,
            values: int | float | None = None,
            borders: dict[str, dict[str, str | float]] | None = None,
        ) -> h5py.File | h5py.Group:
        """
        To add to an h5py.Group, the data and metadata of a cube index spare.COO object. This takes
        also into account the border information.

        Args:
            group (h5py.File | h5py.Group): the Group where to add the data information.
            data (sparse.COO): the data that needs to be included in the file.
            data_name (str): the group name to be used in the file.
            values (int | float | None): the value for the voxels. Set to None when all the voxels
                don't have the same value. Default to None.
            borders (dict[str, dict[str, str | float]] | None): the data border information. Set to
                None if you don't want to add the border information in the created group.
                Default to None.

        Returns:
            h5py.File | h5py.Group: the updated file or group.
        """
        
        raw = {
            'description': "Default",
            'coords': {
                'data': data.coords.astype('uint16'),
                'unit': 'none',
                'description': (
                    "The index coordinates of the initial voxels.\nThe shape is (4, N) where the "
                    "rows represent t, x, y, z where t the time index (i.e. which cube it is), "
                    "and N the total number of voxels.\n"
                ),
            },
            'values': {
                'data': data.data if values is None else values, 
                'unit': 'none',
                'description': (
                    "The values for each voxel. If int or float, then all voxels have the same "
                    "value."
                ),
            },
        }
        # BORDERs add
        if borders is not None: raw |= borders
        
        # ADD group
        self.add_group(group, raw, data_name)
        return group
    
    def to_index_pos(self, coords: np.ndarray, unique: bool = False) -> tuple[np.ndarray, dict]:
        """
        Converts the coordinates from km to indexes and returns the indexes with the new borders.

        Args:
            coords (np.ndarray): the coordinates in km (can have shape (3, n) or (4, n)).
            unique (bool, optional): if True, the unique indexes are returned. Defaults to False.

        Returns:
            tuple[np.ndarray, dict]: the coordinates in indexes and the new borders.
        """
        
        # BORDERs new
        if coords.shape[0] == 3:
            x_min, y_min, z_min = np.min(coords, axis=1)
        else:
            _, x_min, y_min, z_min = np.min(coords, axis=1)
        x_min = x_min if x_min <= self.volume.xt_min else self.volume.xt_min
        y_min = y_min if y_min <= self.volume.yt_min else self.volume.yt_min
        z_min = z_min if z_min <= self.volume.zt_min else self.volume.zt_min
        new_borders = self.create_borders((x_min, y_min, z_min))

        # COORDs indexes
        coords[-3, :] -= x_min
        coords[-2, :] -= y_min
        coords[-1, :] -= z_min
        coords[-3:, :] /= self.volume.dx
        coords = np.round(coords).astype(int)
        if unique: coords = np.unique(coords, axis=1)
        return coords, new_borders     
    
    def dx_to_dict(self) -> dict[str, str | float]:
        """
        Returns the voxel resolution in a dictionary.

        Returns:
            dict[str, str | float]: the voxel resolution information.
        """

        dx_dict = {
            'data': self.volume.dx,  # ? maybe let dx be a method argument
            'unit': 'km',
            'description': "The voxel resolution in kilometres.",
        }
        return dx_dict

"""
To save the dataclasses for the needed information in the animation.
This was created to make it easier to swap between the real and fake protuberance data.
"""

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np
from typing import Self

# IMPORTs sub
from dataclasses import dataclass, field



@dataclass(slots=True, frozen=True, repr=False, eq=False)
class CubesConstants:
    """
    Stores the basic global information for the protuberance.
    It is an immutable class without a __dict__  method (cf. dataclasses.dataclass).
    """

    dx: float
    dates: list[str]
    time_indexes: np.ndarray


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class PolynomialData:
    """
    Stores the polynomial results and some corresponding information.
    It is an immutable class without a __dict__ method (cf. dataclasses.dataclass).
    """

    # INFO
    name: str
    order: int
    color_hex: int

    # DATA
    dataset: h5py.Dataset

    def __getitem__(self, index: int) -> np.ndarray:
        """
        To get a single cube from the polynomial data.

        Args:
            index (int): the time index needed to be fetched.

        Returns:
            np.ndarray: the polynomial data.
        """
                    
        # FILTER
        cube_filter = self.dataset[0] == index
        cube_coords = self.dataset[1:, cube_filter]

        # 3D array
        cube_shape = np.max(cube_coords, axis=1) + 1
        cube = np.zeros(cube_shape, dtype='uint8')
        cube[tuple(cube_coords)] = 1
        return cube


@dataclass(slots=True, repr=False, eq=False)
class ParentInfo:
    """
    Stores the data and information related to each different data group chosen.
    """

    # ID
    group_path: str
    opacity: float  # ? add the colour value for each cube?

    # BORDERs index
    xt_min_index: float
    yt_min_index: float
    zt_min_index: float

    # DATA
    dataset_coords: h5py.Dataset
    dataset_values: h5py.Dataset

    # COORDs max shape
    max_shape: np.ndarray | None = field(default=None, init=False)

    @property
    def name(self) -> str:
        """
        To create the name of the dataset depending on the group path.

        Returns:
            str: the name of the dataset.
        """

        group_names = self.group_path.split('/')

        if 'Fake' in group_names: return 'Fake ' + group_names[-1]
        return group_names[-1]
    
    @property
    def shape(self) -> np.ndarray:
        """
        To get the shape of the data as a dense array.

        Returns:
            tuple[int, int, int, int]: the shape of the data.
        """

        return np.max(self.dataset_coords, axis=1) + 1


@dataclass(slots=True, repr=False, eq=False)
class CubeInfo(ParentInfo):
    """
    Stores the data and information related to the 'real' data cubes.
    """

    # DATA
    polynomials: list[PolynomialData] | None = field(default=None)

    def __getitem__(self, index: int) -> list[np.ndarray]:
        """
        To get a single cube from each the protuberance and polynomials data.

        Args:
            index (int): the time index needed to be fetched.

        Returns:
            list[np.ndarray]: the protuberance and polynomials data.
        """

        # CHECK shape
        if self.max_shape is None: raise ValueError('max_shape is not defined')
        
        # FILTER
        cube_filter = self.dataset_coords[0] == index
        cube_coords = self.dataset_coords[1:, cube_filter]
        if self.dataset_values.shape == ():
            values = self.dataset_values[...]
        else:
            values = self.dataset_values[cube_filter.ravel()]

        # cube_shape = (180, 227, 236)
        # cube_shape = (400, 400, 400)
        
        # 3D array
        cube = np.zeros(self.max_shape, dtype='uint8')
        cube[tuple(cube_coords)] = values
        result = [cube]

        # POLYNOMIALs
        if self.polynomials is not None:
            result += [polynomial[index] for polynomial in self.polynomials]
        return result


@dataclass(slots=True, repr=False, eq=False)
class FakeCubeInfo(ParentInfo):
    """
    Stores the data and information related to the 'fake' data cubes.
    """
    
    # DATA
    time_indexes_fake: np.ndarray
    time_indexes_real: np.ndarray

    # PLACEHOLDERs
    value_to_index: dict[int, int] = field(init=False)
    polynomials: None = field(default=None, init=False)  # just to look like a CubeInfo

    def __post_init__(self) -> None:

        self.value_to_index = {value: index for index, value in enumerate(self.time_indexes_fake)}

    def __getitem__(self, index: int) -> list[np.ndarray]:
        """
        To get the corresponding cube from the fake data.

        Args:
            index (int): the index (from the real data) needed to be fetched.

        Returns:
            list[np.ndarray]: the corresponding fake cube inside a list.
        """

        # INDEX real to fake
        time_index = int(self.time_indexes_real[index])
        fake_index = self.value_to_index[time_index]

        # CHECK shape
        if self.max_shape is None: raise ValueError('max_shape is not defined')

        # FILTER
        cube_filter = self.dataset_coords[0] == fake_index
        cube_coords = self.dataset_coords[1:, cube_filter]
        if self.dataset_values.shape == ():
            values = self.dataset_values[...]
        else:
            values = self.dataset_values[cube_filter.ravel()]

        # 3D array
        cube = np.zeros(self.max_shape, dtype='uint8')
        cube[tuple(cube_coords)] = values
        return [cube]


@dataclass(slots=True, repr=False, eq=False)
class CubesData:
    """
    Stores all the data and information needed to name and position the solar protuberance for the
    k3d animation.
    This class doesn't have a __dict__() method (cf. dataclasses.dataclass).
    """

    # FILE
    hdf5File: h5py.File

    # POS satellites
    sdo_pos: np.ndarray | None = field(default=None, init=False)
    stereo_pos: np.ndarray | None = field(default=None, init=False)

    # CUBES data
    all_data: CubeInfo | None = field(default=None, init=False)
    no_duplicate: CubeInfo | None = field(default=None, init=False)
    integration_all_data: CubeInfo | None = field(default=None, init=False)
    integration_no_duplicate: CubeInfo | None = field(default=None, init=False)
    los_sdo: CubeInfo | None = field(default=None, init=False)
    los_stereo: CubeInfo | None = field(default=None, init=False)

    # FAKE data
    sun_surface: CubeInfo | None = field(default=None, init=False)
    fake_cube: FakeCubeInfo | None = field(default=None, init=False)

    def __enter__(self) -> Self: return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: self.hdf5File.close()

    def add_shape(self) -> None:
        """
        To add the max shape of all the different data cubes to the cubes themselves.
        This was only done as there seems to be an error in the k3d module, forcing me to keep the
        same cube shapes for all the data.
        """

        shapes = []
        for attribute in self.__slots__:

            if attribute in ['hdf5File', 'sdo_pos', 'stereo_pos']: continue
            attr_value = getattr(self, attribute)
            if attr_value is None: continue

            shapes.append(attr_value.shape)
        shapes = np.stack(shapes, axis=0)
        max_shape = np.max(shapes, axis=0)[1:]

        for attribute in self.__slots__:
            if attribute in ['hdf5File', 'sdo_pos', 'stereo_pos']: continue
            attr_value = getattr(self, attribute)
            if attr_value is None: continue

            attr_value.max_shape = max_shape

    def close(self):
        """
        To close the hdf5 file.
        """

        self.hdf5File.close()

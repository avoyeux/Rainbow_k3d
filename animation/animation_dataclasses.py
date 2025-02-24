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
from dataclasses import dataclass



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
        cube = np.zeros(cube_shape, dtype='uint8')  # ! check this to if the problem comes from here
        cube[tuple(cube_coords)] = 1
        return cube


@dataclass(slots=True, repr=False, eq=False)
class CubeInfo:
    """
    Stores the data and information related to each different data group chosen.
    It is an immutable class without a __dict__ method (cf. dataclasses.dataclass).
    """

    # ID
    name: str
    opacity: float

    # BORDERs index
    xt_min_index: float
    yt_min_index: float
    zt_min_index: float

    # DATA
    dataset_coords: h5py.Dataset
    dataset_values: h5py.Dataset
    polynomials: list[PolynomialData] | None

    max_shape: np.ndarray | None = None

    def __getitem__(self, index: int) -> list[np.ndarray]:
        """
        To get a single cube from each the protuberance and polynomials data.

        Args:
            index (int | slice): the time indexes needed to be fetched.

        Returns:
            list[np.ndarray]: the protuberance and polynomials data.
        """

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
        cube = np.zeros(self.max_shape, dtype=self.dataset_values.dtype)
        cube[tuple(cube_coords)] = values
        result = [cube]

        # POLYNOMIALs
        if self.polynomials is not None:
            result += [polynomial[index] for polynomial in self.polynomials]
        return result
    
    @property
    def shape(self) -> np.ndarray:
        """
        To get the shape of the data.

        Returns:
            tuple[int, int, int, int]: the shape of the data.
        """

        return np.max(self.dataset_coords, axis=1) + 1


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
    sdo_pos: np.ndarray | None = None
    stereo_pos: np.ndarray | None = None

    # CUBES data
    all_data: CubeInfo | None = None
    no_duplicate: CubeInfo | None = None
    integration_all_data: CubeInfo | None = None
    integration_no_duplicate: CubeInfo | None = None
    los_sdo: CubeInfo | None = None
    los_stereo: CubeInfo | None = None

    # FAKE data
    sun_surface: CubeInfo | None = None
    fake_cube: CubeInfo | None = None

    def __enter__(self) -> Self: return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: self.hdf5File.close()

    def add_shape(self) -> None:

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

"""
To save the dataclasses for the needed information in the animation.
This was created to make it easier to swap between the real and fake protuberance data.
"""

# IMPORTs
import h5py
import sparse

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
    time_indexes: np.ndarray
    dates: list[str]


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
        cube_filter = self.dataset[:, 0] == index
        cube_coords = self.dataset[1:, cube_filter]

        # 3D array
        cube_shape = np.max(cube_coords, axis=1) + 1
        cube = np.zeros(cube_shape, dtype='uint8')
        cube[tuple(cube_coords)[1:]] = 1
        return cube


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class CubeInfo:
    """
    Stores the data and information related to each different data group chosen.
    It is an immutable class without a __dict__ method (cf. dataclasses.dataclass).
    """
    # todo when nearly finished, add dates and time_indexes here. will need to change a lot though

    # ID
    name: str

    # BORDERs index
    xt_min_index: float
    yt_min_index: float
    zt_min_index: float

    # DATA
    dataset_coords: h5py.Dataset
    dataset_values: h5py.Dataset
    polynomials: list[PolynomialData] | None

    def __getitem__(self, index: int) -> list[np.ndarray]:
        """
        To get a single cube from each the protuberance and polynomials data.

        Args:
            index (int | slice): the time indexes needed to be fetched.

        Returns:
            list[np.ndarray]: the protuberance and polynomials data.
        """

        # FILTER
        cube_filter = self.dataset_coords[0, :] == index
        cube_coords = self.dataset_coords[1:, cube_filter]
        if self.dataset_values.shape == ():
            values = self.dataset_values[...]
        else:
            values = self.dataset_values[cube_filter.ravel()]

        # 3D array
        print('cubes_coords shape = ', cube_coords.shape)
        cube_shape = np.max(cube_coords, axis=1) + 1
        cube = np.zeros(cube_shape, dtype=self.dataset_values.dtype)
        print('cube shape = ', cube.shape)
        cube[tuple(cube_coords)] = values
        result = [cube]

        # POLYNOMIALs
        if self.polynomials is not None:
            result += [polynomial[index] for polynomial in self.polynomials]
        return result


@dataclass(slots=True, repr=False, eq=False)
class CubesData:
    """
    Stores all the data and information needed to name and position the solar protuberance for the
    k3d animation.
    This class doesn't have a __dict__() method (cf. dataclasses.dataclass).
    """

    hdf5File: h5py.File

    # POS satellites
    sdo_pos: h5py.Dataset | None = None
    stereo_pos: h5py.Dataset | None = None

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

    def close(self):
        """
        To close the hdf5 file.
        """

        self.hdf5File.close()
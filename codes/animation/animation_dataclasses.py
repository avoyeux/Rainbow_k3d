"""
To save the dataclasses for the needed information in the animation.
This was created to make it easier to swap between the real and fake protuberance data.
"""

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field

# TYPE ANNOTATIONs
from typing import Any, Self

# API public
__all__ = [
    'CubesConstants',
    'CubesData',
    'PolynomialData',
    'UniquePolynomialData',
    'CubeInfo',
    'FakeCubeInfo',
    'UniqueCubeInfo',
]



@dataclass(slots=True, frozen=True, repr=False, eq=False)
class CubesConstants:
    """
    Stores the basic global information for the protuberance.
    It is an immutable class without a __dict__  method (cf. dataclasses.dataclass).
    """

    dx: float
    dates: list[str]
    time_indexes: np.ndarray[tuple[int], np.dtype[np.int_]]


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

    def __getitem__(self, index: int) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
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

        # NO DATA
        if cube_coords.size == 0: return np.zeros((1, 1, 1), dtype='uint8')
        
        # DATA 3D
        cube_shape = np.max(cube_coords, axis=1) + 1
        cube = np.zeros(cube_shape, dtype='uint8')
        cube[tuple(cube_coords)] = 1
        return cube


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class UniquePolynomialData(PolynomialData):

    def __getitem__(self, index: Any) -> np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]:
        """
        Just gives back the dense representation of the polynomial data.
        The index argument is not used as in this case, there is only one fit in the dataset.
        The index is here to be consistent with the other __getitem__ methods.

        Args:
            index (Any): just a placeholder to look like the other __getitem__ methods.

        Returns:
            np.ndarray: the polynomial data.
        """
        
        # 3D array
        cube_shape = np.max(self.dataset, axis=1) + 1
        cube = np.zeros(cube_shape, dtype='uint8')
        cube[tuple(self.dataset)] = 1
        return cube


@dataclass(slots=True, repr=False, eq=False)
class ParentInfo:
    """
    Stores the data and information related to each different data group chosen.
    """

    # METADATA
    group_path: str
    opacity: float
    colour: str

    # BORDERs index
    xt_min_index: float
    yt_min_index: float
    zt_min_index: float

    # DATA
    dataset_coords: h5py.Dataset
    dataset_values: h5py.Dataset

    # COORDs shape
    name: str = field(init=False)
    shape: tuple = field(init=False)

    def __post_init__(self) -> None:

        self.name = self.get_name()
        self.shape = tuple(np.max(self.dataset_coords, axis=1) + 1)

    def get_name(self) -> str:
        """
        To create the name of the dataset depending on the group path.

        Returns:
            str: the name of the dataset.
        """

        group_names = self.group_path.split('/')

        if 'Fake' in group_names: return 'Fake ' + group_names[-1]
        return group_names[-1]


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

        # FILTER
        cube_filter = self.dataset_coords[0] == index
        cube_coords = self.dataset_coords[1:, cube_filter]
        if self.dataset_values.shape == ():
            values = self.dataset_values[...]
        else:
            values = self.dataset_values[cube_filter.ravel()]
        
        # NO DATA
        if cube_coords.size == 0: 
            result = (
                [np.zeros((1, 1, 1), dtype='uint8')]
                * (1 + (len(self.polynomials) if self.polynomials is not None else 0))
            )
            return result

        # DATA 3D
        cube = np.zeros(self.shape[1:], dtype='uint8')
        cube[tuple(cube_coords)] = values
        result = [cube]

        # POLYNOMIALs
        if self.polynomials is not None:
            result += [polynomial[index] for polynomial in self.polynomials]
        return results


@dataclass(slots=True, repr=False, eq=False)
class FakeCubeInfo(ParentInfo):
    """
    Stores the data and information related to the 'fake' data cubes.
    """
    
    # DATA
    time_indexes_fake: np.ndarray[tuple[int], np.dtype[np.int_]]
    time_indexes_real: np.ndarray[tuple[int], np.dtype[np.int_]]

    # PLACEHOLDERs
    value_to_index: dict[int, int] = field(init=False)
    polynomials: None = field(default=None, init=False)  # just to look like a CubeInfo

    def __post_init__(self) -> None:

        # PARENT post_init
        super(FakeCubeInfo, self).__post_init__()
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

        # FILTER
        cube_filter = self.dataset_coords[0] == fake_index
        cube_coords = self.dataset_coords[1:, cube_filter]
        if self.dataset_values.shape == ():
            values = self.dataset_values[...]
        else:
            values = self.dataset_values[cube_filter.ravel()]

        # 3D array
        cube = np.zeros(self.shape[1:], dtype='uint8')
        cube[tuple(cube_coords)] = values
        return [cube]


@dataclass(slots=True, repr=False, eq=False)
class UniqueCubeInfo(ParentInfo):
    """
    Stores the data and information related to the 'test' data cube.
    """

    # PLACEHOLDER
    polynomials: list[UniquePolynomialData] | None = field(default=None)

    def __getitem__(self, index: Any) -> list[np.ndarray]:
        """
        To get the dense array corresponding to the test data.
        Always gives to the user the same cube but the method was defined so the class can be used
        the same way as the ones for the real and the fake data.

        Args:
            index (Any): can be anything as it is not used.

        Returns:
            list[np.ndarray]: the dense test cube inside a list.
        """

        # CUBE
        cube = np.zeros(self.shape, dtype='uint8')
        cube[tuple(self.dataset_coords)] = self.dataset_values[...]
        result = [cube]
        
        # POLYNOMIALs
        if self.polynomials is not None:  # ! could be wrong as there is only one index here
            result += [polynomial[0] for polynomial in self.polynomials]
        return result


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
    sdo_pos: np.ndarray[tuple[int], np.dtype[np.floating]] | None = field(default=None, init=False)
    stereo_pos: np.ndarray[tuple[int], np.dtype[np.floating]] | None = field(
        default=None,
        init=False,
    )

    # CUBES data
    all_data: CubeInfo | None = field(default=None, init=False)
    no_duplicate: CubeInfo | None = field(default=None, init=False)
    integration_all_data: CubeInfo | None = field(default=None, init=False)
    integration_no_duplicate: CubeInfo | None = field(default=None, init=False)
    full_integration_all_data: UniqueCubeInfo | None = field(default=None, init=False)
    full_integration_no_duplicate: UniqueCubeInfo | None = field(default=None, init=False)
    los_sdo: CubeInfo | None = field(default=None, init=False)
    los_stereo: CubeInfo | None = field(default=None, init=False)

    # FAKE data
    sun_surface: CubeInfo | None = field(default=None, init=False)
    fake_cube: FakeCubeInfo | None = field(default=None, init=False)

    # TEST data
    test_cube: UniqueCubeInfo | None = field(default=None, init=False)

    def __enter__(self) -> Self: return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: self.hdf5File.close()

    def close(self):
        """
        To close the hdf5 file.
        """

        self.hdf5File.close()

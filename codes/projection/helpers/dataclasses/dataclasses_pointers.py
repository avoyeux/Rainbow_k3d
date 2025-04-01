"""
To store dataclasses related to structuring the metadata and the dataset pointers gotten from the
main HDF5 file.
"""

from __future__ import annotations

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass, field

# IMPORTs personal
if TYPE_CHECKING:
    from codes.data.polynomial_fit.polynomial_reprojection import ReprojectionProcessedPolynomial



@dataclass(slots=True, repr=False, eq=False)
class BasePointer:
    """
    Base class to store the dataset pointers and some corresponding information.
    """

    # BORDERs
    xt_min : float
    yt_min : float
    zt_min : float

    # POINTERs coords
    pointer: h5py.Dataset


@dataclass(slots=True, repr=False, eq=False)
class DataPointer(BasePointer):
    """
    Class to store the pointer to the data cubes used in the polynomial projection module.
    """

    # POINTERs fit
    fit_information: list[ReprojectionProcessedPolynomial] | None

    def __getitem__(self, item: int) -> np.ndarray:  # ! change this dunder method bcs of the fit
        """
        To get the data cube from the corresponding pointer.

        Args:
            item (int): the index of the data cube to get.

        Returns:
            np.ndarray: the corresponding data cube.
        """
        
        data_filter = self.pointer[0, :] == item
        return self.pointer[1:, data_filter].astype('float64')


@dataclass(slots=True, repr=False, eq=False)
class FakeDataPointer(BasePointer):
    """
    Class to store the pointer to the fake data cubes used in the polynomial projection module.
    """

    # POINTERs fit
    fit_information: None = field(default=None, init=False)

    # INDEXes time
    real_time_indexes: np.ndarray
    fake_time_indexes: np.ndarray

    # PLACEHOLDERs
    value_to_index: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:

        self.value_to_index = {value: index for index, value in enumerate(self.fake_time_indexes)}

    def __getitem__(self, item: int) -> np.ndarray:
        """
        To get the fake data cube corresponding to the real data cube gotten from the 'item' index.

        Args:
            item (int): the index of the real data cube to get the corresponding fake data cube.

        Returns:
            np.ndarray: the corresponding fake data cube.
        """
        
        # INDEX real to fake
        time_index = int(self.real_time_indexes[item])
        fake_index = self.value_to_index[time_index]

        # FILTER
        data_filter = self.pointer[0, :] == fake_index
        return self.pointer[1:, data_filter].astype('float64')


@dataclass(slots=True, repr=False, eq=False)
class UniqueDataPointer(BasePointer):
    """
    Class to store the pointer to the data cubes that are unique and used in the polynomial
    projection module.
    Hence the getitem dunder method always gives the same coordinate values.
    """

    # POINTERs fit
    fit_information: list[ReprojectionProcessedPolynomial] | None

    def __getitem__(self, item: Any) -> np.ndarray:
        """
        To get the test data cube from the corresponding pointer.
        This dunder method was created only so that the class can be used like it's Parent or 
        the FakeDataPointer class.

        Args:
            item (Any): can be anything as it is not used.

        Returns:
            np.ndarray: the test data cube.
        """

        return self.pointer[...].astype('float64')


@dataclass(slots=True, repr=False, eq=False)
class CubesPointers:
    """
    To store the pointers to the data cubes used in the polynomial projection module.
    """

    all_data: DataPointer | None = field(default=None, init=False)
    no_duplicates: DataPointer | None = field(default=None, init=False)
    full_integration_all_data: UniqueDataPointer | None = field(default=None, init=False)
    full_integration_no_duplicates: UniqueDataPointer | None = field(default=None, init=False)
    integration: list[DataPointer] | None = field(default=None, init=False)
    line_of_sight: DataPointer | None = field(default=None, init=False)
    fake_data: FakeDataPointer | None = field(default=None, init=False)
    test_cube: UniqueDataPointer | None = field(default=None, init=False)

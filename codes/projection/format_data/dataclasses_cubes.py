"""
To store dataclasses related to the saving and formatting of the volumetric protuberance
information.
"""

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Any
from dataclasses import dataclass, field



@dataclass(slots=True, repr=False, eq=False)
class CubeInformation:
    """
    To store the information of a single cube.
    """

    # BORDERs
    xt_min : float
    yt_min : float
    zt_min : float

    # VALUEs
    coords: np.ndarray
    order: int | None = None 


@dataclass(slots=True, repr=False, eq=False)
class FitPointer:
    """
    Class to store the pointer to the polynomial fit parameters used in the polynomial projection
    module.
    """

    # METADATA
    fit_order: int
    integration_time: int | str

    # POINTERs
    parameters: h5py.Dataset

    def __getitem__(self, item: int) -> np.ndarray:
        """
        To get the data coordinates for the corresponding fit.

        Args:
            item (int): the cube index to consider.

        Returns:
            np.ndarray: the corresponding fit coordinates.
        """

        data_filter = self.parameters[0, :] == item
        return self.parameters[1:, data_filter].astype('float64')


@dataclass(slots=True, repr=False, eq=False)
class UniqueFitPointer(FitPointer):
    """
    Class to store the pointer to the unique polynomial fit used in the polynomial projection.
    Created like so that the usage can be exactly the same than for the normal datasets.
    """

    def __getitem__(self, item: Any) -> np.ndarray:
        """
        To get the data coordinates for the unique polynomial fit.
        The 'item' argument here is just a placeholder and is not used at all.

        Args:
            item (Any): can be anything as it is not used.

        Returns:
            np.ndarray: the fit coordinates for the dataset which only has one set of coordinates.
        """

        return self.parameters[...].astype('float64')


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

    # METADATA
    name: str
    group_path: str
    integration_time: int | str | None


@dataclass(slots=True, repr=False, eq=False)
class DataPointer(BasePointer):
    """
    Class to store the pointer to the data cubes used in the polynomial projection module.
    """

    # POINTERs fit
    fit_information: list[FitPointer] | None

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
    fit_information: list[FitPointer] | None

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

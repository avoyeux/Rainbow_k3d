"""
To store the global dataclasses used in the SDO projection code.
"""

# IMPORTs alias
import numpy as np

# IMPORTS sub
from dataclasses import dataclass, field

# IMPORTs personal
from codes.projection.format_data.dataclasses_cubes import CubeInformation
from codes.projection.format_data.dataclasses_sdo_image import PolarImageInfo
from codes.projection.format_data.dataclasses_fit_n_envelopes import FitWithEnvelopes

# TYPE ANNOTATIONs
from typing import cast

# API public
__all__ = [
    'GlobalConstants',
    'ProcessConstants',
    'ProjectedData',
    'ProjectionData',
]



@dataclass(slots=True, repr=False, eq=False)
class GlobalConstants:
    """
    Class to store the global constants used in the polynomial projection module.
    """

    # CONSTANTs
    dx: float
    solar_r: float
    dates: np.ndarray
    time_indexes: np.ndarray

    # PLACEHOLDERs
    d_theta: float = field(init=False)

    def __post_init__(self) -> None:
        self.d_theta = 360 / (2 * np.pi * self.solar_r / self.dx)


@dataclass(slots=True, repr=False, eq=False)
class ProcessConstants:
    """
    Class to store the constants for each processes used in the polynomial projection module.
    """

    # CONSTANTs
    ID: int
    date: str
    time_index: int


@dataclass(slots=True, repr=False, eq=False)
class ProjectedData:

    # METADATA
    name: str
    colour: str
    cube_index: int
    integration_time: int | str | None

    # DATA
    cube: CubeInformation
    fit_n_envelopes: list[FitWithEnvelopes] | None

    def __getstate__(self) -> dict[str, str | int | list[FitWithEnvelopes] | None]:
        """
        To pickle only what is needed. In my case, only the warping data is needed.

        Returns:
            dict[str, str | int | list[FitWithEnvelopes] | None]: _description_
        """

        state = {  # ? do I need to pickle the cube ?
            'name': self.name,
            'colour': self.colour,
            'cube_index': self.cube_index,
            'integration_time': self.integration_time,
            'fit_n_envelopes': self.fit_n_envelopes,
        }
        return state
    
    def __setstate__(self, state: dict[str, str | int | list[FitWithEnvelopes] | None]) -> None:
        """
        To unpickle the object. Sets all the non-warping related attributes to None.

        Args:
            state (dict[str, str | int | list[FitWithEnvelopes] | None]): the state of the pickled
                object.
        """

        self.__init__(
            name=cast(str, state['name']),
            colour=cast(str, state['colour']),
            cube_index=cast(int, state['cube_index']),
            integration_time=cast(int | str | None, state['integration_time']),
            cube=cast(CubeInformation, None),  # * as I am not keeping the cube
            fit_n_envelopes=cast(list[FitWithEnvelopes] | None, state['fit_n_envelopes']),
        )


@dataclass(slots=True, repr=False, eq=False)
class ProjectionData:
    """
    To store the data used in the polynomial projection module.
    """

    ID: int
    sdo_image: PolarImageInfo | None = None
    sdo_mask: PolarImageInfo | None = None
    all_data: ProjectedData | None = None
    no_duplicates: ProjectedData | None = None
    full_integration_no_duplicates: ProjectedData | None = None
    integration: list[ProjectedData] | None = None
    line_of_sight: ProjectedData | None = None
    fake_data: ProjectedData | None = None
    test_cube: ProjectedData | None = None

    def __getstate__(self) -> dict[str, int | ProjectedData | list[ProjectedData] | None]:
        """
        To pickle only what is needed. In my case, only the warping data is needed.

        Returns:
            dict[str, int | ProjectedData | list[ProjectedData] | None]: the data which is needed
                when pickling the object. In my case, only the warping data is needed.
        """

        state = {
            'ID': self.ID,
            'full_integration_no_duplicates': self.full_integration_no_duplicates,
            'integration': self.integration,
        }
        return state
    
    def __setstate__(
            self,
            state: dict[str, int | ProjectedData | list[ProjectedData] | None],
        ) -> None:
        """
        To unpickle the object. Sets all the non-warping related attributes to None.

        Args:
            state (dict[str, int | ProjectedData | list[ProjectedData] | None]): the state of the
                pickled object.
        """

        self.__init__(
            ID=cast(int, state['ID']),
            sdo_image=None,
            sdo_mask=None,
            all_data=None,
            no_duplicates=None,
            full_integration_no_duplicates=cast(
                ProjectedData | None,
                state['full_integration_no_duplicates'],
            ),
            integration=cast(list[ProjectedData] | None, state['integration']),
            line_of_sight=None,
            fake_data=None,
            test_cube=None,
        )

"""
To store the global dataclasses used in the SDO projection code.
"""

# IMPORTs alias
import numpy as np

# IMPORTS sub
from dataclasses import dataclass, field

# IMPORTs personal
from .dataclasses_cubes import CubeInformation
from .dataclasses_sdo_image import PolarImageInfo
from .dataclasses_fit_n_envelopes import FitWithEnvelopes

# TYPE ANNOTATIONs
from typing import Literal



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


@dataclass(slots=True, repr=False, eq=False)
class WarpedInformation:
    """
    To store all the warped information data for each date.
    It also stores the corresponding polynomial fit angles.
    """

    # todo add the contours also in the integration ?

    # DATA
    warped_values: np.ndarray
    angles: np.ndarray

    # METADATA
    name: str
    date: str
    max_arc_length: int | float  # ? which type is it ?
    integration_type: Literal['mean', 'median'] = 'mean'

    def __post_init__(self) -> None:

        # INTEGRATION
        self.warped_integration()

    def warped_integration(self) -> None:
        """
        To compute the integration of the warped values.
        The values are saved back into the warped_values attribute.

        Raises:
            ValueError: if the integration type is not 'mean' or 'median'.
        """

        if self.integration_type == 'mean':
            self.warped_values = np.mean(self.warped_values.T, axis=0)
        elif self.integration_type == 'median':
            self.warped_values = np.median(self.warped_values.T, axis=0)
        else:
            raise ValueError(
                f"\033[1;31mUnknown integration type: {self.integration_type}. "
                "Choose between 'mean' and 'median'.\033[0m"
            )


@dataclass(slots=True, repr=False, eq=False)
class AllWarpedInformation:
    """
    To store the warped information for all the dates.
    """

    # ! make sure that the cadence is the same for all the dates or do a time interpolation later

    # METADATA
    name: str
    dates: list[str] = field(init=False)

    # DATA
    warped_informations: list[WarpedInformation]

    def __post_init__(self) -> None:

        # DATEs
        self.dates = self.get_dates()

    def get_dates(self) -> list[str]:
        """
        To get the dates of the warped information.

        Returns:
            list[str]: the dates of the warped information.
        """

        return [warped_information.date for warped_information in self.warped_informations]

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
    cube_index: int | None


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
    full_integration_all_data: ProjectedData | None = None
    full_integration_no_duplicates: ProjectedData | None = None
    integration: list[ProjectedData] | None = None
    line_of_sight: ProjectedData | None = None
    fake_data: ProjectedData | None = None
    test_cube: ProjectedData | None = None

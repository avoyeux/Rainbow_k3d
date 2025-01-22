"""
To save the dataclasses for the needed information in the animation.
This was created to make it easier to swap between the real and fake protuberance data.
"""

# IMPORTs
import sparse

# IMPORTs alias
import numpy as np

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
class InterpolationData:
    """
    Stores the interpolation results and some corresponding information.
    It is an immutable class without a __dict__ method (cf. dataclasses.dataclass).
    """

    # INFO
    name: str
    order: int
    color_hex: int

    # DATA
    coo: sparse.COO

    def __getitem__(self, index: int | slice) -> sparse.COO: return self.coo[index]


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
    coo: sparse.COO
    interpolations: list[InterpolationData] | None

    def __getitem__(self, index: int | slice) -> list[sparse.COO]:
        """
        To get a section of the protuberance and interpolations data.

        Args:
            index (int | slice): the time indexes needed to be fetched.

        Returns:
            list[sparse.COO]: list of the different data sections.
        """
        
        # ALL COORDs list
        result = [self.coo[index]]
        if self.interpolations is not None:
            result += [interpolation[index] for interpolation in self.interpolations]
        return result


@dataclass(slots=True, repr=False, eq=False)
class CubesData:
    """
    Stores all the data and information needed to name and position the solar protuberance for the
    k3d animation.
    This class doesn't have a __dict__() method (cf. dataclasses.dataclass).
    """

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

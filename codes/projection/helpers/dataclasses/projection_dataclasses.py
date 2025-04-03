"""
To store the dataclasses used in the polynomial projection module.
"""

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np

# IMPORTS sub
from typing import overload, Iterator, Literal
from dataclasses import dataclass, field



### GLOBALS ###


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class ImageBorders:
    """
    Class to store the image borders used in the polynomial projection module.
    """

    # BORDERs polar
    polar_angle: tuple[int, int]
    radial_distance: tuple[int, int]  # in km


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


### CARTESIAN TO POLAR ###


@dataclass(slots=True, repr=False, eq=False)
class PolarImageInfo:
    """
    To store the information gotten by creating the polar image.
    """

    # DATA
    image: np.ndarray
    sdo_pos: np.ndarray

    # IMAGE properties  # todo add plot information to the dataclass itself.
    colour: str
    resolution_km: float
    resolution_angle: float


@dataclass(slots=True, repr=False, eq=False)
class ImageInfo:
    """
    To store the information needed for the image processing.
    """

    # DATA
    image: np.ndarray
    sdo_pos: np.ndarray
    sun_radius: float  # ? should I take solar_r or RSUN_REF from the header?

    # IMAGE properties
    image_borders: ImageBorders
    sun_center: tuple[float, float]
    resolution_km: float

    # PLACEHOLDERs
    resolution_angle: float = field(init=False)
    max_index: float = field(init=False)

    def __post_init__(self) -> None:
        
        self.resolution_angle = 360 / (2 * np.pi * self.sun_radius / (self.resolution_km * 1e3))
        self.max_index = max(self.image_borders.radial_distance) * 1e3 / self.resolution_km



### POLYNOMIAL PROJECTION ###


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
class FitEnvelopes:
    """
    To format the result of the fit and envelope processing.
    """
    
    # METADATA
    order: int

    # COORDs polar
    polar_r: np.ndarray
    polar_theta: np.ndarray


@dataclass(slots=True, repr=False, eq=False)
class FitWithEnvelopes:
    """
    To format the results of the envelope processing.
    """

    # METADATA
    name: str
    colour: str
    fit_order: int

    # FIT processed
    fit_polar_r: np.ndarray
    fit_polar_theta: np.ndarray
    fit_angles: np.ndarray

    # ENVELOPEs
    envelopes: list[FitEnvelopes] | None

    # WARPED image
    warped_image: np.ndarray | None = field(default=None, init=False)


@dataclass(slots=True, repr=False, eq=False)
class ProjectedCube:

    # ? add xt_min, yt_min, zt_min ?
    # DATA
    data: np.ndarray

    # PLOT config
    name: str
    colour: str

    def __iter__(self) -> Iterator[np.ndarray]: return iter(self.data)  # ? why did I add this?

    # ? add a getitem dunder method ?





### EXTRACT ENVELOPE ###


@dataclass(slots=True, repr=False, eq=False)
class EnvelopeMiddleInformation:
    """
    To store the middle path of the envelope created by Dr. Auchere.
    """

    x_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_t: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __getitem__(self, item: int) -> np.ndarray:

        if item == 0: return self.x_t
        if item == 1: return self.y_t
        raise IndexError("Index out of range.")
    
    def __setitem__(self, key, value) -> None:

        if key == 0:
            self.x_t = value
        elif key == 1:
            self.y_t = value
        else:
            raise IndexError("Index out of range.")


@dataclass(slots=True, repr=False, eq=False)
class EnvelopeLimitInformation:
    """
    To store the upper and lower limits of the envelope created by Dr. Auchere.
    """

    x: np.ndarray = field(default_factory=lambda: np.empty(0))
    y: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __getitem__(self, item: int) -> np.ndarray:

        if item == 0: return self.x
        if item == 1: return self.y
        raise IndexError("Index out of range.")


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class EnvelopeInformation:
    """
    To store the envelope information created by Dr. Auchere.
    """

    upper: EnvelopeLimitInformation
    lower: EnvelopeLimitInformation
    middle: EnvelopeMiddleInformation

    @overload
    def __getitem__(self, item: Literal[0] | Literal[1]) -> EnvelopeLimitInformation: ...

    @overload
    def __getitem__(self, item: Literal[2]) -> EnvelopeMiddleInformation: ...

    @overload # fallback
    def __getitem__(self, item: int) -> EnvelopeLimitInformation | EnvelopeMiddleInformation: ...

    def __getitem__(self, item: int) -> EnvelopeLimitInformation | EnvelopeMiddleInformation:

        if item == 0: return self.upper
        if item == 1: return self.lower
        if item == 2: return self.middle
        raise IndexError("Index out of range.")

"""
To store the dataclasses used in the polynomial projection module.
"""

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np

# IMPORTS sub
from dataclasses import dataclass, field



### GLOBALS ###

@dataclass(slots=True, frozen=True, repr=False, eq=False)
class ImageBorders:
    """
    Class to store the image borders used in the polynomial projection module.
    """

    radial_distance: tuple[int, int]  # in km
    polar_angle: tuple[int, int]


@dataclass(slots=True, repr=False, eq=False)
class CubeInformation:
    """
    To store the information of a single cube.
    """

    xt_min : float
    yt_min : float
    zt_min : float

    coords: np.ndarray

    order: int | None = None  # for the polynomial



### CARTESIAN TO POLAR ###


@dataclass(slots=True, repr=False, eq=False)
class PolarImageInfo:
    """
    To store the information gotten by creating the polar image.
    """

    # DATA
    image: np.ndarray
    sdo_pos: np.ndarray

    # IMAGE properties
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

    def __post_init__(self):
        
        self.resolution_angle = 360 / (2 * np.pi * self.sun_radius / (self.resolution_km * 1e3))
        self.max_index = max(self.image_borders.radial_distance) * 1e3 / self.resolution_km



### POLYNOMIAL PROJECTION ###


@dataclass(slots=True, repr=False, eq=False)
class GlobalConstants:
    """
    Class to store the global constants used in the polynomial projection module.
    """

    dx: float
    solar_r: float
    time_indexes: np.ndarray
    dates: np.ndarray

    d_theta: float = field(init=False)

    def __post_init__(self):
        self.d_theta = 360 / (2 * np.pi * self.solar_r / self.dx)


@dataclass(slots=True, repr=False, eq=False)
class ProcessConstants:
    """
    Class to store the constants for each processes used in the polynomial projection module.
    """

    ID: int
    time_index: int
    date: str


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class CubesInformation:
    """
    Class to store the pointer to the data cubes used in the polynomial projection module.
    """

    xt_min : float
    yt_min : float
    zt_min : float

    pointer: h5py.Dataset

    def __getitem__(self, item: int) -> np.ndarray:
        data_filter = self.pointer[0, :] == item
        return self.pointer[1:, data_filter].astype('float64')


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class PolynomialInformation:
    """
    To store the polynomial information in polar coordinates and with the angle values.
    """

    order: int

    xt_min: float  # todo need to decide what to do with the recurent xt_min, yt_min, ...
    yt_min: float
    zt_min: float
    
    polar_r: np.ndarray
    polar_theta: np.ndarray
    angles: np.ndarray


@dataclass(slots=True, repr=False, eq=False)
class ProjectionData:
    """
    To store the data used in the polynomial projection module.
    """

    ID: int
    sdo_image: PolarImageInfo | None = None
    sdo_mask: PolarImageInfo | None = None
    integration: np.ndarray | None = None  # ? should I add the xt_min, yt_min, ... values
    cube: np.ndarray | None = None
    fits: list[PolynomialInformation] | None = None
    test_cube: np.ndarray | None = None



### EXTRACT ENVELOPE ###


@dataclass(slots=True, repr=False, eq=False)
class EnvelopeMiddleInformation:
    """
    To store the middle path of the envelope created by Dr. Auchere.
    """

    x_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    y_t: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __getitem__(self, item):

        if item == 0: return self.x_t
        if item == 1: return self.y_t
        raise IndexError("Index out of range.")
    
    def __setitem__(self, key, value):

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

    def __getitem__(self, item):

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

    def __getitem__(self, item):

        if item == 0: return self.upper
        if item == 1: return self.lower
        if item == 2: return self.middle
        raise IndexError("Index out of range.")



### GET POLYNOMIAL ###


@dataclass(slots=True, repr=False, eq=False)
class HDF5GroupPolynomialInformation:
    """
    To store the polynomial information stored in the HDF5 file.
    """

    HDF5Group: h5py.Group
    polynomial_order: int

    xt_min: float = field(init=False)
    yt_min: float = field(init=False)
    zt_min: float = field(init=False)

    coords: h5py.Dataset = field(init=False)

    def __post_init__(self):
        self.xt_min = self.HDF5Group['xt_min'][...]
        self.yt_min = self.HDF5Group['yt_min'][...]
        self.zt_min = self.HDF5Group['zt_min'][...]
        self.coords = self.HDF5Group[
            f'{self.polynomial_order}th order polynomial/parameters'
        ]#type: ignore

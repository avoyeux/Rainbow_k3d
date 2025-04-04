"""
To store dataclasses for the formatting of the data and metadata related to the processing of the
sdo image used inside the reprojection plot.
"""

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field



@dataclass(slots=True, frozen=True, repr=False, eq=False)
class ImageBorders:
    """
    Class to store the image borders used in the polynomial projection module.
    """

    # BORDERs polar
    polar_angle: tuple[int, int]
    radial_distance: tuple[int, int]  # in km


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

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



@dataclass(repr=False, eq=False)
class AnimationDefaultData:

    dx: float
    time_indexes: np.ndarray
    dates: list[str]
    xt_min: float
    yt_min: float
    zt_min: float

    def __post_init__(self):
        print('sun center created', flush=True)
        self.sun_center = - np.array([self.xt_min, self.yt_min, self.zt_min]) / self.dx


@dataclass(repr=False, eq=False)
class CubesData(AnimationDefaultData):

    # POS satellites
    sdo_pos: np.ndarray | None = None
    stereo_pos: np.ndarray | None = None

    # CUBES data
    all_data: sparse.COO | None = None
    no_duplicate: sparse.COO | None = None
    integration_all_data: sparse.COO | None = None
    integration_no_duplicate: sparse.COO | None = None
    los_sdo: sparse.COO | None = None
    los_stereo: sparse.COO | None = None

    # FAKE data
    sun_surface: sparse.COO | None = None
    fake_cube: sparse.COO | None = None
"""
To store dataclasses related to the polynomial fit of the protuberance data.
"""

# IMPORTs
import h5py

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import Any
from dataclasses import dataclass



@dataclass(slots=True, repr=False, eq=False)
class FitPointer:
    """
    Class to store the pointer to the polynomial fit parameters used in the polynomial projection
    module.
    """

    # METADATA
    fit_order: int
    integration_time: int

    # BORDERs
    xt_min: float
    yt_min: float
    zt_min: float

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

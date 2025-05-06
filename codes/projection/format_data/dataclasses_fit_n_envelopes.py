"""
To store the dataclasses related to the formatting of the fit and corresponding envelope
information.
"""

from __future__ import annotations

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field

# TYPE ANNOTATIONs
from typing import Literal, cast, Self

# API public
__all__ = [
    'WarpedInformation',
    'WarpedIntegration',
    'AllWarpedInformation',
    'FitEnvelopes',
    'EnvelopeInformation',
    'FitWithEnvelopes',
]


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
    fit_order: int
    arc_length: np.ndarray  # in km
    integration_type: Literal['mean', 'median'] = 'mean'

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

    def __add__(self, other: Self | WarpedIntegration) -> WarpedIntegration:
        """
        To add two WarpedInformation objects together. It creates a new WarpedIntegration
        object with the two WarpedInformation objects inside.

        Args:
            other (Self | WarpedIntegration): the other instance to add to the current one.

        Raises:
            TypeError: if the other instance is not a WarpedInformation or WarpedIntegration.

        Returns:
            WarpedIntegration: the new WarpedIntegration object with the two instances added
                together.
        """

        # CHECK
        if not self.check_adding(other):
            raise TypeError(
                f"\033[1;31mCannot add {type(self)} and {type(other)}.\033[0m"
            )

        if isinstance(other, type(self)):
            # ATTRIBUTEs
            dates = [self.date, other.date]
            warped_informations = cast(list[WarpedInformation], [self, other])
            arc_lengths = [self.arc_length, other.arc_length]
        elif isinstance(other, WarpedIntegration):
            # ATTRIBUTEs
            dates = [self.date] + other.dates
            warped_informations = [self] + other.warped_informations
            arc_lengths = [self.arc_length] + other.arc_lengths
        else:
            raise TypeError(f"\033[1;31mCannot add {type(self)} and {type(other)}.\033[0m")
        
        # ADD save
        instance = WarpedIntegration(
            name=f"warped integration",
            dates=dates,
            arc_lengths=arc_lengths,
            fit_order=self.fit_order,
            integration_type=self.integration_type,
            warped_informations=warped_informations,
        )
        return instance
    
    def __radd__(self, other: Self | WarpedIntegration) -> WarpedIntegration:
        return self.__add__(other)
    
    def check_adding(self, other: Self | WarpedIntegration) -> bool:
        """
        To check if the two objects can be added together.

        Args:
            other (Self | WarpedIntegration): the two instances to check.

        Returns:
            bool: True if the two instances can be added together, False otherwise.
        """

        check_fit_order = (self.fit_order == other.fit_order)
        check_integration_type = (self.integration_type == other.integration_type)
        return check_integration_type & check_fit_order


@dataclass(slots=True, repr=False, eq=False)
class WarpedIntegration:
    """
    To store the warped information for all the dates for a given dataset.
    """

    # todo __add__ and __radd__ to add WarpedIntegration instances together
    # ! make sure that the cadence is the same for all the dates or do a time interpolation later

    # METADATA
    name: str
    dates: list[str]
    fit_order: int
    integration_type: Literal['mean', 'median']

    # DATA
    arc_lengths: list[np.ndarray]
    warped_informations: list[WarpedInformation]  # * could change it to a sequence for subclassing

    # METADATA optional
    integration_time: int | None = field(default=None, init=False)


@dataclass(slots=True, repr=False, eq=False)
class AllWarpedInformation:
    """
    To store all the warped information of all the dates for all datasets.
    """

    # DATASETs
    integration: list[WarpedIntegration] | None  # todo check if the value can be None
    full_integration_no_duplicates: AllWarpedInformation | None


@dataclass(slots=True, repr=False, eq=False)
class FitEnvelopes:
    """
    To format the result of the fit and envelope processing.
    """
    
    # METADATA
    order: int

    # COORDs polar
    polar_r: np.ndarray = field(default_factory=lambda: np.empty(0))  # ? default needed ?
    polar_theta: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(slots=True, repr=False, eq=False)
class EnvelopeInformation:
    """
    To store the envelope information created by Dr. Auchere.
    """

    upper: FitEnvelopes
    lower: FitEnvelopes
    middle: FitEnvelopes


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
    envelopes: EnvelopeInformation | None

    # WARPED image
    warped_information: WarpedInformation | None = field(default=None, init=False)

    def __getstate__(self) -> dict[str, str | int | WarpedInformation | None]:
        """
        To pickle only what is needed. In my case, only the warping data is needed.

        Returns:
            dict[str, str | int | WarpedInformation | None]: the state of the object.
        """

        state = {  # ? should I keep the envelopes ?
            'name': self.name,
            'colour': self.colour,
            'fit_order': self.fit_order,
            'warped_information': self.warped_information,
        }
        return state
    
    def __setstate__(
            self,
            state: dict[str, str | int | np.ndarray | WarpedInformation | None],
        ) -> None:
        """
        To unpickle the object. Sets all the non-warping related attributes to None or empty array.

        Args:
            state (dict[str, str | int | WarpedInformation | None]): the state of the pickled
                object.
        """

        self.__init__(
            name=cast(str, state['name']),
            colour=cast(str, state['colour']),
            fit_order=cast(int, state['fit_order']),
            fit_polar_r=np.empty(0),
            fit_polar_theta=np.empty(0),
            fit_angles=np.empty(0),
            envelopes=None,
        )
        self.warped_information = cast(WarpedInformation | None, state['warped_information'])

"""
To store the dataclasses related to the formatting of the fit and corresponding envelope
information.
"""

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field

# IMPORTs personal
from codes.projection.format_data.dataclasses_warp import WarpedInformation

# TYPE ANNOTATIONs
from typing import cast

# API public
__all__ = [
    'FitEnvelopes',
    'EnvelopeInformation',
    'FitWithEnvelopes',
]



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

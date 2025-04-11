"""
To store the dataclasses related to the formatting of the fit and corresponding envelope
information.
"""

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field



@dataclass(slots=True, repr=False, eq=False)
class FitEnvelopes:
    """
    To format the result of the fit and envelope processing.
    """
    
    # METADATA
    order: int

    # COORDs polar
    polar_r: np.ndarray = field(default_factory=lambda: np.empty(0))  # ? is the default value needed
    polar_theta: np.ndarray = field(default_factory=lambda: np.empty(0))


@dataclass(slots=True, frozen=True, repr=False, eq=False)
class EnvelopeInformation:
    """
    To store the envelope information created by Dr. Auchere.
    """

    upper: FitEnvelopes
    lower: FitEnvelopes
    middle: FitEnvelopes

    def __getitem__(self, item: int) -> FitEnvelopes:

        if item == 0: return self.upper
        if item == 1: return self.lower
        if item == 2: return self.middle
        raise IndexError("Index out of range.")


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

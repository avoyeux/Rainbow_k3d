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

    # WARPED image
    warped_image: np.ndarray | None = field(default=None, init=False)


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

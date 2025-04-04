"""
To store the dataclasses related to the formatting of the fit and corresponding envelope
information.
"""

# IMPORTs alias
import numpy as np

# IMPORTs sub
from typing import overload, Literal
from dataclasses import dataclass, field



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


### FINAL REPROJECTION ###

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

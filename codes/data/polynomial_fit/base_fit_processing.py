"""
To store parent class(es) to process the fit and the corresponding envelopes to then be used in the
final plots.
"""

# IMPORTs alias
import numpy as np

# IMPORTs sub
from dataclasses import dataclass, field



@dataclass(slots=True, repr=False, eq=False)
class BaseFitProcessing:
    """
    Base to create and easily access the uniform coordinates for the fit and the envelope.
    """

    # DATA unprocessed
    polar_r: np.ndarray
    polar_theta: np.ndarray

    # PARAMETERs
    nb_of_points: int

    # PLACEHOLDERs
    cumulative_distance: np.ndarray = field(init=False)
    polar_r_normalised: np.ndarray = field(init=False)
    polar_theta_normalised: np.ndarray = field(init=False)

    def normalise_coords(self) -> None:
        """
        Normalise the coordinates so that they are between 0 and 1. As such, the cumulative
        distance won't only depend on one axis.
        """

        # COORDs
        coords = np.stack([self.polar_r, self.polar_theta], axis=0)

        # NORMALISE
        min_vals = np.min(coords, axis=1, keepdims=True)
        max_vals = np.max(coords, axis=1, keepdims=True)
        coords = (coords - min_vals) / (max_vals - min_vals)

        # COORDs update
        self.polar_r_normalised, self.polar_theta_normalised = coords
    
    def cumulative_distance_normalised(self) -> None:
        """
        To calculate the cumulative distance of the data and normalise it.
        """
        
        # COORDs
        coords = np.stack([self.polar_theta_normalised, self.polar_r_normalised], axis=0)

        # DISTANCE cumulative
        t = np.empty(coords.shape[1], dtype='float64')
        t[0] = 0
        for i in range(1, coords.shape[1]):
            t[i] = t[i - 1] + np.linalg.norm(coords[:, i] - coords[:, i - 1])
        t /= t[-1]  # normalise

        # DISTANCE update
        self.cumulative_distance = t

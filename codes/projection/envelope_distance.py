"""
To determine the 2D distance corresponding to the polar path of the fit or the fit envelope
depending on what is decided at the end.
"""

# IMPORTs alias
import numpy as np

# IMPORTs personal
from common import AnnotateAlongCurve
from codes.projection.format_data import FitEnvelopes



class EnvelopeDistanceAnnotation:
    """
    To annotate a curve with the arc-length along the curve itself.
    """

    def __init__(self, fit_envelope: FitEnvelopes, colour: str) -> None:

        # ATTRIBUTEs
        self.polar_r = fit_envelope.polar_r.astype('float64') * 1e3  # in m
        self.polar_theta = np.deg2rad(fit_envelope.polar_theta.astype('float64'))  # in rad

        # RUN
        fit_cartesian_coords = self.coords_to_cartesian()
        curve_distance = self.get_curve_distance(fit_cartesian_coords) / 1e3  # in Mm

        # ANNOTATE
        AnnotateAlongCurve(
            y=fit_envelope.polar_r.astype('float64') / 1e3,
            x=fit_envelope.polar_theta.astype('float64'),
            arc_length=curve_distance,
            step=50,
            offset=0.1,
            annotate_kwargs={
                'fontsize': 9,
                'color': colour,
                'alpha': 0.8,
            },
        )

    def coords_to_cartesian(self) -> np.ndarray:
        """
        To convert the polar coordinates to cartesian coordinates.

        Returns:
            np.ndarray: the corresponding cartesian coordinates.
        """
        
        # COORDs cartesian
        x = self.polar_r * np.cos(self.polar_theta)
        y = self.polar_r * np.sin(self.polar_theta)
        return np.stack([x, y], axis=0)
    
    def get_curve_distance(self, coords: np.ndarray) -> np.ndarray:
        """
        To compute the distance along the curve given the cartesian coordinates.

        Args:
            coords (np.ndarray): the cartesian coordinates of the curve.

        Returns:
            np.ndarray: the corresponding distance along the curve.
        """

        # COORDs km
        coords /= 1e3

        # DISTANCE curve path
        distance = np.empty((coords.shape[1],), dtype='float64')
        distance[0] = 0
        for i in range(1, distance.size):
            distance[i] = distance[i - 1] + np.sqrt(
                (coords[0, i] - coords[0, i - 1])**2 + (coords[1, i] - coords[1, i - 1])**2
            )
        return distance # in km
